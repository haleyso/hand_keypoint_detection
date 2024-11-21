import copy
import logging
import re
from collections import defaultdict
from typing import Dict, List, Any, Callable

import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
from torch.ao.quantization.utils import has_no_children_ignoring_parametrizations
from torch.nn import Module
from torch.nn.utils.parametrize import type_before_parametrizations

import peft
import torchvision
import transformers
from transformers.activations import GELUActivation
from transformers.models.mobilebert.modeling_mobilebert import NoNorm

import quantization.modules.bert as bert
import quantization.modules.mobilebert as mobilebert
import quantization.modules.qat as qat
import quantization.modules.quantizable as quantizable
from .modules.quantizable.activation import Matmul, FloatFunctional
from .modules.softmax import Softmax

from .fp8 import quantize_to_fp8_e4m3, quantize_to_fp8_e5m2
from .posit import quantize_to_posit
from .qconfig import QConfig


logger = logging.getLogger(__name__)

QCONFIG_PROPAGATE_MODULE_CLASS_LIST = {
    'gemm': (nn.Conv2d, nn.Conv3d, nn.Linear, Matmul),
    'act': (nn.ReLU, nn.GELU, GELUActivation),
    'norm': (nn.GroupNorm, nn.LayerNorm, NoNorm),
    'bn': (nn.BatchNorm2d, nn.BatchNorm3d),
    'pool': (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d),
    'softmax': (nn.Softmax,),
    'qk_scaling_product': (FloatFunctional,),
    'residual': (torchvision.models.resnet.BasicBlock, torchvision.models.resnet.Bottleneck),
}

DEFAULT_QAT_MODULE_MAPPINGS : Dict[Callable, Any] = {
    nn.Conv2d: qat.Conv2d,
    nn.Conv3d: qat.Conv3d,
    nn.Linear: qat.Linear,
    peft.tuners.lora.Linear: qat.LoraLinear,
    # Intrinsic modules:
    nni.ConvBn2d: qat.ConvBn2d,
}

SELF_ATTENTION_MODULE_MAPPINGS : Dict[Callable, Any] = {
    transformers.models.bert.modeling_bert.BertSelfAttention: quantizable.BertSelfAttention,
    transformers.models.mobilebert.modeling_mobilebert.MobileBertSelfAttention: quantizable.MobileBertSelfAttention,
    transformers.models.roberta.modeling_roberta.RobertaSelfAttention: quantizable.BertSelfAttention,
}

POSIT_EXP_UNOPT_FILE = "src/posit/quantization/posit_gold/posit16_1_exp_unopt.txt"
POSIT_EXP_OPT_FILE = "src/posit/quantization/posit_gold/posit16_1_exp.txt"
POSIT_RECIPROCAL_FILE = "src/posit/quantization/posit_gold/posit16_1_reciprocal.txt"


def _get_unique_devices_(mod):
    return {p.device for p in mod.parameters()} | \
        {p.device for p in mod.buffers()}


def has_trainable_params(mod):
    return any(p.requires_grad for p in mod.parameters())


def add_training_args(parser):
    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--bf16", action="store_true", help="Whether to use fp16 (mixed) precision instead of 32-bit")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU id to use.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="posit8_1",
        help="Quantization data type to use. Choose between posit(nbits)_(es), fp8_(e4m3|e5m2), and fp8.",
    )
    parser.add_argument(
        "--quant_max",
        type=float,
        default=16,
        help="Maximum exponent value of a data type when performing scaling."
    )
    parser.add_argument(
        "--amax_history_len",
        type=int,
        default=10,
        help="The length of the amax history window used for scaling factor computation."
    )
    parser.add_argument(
        "--quantize_forward",
        action="store_true",
        help="Whether to use 8-bit data type for activations.",
    )
    parser.add_argument(
        "--quantize_backward",
        action="store_true",
        help="Whether to use 8-bit data type for activation gradients.",
    )
    parser.add_argument(
        "--quantize_weights",
        action="store_true",
        help="Whether to use 8-bit data type for weights.",
    )
    parser.add_argument(
        "--quantized_ops_fwd",
        type=str,
        default="gemm",
        help=(
            "Quantize inputs (activations) to operation during forward pass."
            "Choose from gemm, act, norm, bn, softmax, qk_scaling_product, and residual."
        ),
    )
    parser.add_argument(
        "--quantized_ops_bwd",
        type=str,
        default="gemm",
        help=(
            "Quantize inputs (activation gradients) to operations during backward pass."
            "Choose from gemm, act, norm, bn, softmax, qk_scaling_product, and residual."
        ),
    )
    parser.add_argument(
        "--op_fusion",
        type=str,
        default=None,
        help=(
            'Fuse operations during forward pass to reduce quantization error. Choose from '
            'attention_scaling, softmax, residual, LayerNorm, and intermediate_act_fn.'
        ),
    )
    parser.add_argument(
        "--posit_exp",
        action="store_true",
        help="Whether to use posit approximated exponential."
    )
    parser.add_argument(
        "--posit_exp_opt",
        action="store_true",
        help="Whether to use optimized posit approximated exponential."
    )
    parser.add_argument(
        "--posit_reciprocal",
        action="store_true",
        help="Whether to use posit approximated reciprocal."
    )
    parser.add_argument(
        "--per_tensor_scaling",
        action="store_true",
        help="Whether to use 8-bit data type with per-tensor scaling for activation gradients.",
    )
    parser.add_argument(
        "--loss_scaling",
        action="store_true",
        help="Multiply losses with a constant to scale activation gradients to 8-bit representable range.",
    )
    parser.add_argument(
        "--gradient_threshold",
        type=float,
        default=256,
        help="Threshold value used to adjust loss scaling factor.",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=1,
        help=(
            "Perform weight splitting if greater than 1. Reduce weight update frequency"
            " and compute updated weights during forward and backward passes."
        ),
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Whether to use low rank adaptation."
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--qat",
        action="store_true",
        help="Whether to perform quantization aware training."
    )
    parser.add_argument(
        "--sgd",
        action="store_true",
        help="Whether to use SGD optimizer."
    )
    parser.add_argument(
        "--allow_bf16_reduced_precision_reduction",
        action="store_true",
        help="Whether to use reduced precision reduction."
    )
    parser.add_argument(
        "--print_loss",
        action="store_true",
        help="Whether to print loss during each step.",
    )

def get_qconfig(dtype: str, activation: bool, weight: bool, error: bool,
                device=None) -> QConfig:
    def _fake_quant(input: torch.Tensor, values: torch.Tensor):
        if not isinstance(input, torch.Tensor):
            return input

        if input.dtype == values.dtype == torch.bfloat16:
            indices = input.view(torch.int16).int() & 0xffff
            return values[indices]

        raw_bits = input.float().view(torch.int32)
        indices = ((raw_bits >> 16) & 0xffff) | (raw_bits & 0xffff != 0).int()
        return values[indices].to(input.dtype)

    def create_qconfig(quantized_values, error_quantized_values=None):
        class FakeQuantFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                logger.debug(f"FakeQuantFunction.apply {getattr(input, 'dtype', 'None')}")
                return _fake_quant(input, quantized_values)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, None
    
        if error_quantized_values is None:
            error_quantized_values = quantized_values

        return QConfig(
            activation=FakeQuantFunction.apply if activation else nn.Identity(),
            weight=FakeQuantFunction.apply if weight else nn.Identity(),
            error=(lambda x: _fake_quant(x, error_quantized_values)) if error else nn.Identity()
        )

    # TODO: Check hardware support for BF16
    input_tensor = torch.arange(2 ** 16, dtype=torch.int16, device=device).view(torch.bfloat16)

    if (match := re.match(r'posit(\d+)_(\d+)', dtype)):
        nbits, es = match.groups()
        values = quantize_to_posit(input_tensor, int(nbits), int(es), round_to_even=True)
        return create_qconfig(values)
    elif (match := re.match(r'(?:FP8.)?(E4M3|E5M2)', dtype)):
        fp8_format = match.group(1)
        values = (quantize_to_fp8_e4m3(input_tensor) if fp8_format == 'E4M3' else
                  quantize_to_fp8_e5m2(input_tensor))
        return create_qconfig(values)
    elif dtype == "FP8" or dtype == "FP8.MIXED":
        e4m3_values = quantize_to_fp8_e4m3(input_tensor)
        e5m2_values = quantize_to_fp8_e5m2(input_tensor)
        return create_qconfig(e4m3_values, e5m2_values)
    else:
        raise ValueError(f"Unrecognized quantization dtype: {dtype}")

def _observer_forward_hook(self, input, output):
    r"""Forward hook that calls observer on the output
    """
    return self.activation_post_process(output)

def _observer_forward_pre_hook(self, input):
    r"""Forward pre hook that calls observer on the input
    """
    return tuple(self.activation_post_process(x) for x in input)

def _register_activation_post_process_hook(module, pre_hook=False):
    assert hasattr(module, 'activation_post_process'), \
        'Expect activation_post_process attribute already attached to the module'
    if pre_hook:
        handle = module.register_forward_pre_hook(
            _observer_forward_pre_hook, prepend=True
        )
    else:
        handle = module.register_forward_hook(
            _observer_forward_hook, prepend=True
        )

def _add_observer_(module: Module, prefix="", forward_pre_hook_module_list=None,
                   backward_pre_hook_module_list=None, qconfig=None, device=None):
    if device is None:
        devices = _get_unique_devices_(module)
        assert len(devices) <= 1, (
            f"_add_observer_ only works with cpu or single-device CUDA modules, but got devices {devices}"
        )
        device = next(iter(devices)) if len(devices) > 0 else None

    def get_activation_post_process(qconfig, device):
        activation = qconfig.activation()
        if device is not None:
            activation.to(device)
        return activation

    def insert_activation_post_process(m, pre_hook=True):
        """ Adds an activation post process module and register
        a pre or post hook that calls the module
        """
        m.add_module('activation_post_process', get_activation_post_process(
            qconfig, device))
        # Register observer as the first entry in the hook list
        # All post forward hooks are preserved and will be executed after the observer before convert
        _register_activation_post_process_hook(m, pre_hook=pre_hook)

    for name, child in module.named_children():
        if not isinstance(child, nni._FusedModule):
            get_activation_post_process(child)
        else:
            _add_observer_(child, prefix + name, get_activation_post_process)

    if has_no_children_ignoring_parametrizations(module) and not isinstance(module, torch.nn.Sequential):
        get_activation_post_process(module)

def prepare(
    model: Module,
    qconfig: QConfig,
    quantized_ops_fwd=None,
    quantized_ops_bwd=None,
    per_tensor_scaling: bool = False,
    quant_max: int = 16,
    amax_history_len: int = 10,
    device=None,
):
    if device is None:
        devices = _get_unique_devices_(model)
        assert len(devices) <= 1, (
            f"_add_observer_ only works with cpu or single-device CUDA modules, but got devices {devices}"
        )
        device = next(iter(devices)) if len(devices) > 0 else None

    amax_buffer = defaultdict(lambda: torch.zeros((amax_history_len,), device=device))

    def quantize(x, fake_quant, tensor_key=None, is_per_tensor=False):
        if x is None:
            return None

        if is_per_tensor:
            x, amax_buffer[tensor_key] = quantize_per_tensor_max(
                x, amax_buffer[tensor_key], fake_quant, quant_max
            )
        else:
            x = fake_quant(x)
        return x

    def get_forward_pre_hook(name):
        logger.debug(f"Registering forward pre hook on: {name}")
        def forward_pre_hook(m, inputs):
            logger.debug(f"forward_pre_hook {name}")
            new_inputs = [quantize(x, qconfig.activation) for x in inputs]
            return tuple(new_inputs)
        return forward_pre_hook

    def get_forward_hook(name):
        logger.debug(f"Registering forward hook on: {name}")
        def forward_hook(m, inputs, output):
            logger.debug(f"forward_hook {name}")
            return quantize(output, qconfig.activation)
        return forward_hook

    def get_backward_pre_hook(name):
        logger.debug(f"Registering backward pre hook on: {name}")
        def backward_pre_hook(m, grad_outputs):
            logger.debug(f"backward_pre_hook {name}")
            tensor_key = f"{name}.grad_outputs"
            new_grad_outputs = [quantize(grad, qconfig.error, tensor_key, per_tensor_scaling) for grad in grad_outputs]
            return tuple(new_grad_outputs)
        return backward_pre_hook

    def get_backward_hook(name):
        logger.debug(f"Registering backward hook on: {name}")
        def backward_hook(m, grad_inputs, grad_outputs):
            logger.debug(f"backward_hook {name}")
            tensor_key = f"{name}.grad_inputs"
            new_grad_inputs = [quantize(grad, qconfig.error, tensor_key, per_tensor_scaling) for grad in grad_inputs]
            return tuple(new_grad_inputs)
        return backward_hook

    valid_ops = set(QCONFIG_PROPAGATE_MODULE_CLASS_LIST.keys())
    def parse_ops(ops, key):
        if ops is None or isinstance(getattr(qconfig, key), nn.Identity):
            return tuple()

        ops = {op.lower() for op in ops.split(',')}
        logger.info(f"Quantizing {key}: {ops}")

        invalid_ops = ops - valid_ops
        if invalid_ops:
            raise ValueError(f"Invalid operation(s) {', '.join(invalid_ops)}.")

        return tuple(mod for op in ops for mod in QCONFIG_PROPAGATE_MODULE_CLASS_LIST[op])

    forward_pre_hook_module_list = parse_ops(quantized_ops_fwd, "activation")
    backward_pre_hook_module_list = parse_ops(quantized_ops_bwd, "error")

    def insert_activation_post_process(m, name):
        if isinstance(m, forward_pre_hook_module_list):
            m.register_forward_pre_hook(get_forward_pre_hook(name))

        if isinstance(m, backward_pre_hook_module_list):
            m.register_full_backward_pre_hook(get_backward_pre_hook(name))

        # Checks for ResNet residuals
        resnet_residual = any(substr in name for substr in ["bn2", "bn3", "downsample"])

        # Checks for Bert residuals
        is_norm = isinstance(m, QCONFIG_PROPAGATE_MODULE_CLASS_LIST["norm"])
        is_linear = isinstance(m, nn.Linear)
        is_mha_or_ffn_output = is_linear and "output" in name and not "qa_output" in name
        is_mha_or_ffn_input = is_linear and any(layer in name for layer in [
            "query", "key", "value", "intermediate", "bottleneck.input", "bottleneck.attention"])

        if "residual" in quantized_ops_fwd and (resnet_residual or is_norm or is_mha_or_ffn_output):
            m.register_forward_hook(get_forward_hook(name))

        if "residual" in quantized_ops_bwd and (is_norm or is_mha_or_ffn_input):
            m.register_full_backward_hook(get_backward_hook(name))

    def add_observer(module: Module, prefix="", insert_activation_post_process=None):
        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nni._FusedModule):
                insert_activation_post_process(child, child_prefix)
            else:
                add_observer(child, child_prefix, insert_activation_post_process)

        insert_activation_post_process(module, prefix)

    add_observer(model, insert_activation_post_process=insert_activation_post_process)
    return model

def quantize_per_tensor_max(
    input: torch.Tensor,
    amax_history: torch.Tensor,
    fake_quant: Callable,
    quant_max: int = 16,
) -> torch.Tensor:
    amax = torch.amax(amax_history)

    exp = torch.floor(torch.log2(quant_max / amax))
    # sf = torch.pow(2, torch.clamp(exp, min=0)) # replace torch.clamp and torch.pow
    mini = 0
    sf = torch.square(torch.maximum(exp, torch.tensor(mini)))
    sf = torch.where(amax > 0.0, sf, 1.0)
    sf = torch.where(torch.isfinite(amax), sf, 1.0)
    input = fake_quant(input * sf) / sf

    amax_history = torch.roll(amax_history, -1, 0)
    amax_history[0] = torch.amax(torch.abs(input))

    return input, amax_history

def get_quantized_model(
    model: Module,
    qconfig: QConfig,
    op_fusion=None,
    per_tensor_scaling: bool = False,
    quant_max: int = 4,
    device=None,
) -> Module:
    fused_ops = []
    if op_fusion is not None:
        logger.info(f"Fusing operations: {op_fusion}")
        fused_ops = op_fusion.replace(" ", "").split(',')

    if device is None:
        devices = _get_unique_devices_(model)
        assert len(devices) <= 1, (
            f"_add_observer_ only works with cpu or single-device CUDA modules, but got devices {devices}"
        )
        device = next(iter(devices)) if len(devices) > 0 else None

    amax_buffer = defaultdict(lambda: torch.zeros((5,), device=device))

    class Quantizer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, layer=None):
            ctx.layer = layer
            if not any(x in layer for x in fused_ops):
                input = qconfig.activation(input)
            return input

        @staticmethod
        def backward(ctx, grad_output):
            layer = ctx.layer
            if per_tensor_scaling:
                grad_output, amax_buffer[layer] = quantize_per_tensor_max(
                    grad_output, amax_buffer[layer], qconfig.error, quant_max
                )
            else:
                grad_output = qconfig.error(grad_output)
            return grad_output, None

    model_name = type(model).__name__
    model_type = model_name.split("For", 1)[0]
    assert model_type in ("MobileBert", "Bert"), (
        f"'{model_type}' models are not support for quantization."
    )

    module = bert if model_type == "Bert" else mobilebert
    model_cls = getattr(module, model_name)
    quantized_model = model_cls(model.config, Quantizer.apply)
    quantized_model.load_state_dict(model.state_dict())
    quantized_model.to(device)

    return quantized_model

def quantize_qat(module: Module, qconfig=None):
    assert qconfig, "qconfig must be provided for quantization aware training."
    _convert(module, inplace=True, qconfig=qconfig)
    return module

def _convert(
        module, mapping=None, inplace=False,
        convert_custom_config_dict=None, qconfig=None):
    r"""Converts submodules in input mod to a different mod according to `mapping`
    by calling `from_float` method on the target mod class

    Args:
        mod: input mod
        mapping: a dictionary that maps from source mod type to target
                 mod type, can be overwritten to allow swapping user defined
                 Modules
        inplace: carry out model transformations in-place, the original mod
                 is mutated
        is_reference: a flag to enable quantized reference mod

    """
    if mapping is None:
        mapping = DEFAULT_QAT_MODULE_MAPPINGS
    if convert_custom_config_dict is None:
        convert_custom_config_dict = {}
    custom_module_class_mapping = convert_custom_config_dict.get("observed_to_quantized_custom_module_class", {})

    if not inplace:
        module = copy.deepcopy(module)
    reassign = {}
    for name, mod in module.named_children():
        if not isinstance(mod, nni._FusedModule) and \
           type_before_parametrizations(mod) not in custom_module_class_mapping:
            _convert(mod, mapping, True, convert_custom_config_dict, qconfig)
        reassign[name] = swap_module(mod, mapping, custom_module_class_mapping, qconfig)

    for key, value in reassign.items():
        module._modules[key] = value

    return module

def swap_module(mod, mapping, custom_module_class_mapping, qconfig):
    r"""Swaps the mod if it has a quantized counterpart and it has an
    `observer` attached.

    Args:
        mod: input mod
        mapping: a dictionary that maps from nn mod to nnq mod

    Return:
        The corresponding quantized mod of `mod`
    """
    new_mod = mod
    if qconfig is not None:
        swapped = False
        if type_before_parametrizations(mod) in custom_module_class_mapping:
            new_mod = custom_module_class_mapping[type_before_parametrizations(mod)].from_observed(mod)
            swapped = True
        elif type_before_parametrizations(mod) in mapping:
            new_mod = mapping[type_before_parametrizations(mod)].from_float(mod, qconfig)
            swapped = True
        elif has_trainable_params(mod) and has_no_children_ignoring_parametrizations(mod):
            logger.warn(f"Module {type_before_parametrizations(mod)} has trainable parameters, "
                        "but does not have a quantized counterpart.")

        if swapped:
            # Preserve mod's forward and backward hooks.
            for pre_hook_fn in mod._forward_pre_hooks.values():
                new_mod.register_forward_pre_hook(pre_hook_fn)
            for hook_fn in mod._forward_hooks.values():
                new_mod.register_forward_hook(hook_fn)
            for pre_hook_fn in mod._backward_pre_hooks.values():
                new_mod.register_full_backward_pre_hook(pre_hook_fn)
            for hook_fn in mod._backward_hooks.values():
                new_mod.register_full_backward_hook(hook_fn)

            # respect device affinity when swapping modules
            devices = _get_unique_devices_(mod)
            assert len(devices) <= 1, (
                f"swap_module only works with cpu or single-device CUDA modules, but got devices {devices}"
            )
            device = next(iter(devices)) if len(devices) > 0 else None
            if device:
                new_mod.to(device)
    return new_mod

def replace_self_attention(module, config):
    reassign = {}
    for name, mod in module.named_children():
        if type_before_parametrizations(mod) not in SELF_ATTENTION_MODULE_MAPPINGS:
            replace_self_attention(mod, config)
        else:
            new_mod = SELF_ATTENTION_MODULE_MAPPINGS[type_before_parametrizations(mod)](config)
            for key, child in mod.named_children():
                new_mod._modules[key] = child
            reassign[name] = new_mod

    for key, value in reassign.items():
        module._modules[key] = value

    return module

def replace_softmax(
        model: Module, posit_exp: bool, posit_exp_opt: bool,
        posit_reciprocal: bool, dtype=None, device=None):
    """
    This function requires torch.nn.Softmax mod to be used.
    """
    if device is None:
        devices = _get_unique_devices_(model)
        assert len(devices) <= 1, (
            f"_add_observer_ only works with cpu or single-device CUDA modules, but got devices {devices}"
        )
        device = next(iter(devices)) if len(devices) > 0 else None

    def read_file(filepath: str):
        with open(filepath, 'r') as file:
            values = [float.fromhex(line.rstrip()) for line in file]
        return torch.tensor(values, dtype=dtype, device=device)

    posit_exps = None
    if posit_exp:
        logger.info("Using unoptimized posit exp in softmax.")
        posit_exps = read_file(POSIT_EXP_UNOPT_FILE)
    elif posit_exp_opt:
        logger.info("Using optimized posit exp in softmax.")
        posit_exps = read_file(POSIT_EXP_OPT_FILE)

    posit_reciprocals = None
    if posit_reciprocal:
        logger.info("Using posit reciprocal in softmax.")
        posit_reciprocals = read_file(POSIT_RECIPROCAL_FILE)

    for name, mod in model.named_modules():
        if type_before_parametrizations(mod) == nn.Softmax:
            new_mod = Softmax(posit_exps, posit_reciprocals, dim=-1)
            parent_name, target_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, target_name, new_mod)

def get_fused_modules(model: nn.Module, modules_to_fuse: List):
    module_list = []
    fused_module_list = []
    index = 0

    for name, mod in model.named_modules():
        if type_before_parametrizations(mod) != modules_to_fuse[index]:
            module_list = []
            index = 0

        if type_before_parametrizations(mod) == modules_to_fuse[index]:
            module_list.append(name)
            index += 1
            if index == len(modules_to_fuse):
                fused_module_list.append(module_list)
                module_list = []
                index = 0

    return fused_module_list