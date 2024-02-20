import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import type_before_parametrizations

import peft.tuners.lora as lora
from peft.utils.other import transpose

__all__ = [
    "Linear"
]

class Linear(lora.Linear):
    _FLOAT_MODULE = lora.Linear

    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        qconfig=None,
        **kwargs,
    ):
        super().__init__(adapter_name, in_features, out_features, r, lora_alpha,
                         lora_dropout, fan_in_fan_out, is_target_conv_1d_layer, **kwargs)
        assert qconfig, 'quantizer must be provided for QAT module'
        self.weight_fake_quant = qconfig.weight
        
    def forward(self, x: torch.Tensor):
        if self.active_adapter not in self.lora_A.keys():
            return self._linear(x)

        previous_dtype = x.dtype

        if self.disable_adapters:
            if (self.r[self.active_adapter] > 0) and self.merged:
                self.unmerge()
            result = self._linear(x)
        elif (self.r[self.active_adapter] == 0) or self.merged:
            result = self._linear(x)
        else:
            # lora_A = self.lora_A[self.active_adapter]
            # lora_B = self.lora_B[self.active_adapter]
            # dropout = self.lora_dropout[self.active_adapter]
            # scaling = self.scaling[self.active_adapter]

            # result = self._linear(x)
            # x = x.to(lora_A.weight.dtype)
            # result += lora_B(lora_A(dropout(x))) * scaling

            lora_A = self.weight_fake_quant(self.lora_A[self.active_adapter].weight)
            lora_B = self.weight_fake_quant(self.lora_B[self.active_adapter].weight)
            scaling = self.scaling[self.active_adapter]
            merged_weight = transpose(self.weight, self.fan_in_fan_out) + lora_B @ lora_A * scaling
            result = F.linear(x, self.weight_fake_quant(merged_weight), bias=self.bias)

        result = result.to(previous_dtype)
        return result

    @classmethod
    def from_float(cls, mod, qconfig):
        assert type_before_parametrizations(mod) == cls._FLOAT_MODULE, (
            " qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )

        if mod.merged:
            mod.unmerge()

        adaptor = mod.active_adapter
        qat_linear = cls(adaptor, mod.in_features, mod.out_features, bias=mod.bias is not None,
                         r=mod.r[adaptor], lora_alpha=mod.lora_alpha[adaptor],
                         fan_in_fan_out=mod.fan_in_fan_out, init_lora_weights=False, qconfig=qconfig)
        qat_linear.weight = mod.weight
        qat_linear.bias = mod.bias
        # copy ModuleDict
        qat_linear.lora_dropout = copy.deepcopy(mod.lora_dropout)
        qat_linear.lora_A = copy.deepcopy(mod.lora_A)
        qat_linear.lora_B = copy.deepcopy(mod.lora_B)
        return qat_linear

    @classmethod
    def to_float():
        raise NotImplementedError
