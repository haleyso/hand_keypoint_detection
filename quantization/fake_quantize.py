from functools import partial
import re
from typing import Union, Callable, Literal

import torch
import torch.nn as nn


class _PartialWrapper:
    def __init__(self, p):
        self.p = p
        self.callable_args = {}

    def __call__(self, *args, **keywords):
        # call each arg in callable_args and add them partial, then run with keywords
        # skip if arg_name in keywords so its possible to overwrite
        for arg_name in self.callable_args:
            if arg_name not in keywords:
                keywords = {**keywords, **{arg_name: self.callable_args[arg_name]()}}
        return self.p(*args, **keywords)

    def __repr__(self):
        return self.p.__repr__() + self.callable_args.__repr__()

    def with_args(self, **kwargs):
        return _with_args(self, **kwargs)

    def with_callable_args(self, **kwargs):
        result = _PartialWrapper(p=self.p)
        result.callable_args = {**self.callable_args, **kwargs}
        return result

def _with_args(cls_or_self, **kwargs):
    r"""Wrapper that allows creation of class factories.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances. Can be used in conjunction with
    _callable_args

    Example::

        >>> # xdoctest: +SKIP("Undefined vars")
        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
        >>> foo_instance1 = foo_builder()
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1) == id(foo_instance2)
        False
    """
    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r

def _with_callable_args(cls_or_self, **kwargs):
    r"""Wrapper that allows creation of class factories args that need to be
    called at construction time.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances and those arguments should only
    be calculated at construction time. Can be used in conjunction with _with_args

    Example::

        >>> # xdoctest: +SKIP("Undefined vars")
        >>> Foo.with_callable_args = classmethod(_with_callable_args)
        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_callable_args(cur_time=get_time_func).with_args(name="dan")
        >>> foo_instance1 = foo_builder()
        >>> # wait 50
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1.creation_time) == id(foo_instance2.creation_time)
        False
    """
    r = _PartialWrapper(partial(cls_or_self))
    return r.with_callable_args(**kwargs)


class FusedAmaxObsFakeQuantize(nn.Module):
    r"""Observer module for computing the quantization parameters based on the
    historical amax values.

    This observer uses the tensor min/max statistics to compute the quantization
    parameters. The module records the running minimum and maximum of incoming
    tensors, and uses this statistic to compute the quantization parameters.
    """
    amax_history: torch.Tensor

    def __init__(
        self,
        quant_max: float = 256,
        is_per_tensor: bool = False,
        amax_history_len: int = 5,
        amax_compute_algo: Union[Literal["max", "most_recent"], Callable] = "max",
        fake_quant=None,
        factory_kwargs=None
    ) -> None:
        super().__init__()
        self.quant_max = quant_max
        self.is_per_tensor = is_per_tensor
        self.register_buffer("amax_history", torch.zeros((amax_history_len,)))
        self.fake_quant = fake_quant

    def forward(self, x_orig):
        r"""Records the running amax of ``x``."""
        if x_orig.numel() == 0:
            return x_orig

        if self.is_per_tensor:
            amax = torch.amax(self.amax_history)

            exp = torch.floor(torch.log2(self.quant_max / amax))
            # sf = torch.pow(2, torch.clamp(exp, min=0)) # replace to not use torch.clamp and torch.pow
            mini = 0
            sf = torch.pow(2, torch.maximum(exp, torch.tensor(mini)))   
            sf = torch.where(amax > 0.0, sf, 1.0)
            sf = torch.where(torch.isfinite(amax), sf, 1.0)
            x = self.fake_quant(x_orig * sf) / sf
        else:
            x = self.fake_quant(x_orig)

        amax_history = torch.roll(self.amax_history, -1, 0)
        amax_history[0] = torch.amax(torch.abs(x))
        self.amax_history = amax_history

        return x

    @torch.jit.export
    def extra_repr(self):
        return f"amax_history={self.amax_history}"

    @torch.jit.export
    def reset_amax_history(self):
        """Resets the amax history values."""
        self.amax_history.zero_()

    with_args = classmethod(_with_args)
    with_callable_args = classmethod(_with_callable_args)