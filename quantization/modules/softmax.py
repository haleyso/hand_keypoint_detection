from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function


def _convert(input: torch.Tensor, values: torch.Tensor):
    if input.dtype == values.dtype == torch.bfloat16:
        indices = (input.view(torch.int16).int() << 7) & 0x3fffff
        return values[indices]

    raw_bits = input.float().view(torch.int32)
    indices = ((raw_bits >> 9) & 0x3fffff) | ((raw_bits & 0x1ff) != 0).int()
    return values[indices].to(input.dtype)


class SoftmaxFunction(Function):
    @staticmethod
    def forward(ctx, i, dim, posit_exps=None, posit_reciprocals=None):
        # i = i - torch.max(i, dim=dim, keepdim=True)[0]

        if posit_exps is None:
            exp_x = torch.exp(i)
        else:
            exp_x = _convert(i, posit_exps)

        exp_x_sum = torch.sum(exp_x, dim=dim, keepdim=True)

        if posit_reciprocals is None:
            output = exp_x / exp_x_sum
            ctx.save_for_backward(output, None, None)
        else:
            output = exp_x * _convert(exp_x_sum, posit_reciprocals)
            ctx.save_for_backward(output, exp_x, exp_x_sum)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, exp_x, exp_x_sum = ctx.saved_tensors

        if exp_x is None:
            grad_input = output * grad_output
            sum_grad = torch.sum(grad_input, dim=-1, keepdims=True)
            grad_input -= output * sum_grad
        else:
            # Softmax gradient with posit reciprocal approximation
            grad_input = output * grad_output
            sum_grad = torch.sum(exp_x * grad_output, dim=-1, keepdims=True)
            deriv = torch.pow(2, torch.floor(torch.log2(exp_x_sum)) * -2 - 1)
            grad_input -= deriv * exp_x * sum_grad

            # deriv = -torch.pow(2, torch.floor(torch.log2(exp_x_sum)) * -2 - 1)
            # is_max = output == output.amax(dim=-1, keepdims=True)

            # grad_input = output * grad_output
            # sum_grad = torch.sum(exp_x * deriv * grad_output, dim=-1, keepdims=True)
            # grad_input = torch.where(~is_max, grad_input + sum_grad * exp_x, grad_input)

            # col_grad = -output + exp_x * deriv * (exp_x.amax() - exp_x_sum)
            # sum_grad = torch.sum(col_grad * grad_output, dim=-1, keepdims=True)
            # grad_input = torch.where(is_max, grad_input + sum_grad, grad_input)

        return grad_input, None, None, None


class Softmax(nn.Softmax):
    def __init__(
        self,
        posit_exps: Optional[Tensor] = None,
        posit_reciprocals: Optional[Tensor] = None,
        dim: Optional[int] = None
    ) -> None:
        super().__init__(dim)
        self.posit_exps = posit_exps
        self.posit_reciprocals = posit_reciprocals

    def forward(self, input: Tensor) -> Tensor:
        input = input - torch.amax(input, dim=self.dim, keepdim=True)
        return SoftmaxFunction.apply(
            input, self.dim, self.posit_exps, self.posit_reciprocals
        )