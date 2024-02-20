import math

import torch


def quantize_to_posit(
    input_tensor: torch.Tensor,
    nbits: int = 8,
    es: int = 1,
    return_pbits: bool = False,
    round_to_even: bool = True,
) -> torch.Tensor:
    # Convert input into 32-bit integer
    raw_bits = input_tensor.float().view(torch.int32)

    scale = ((raw_bits & 0x7f800000) >> 23) - 127
    r = scale >= 0
    run = torch.where(r, 1 + (scale >> es), -(scale >> es))

    regime = torch.where(r, (1 << (run + 1)) - 1, 0) ^ 1
    exponent = scale % (1 << es)
    fraction = raw_bits & 0x7fffff
    pt_bits = (regime << (23 + es)) | (exponent << 23) | fraction

    # Calculate if rounding is needed
    len = 2 + run + es + 23
    blast = (pt_bits & (1 << (len - nbits))).bool()
    bafter = (pt_bits & (1 << (len - nbits - 1))).bool()
    bsticky = (pt_bits << (32 - len + nbits + 1)).bool()
    rb = (blast & bafter) | (bafter & bsticky)

    max_scale = (nbits - 2) * (1 << es)
    inward_projection = torch.where(r, scale > max_scale, scale < -max_scale)
    rb &= ~inward_projection

    # Adjust exponent value
    es_bits = torch.clamp(nbits - 2 - run, min=0, max=es)
    scale -= exponent & ((1 << (es - es_bits)) - 1)
    scale = torch.clamp(scale, min=-max_scale, max=max_scale)

    # Mask out extra fraction bits
    nf_trunc = torch.clamp(len - nbits, min=0, max=23)
    bit_mask = torch.tensor(-1, dtype=torch.int32, device=input_tensor.device) << nf_trunc
    output = (fraction & bit_mask) | ((scale + 127) << 23)

    # Perform rounding
    output = torch.where(rb, output + (1 << (nf_trunc + es - es_bits)), output)
    output = output.view(torch.float32) * torch.sign(input_tensor)

    # Thresholding
    if round_to_even:
        min_scale = math.floor(-(nbits - 1) * (1 << es) + 2 ** (es - 1))
        output[input_tensor.abs() < 2 ** min_scale] = 0

    # Special values: 0 and NaN
    output[input_tensor == 0] = 0
    output[torch.isnan(input_tensor) | torch.isinf(input_tensor)] = float('nan')

    output.requires_grad = input_tensor.requires_grad
    output = output.to(input_tensor.dtype)

    if return_pbits:
        pt_bits = (pt_bits >> (len - nbits)) & ((1 << (nbits - 1)) - 1)
        pt_bits[rb] += 1
        pt_bits *= torch.sign(input_tensor).int()
        return output, pt_bits

    return output