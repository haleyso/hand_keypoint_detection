import torch


def quantize_to_fp8_e4m3(
    input: torch.Tensor,
) -> torch.Tensor:
    raw_bits = input.float().view(torch.int32)

    # Extract scale and fraction
    scale = ((raw_bits & 0x7f800000) >> 23) - 127
    fraction = raw_bits & 0x7fffff

    # Handle subnormal number
    subnormal = scale < -6
    shamt = -6 - scale
    fraction[subnormal] >>= shamt[subnormal]

    truncated_bits = raw_bits[subnormal] & ((1 << shamt[subnormal]) - 1)
    fraction[subnormal] |= (truncated_bits != 0).int()

    lb = (fraction & 0x100000).bool()
    gb = (fraction & 0x080000).bool()
    sb = (fraction & 0x07ffff).bool()
    rb = (lb & gb) | (gb & sb)

    fraction[rb] += 0x100000
    fraction &= 0xf00000
    fraction[subnormal] <<= shamt[subnormal]

    # scale = torch.clamp(scale, min=-9, max=8) # replace to not use torch.clamp
    mini = -9
    maxi = 8
    scale = torch.maximum(torch.minimum(scale, torch.tensor(maxi)), torch.tensor(mini))

    output = ((scale + 127) << 23) + fraction
    output = output.view(torch.float32) * torch.sign(input)

    # Saturate values that are out of range
    # output = torch.clamp(output, min=-448, max=448) # replace to not use torch.clamp
    mini = -448
    maxi = 448
    output = torch.maximum(torch.minimum(output, torch.tensor(maxi)), torch.tensor(mini))

    output[(input.abs() <= 2**-10) | (input == 0)] = 0
    output[~torch.isfinite(input)] = torch.nan

    return output.to(input.dtype)


def quantize_to_fp8_e5m2(
    input: torch.Tensor,
) -> torch.Tensor:
    raw_bits = input.float().view(torch.int32)

    # Extract scale and fraction
    scale = ((raw_bits & 0x7f800000) >> 23) - 127
    fraction = raw_bits & 0x7fffff

    # Handle subnormal number
    subnormal = scale < -14
    shamt = -14 - scale
    fraction[subnormal] >>= shamt[subnormal]

    # FIXME: add sticky bit for truncated bits

    lb = (fraction & 0x200000).bool()
    gb = (fraction & 0x100000).bool()
    sb = (fraction & 0x0fffff).bool()
    rb = (lb & gb) | (gb & sb)

    fraction[rb] += 0x200000
    fraction &= 0xe00000
    fraction[subnormal] <<= shamt[subnormal]

    # scale = torch.clamp(scale, min=-14, max=15) # replace to not use torch.clamp
    mini = -14
    maxi = 15
    scale = torch.maximum(torch.minimum(scale, torch.tensor(maxi)), torch.tensor(mini))
    output = (raw_bits & 0x80000000) + ((scale + 127) << 23) + fraction
    output = output.view(torch.float32)

    # output = torch.clamp(output, min=-61440, max=61440) # replace to not use torch.clamp
    mini = -61440
    maxi = 61440
    output = torch.maximum(torch.minimum(output, torch.tensor(maxi)), torch.tensor(mini))
    output[(input.abs() <= 2**-17) | (input == 0)] = 0
    output[~torch.isfinite(input)] = torch.nan

    return output.to(input.dtype)