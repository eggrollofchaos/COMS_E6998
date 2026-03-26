#!/usr/bin/env python3
#
# Wei Alexander Xin - wax1
#
# C4: Convolution using OpenAI Triton
# Script version of c4.ipynb for running on Insomnia

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Global Variables
DEVICE = "cuda"
C_OUT = 64
C_IN = 3
H = 1024
W = 1024
FH = 3
FW = 3

# Making torch tensors
tensor_I = torch.rand(1, C_IN, H, W, device=DEVICE)
tensor_F = torch.rand(C_OUT, C_IN, FH, FW, device=DEVICE)

# Golden output from PyTorch for correctness check
golden_out = F.conv2d(tensor_I, tensor_F, padding=1)
print(f"Output shape: {golden_out.shape}")


@triton.jit
def my_triton_kernel(
    input_ptr,
    kernel_ptr,
    output_ptr,
    # Tensor dimensions + padding
    C_IN: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C_OUT: tl.constexpr,
    FH: tl.constexpr,
    FW: tl.constexpr,
    PAD: tl.constexpr,
):
    # Which output element are we computing??
    k = tl.program_id(0)  # output channel
    out_x = tl.program_id(1)  # output row
    out_y = tl.program_id(2)  # output col

    # Declare accumulator
    acc = tl.zeros([], dtype=tl.float32)

    # Iterate... a lot, over input channels and filter taps
    for c in range(C_IN):
        for fh in range(FH):
            for fw in range(FW):
                in_x = out_x + fh - PAD
                in_y = out_y + fw - PAD

                in_bounds = (in_x >= 0) & (in_x < H) & (in_y >= 0) & (in_y < W)

                # input layout: (1, C_IN, H, W)
                # offset = c*H*W + in_x*W + in_y
                input_offset = c * H * W + in_x * W + in_y
                input_val = tl.load(input_ptr + input_offset, mask=in_bounds, other=0.0)

                # kernel layout: (C_OUT, C_IN, FH, FW)
                # offset = k*C_IN*FH*FW + c*FH*FW + fh*FW + fw
                kernel_offset = k * C_IN * FH * FW + c * FH * FW + fh * FW + fw
                kernel_val = tl.load(kernel_ptr + kernel_offset)

                acc += input_val * kernel_val

    # output layout: (1, C_OUT, H, W)
    # offset = k*H*W + out_x*W + out_y
    output_offset = k * H * W + out_x * W + out_y
    tl.store(output_ptr + output_offset, acc)


def my_conv2d(input, kernel):
    _, C_IN, H, W = input.shape
    C_OUT, _, FH, FW = kernel.shape
    PAD = 1

    OUT_H = H
    OUT_W = W
    output = torch.empty(1, C_OUT, OUT_H, OUT_W, device=input.device, dtype=input.dtype)

    grid = (C_OUT, OUT_H, OUT_W)

    # Warmup
    my_triton_kernel[grid](
        input,
        kernel,
        output,
        C_IN,
        H,
        W,
        C_OUT,
        FH,
        FW,
        PAD,
    )
    torch.cuda.synchronize()

    # Time it!
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    my_triton_kernel[grid](
        input,
        kernel,
        output,
        C_IN,
        H,
        W,
        C_OUT,
        FH,
        FW,
        PAD,
    )
    end.record()
    torch.cuda.synchronize()

    execution_time = start.elapsed_time(end)
    return output, execution_time


my_output, execution_time = my_conv2d(tensor_I, tensor_F)
torch.testing.assert_close(golden_out, my_output)
print(f"Correctness: PASSED")
print(f"%.3f" % execution_time)
