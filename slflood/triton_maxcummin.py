"""
Triton replacement for:

    inter          = torch.cdist(b_mask_points, b_sample)     # (B, M, G) - materialized
    cummin         = inter.cummin(dim=1)[0]                   # (B, M, G) - materialized
    max_diameters  = torch.amax(cummin, dim=2)                # (B, M)    - only this survives

Algorithm (online scan, same structural pattern as flash-attention's
online softmax):

  For each batch row b and grid point g, maintain a running minimum
  distance as witness points are added in order (m = 0, 1, 2, ...).
  After adding point m, the "covering radius" at that prefix length is
  max_g(running_min[g]). Only that (B, M) result is ever needed downstream
  so the (B, M, G) intermediate never needs to exist in memory.

  Grid points are split into blocks of BLOCK_G for parallelism (so a run
  isn't limited to B independent programs, which would badly underuse a
  GPU with far more SMs than typical batch sizes). Each (batch, g-block)
  program computes its own partial max per m-step; a final cheap
  torch.amax over the small g-block axis combines them.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_G": 256}, num_warps=4),
        triton.Config({"BLOCK_G": 512}, num_warps=4),
        triton.Config({"BLOCK_G": 512}, num_warps=8),
        triton.Config({"BLOCK_G": 1024}, num_warps=8),
        triton.Config({"BLOCK_G": 2048}, num_warps=8),
    ],
    key=["G", "D"],
)
@triton.jit(do_not_specialize=["M", "G"])
def _flood_scan_kernel(
    points_ptr, sample_ptr, out_ptr,
    stride_pb, stride_pm, stride_pd,
    stride_sb, stride_sg, stride_sd,
    stride_ob, stride_om, stride_ogb,
    M, G,
    D: tl.constexpr,
    D_PADDED: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    g_offsets = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    g_mask = g_offsets < G

    d_offsets = tl.arange(0, D_PADDED)
    d_mask = d_offsets < D

    # Load this program's slice of sample points once: (BLOCK_G, D_PADDED)
    sample_ptrs = (
        sample_ptr
        + pid_b * stride_sb
        + g_offsets[:, None] * stride_sg
        + d_offsets[None, :] * stride_sd
    )
    load_mask = g_mask[:, None] & d_mask[None, :]
    sample_block = tl.load(sample_ptrs, mask=load_mask, other=0.0)

    running_min = tl.full([BLOCK_G], float("inf"), dtype=tl.float32)
    out_base = out_ptr + pid_b * stride_ob + pid_g * stride_ogb

    # Sequential scan over M - inherent to the cumulative-min structure.
    # M is a runtime value (not tl.constexpr), so this compiles to an
    # actual device-side loop rather than being unrolled at compile time
    # (same pattern as the K-reduction loop in Triton's matmul tutorial).
    for i in range(0, M):
        point_ptrs = (
            points_ptr
            + pid_b * stride_pb
            + i * stride_pm
            + d_offsets * stride_pd
        )
        point_i = tl.load(point_ptrs, mask=d_mask, other=0.0)  # (D_PADDED,)

        diff = sample_block - point_i[None, :]
        sqdist = tl.sum(diff * diff, axis=1)  # (BLOCK_G,) - padded dims contribute 0
        dist = tl.sqrt(sqdist)

        running_min = tl.minimum(running_min, dist)

        masked_min = tl.where(g_mask, running_min, float("-inf"))
        local_max = tl.max(masked_min, axis=0)

        tl.store(out_base + i * stride_om, local_max)


def maxcummin(b_mask_points: torch.Tensor, b_sample: torch.Tensor) -> torch.Tensor:
    """
    Drop-in replacement for:
        inter = torch.cdist(b_mask_points, b_sample)
        cummin = inter.cummin(dim=1)[0]
        max_diameters = torch.amax(cummin, dim=2)

    b_mask_points: (B, M, D) float32, CUDA
    b_sample:      (B, G, D) float32, CUDA
    returns:       (B, M)    float32
    """
    assert b_mask_points.is_cuda and b_sample.is_cuda, "kernel is CUDA-only"
    assert b_mask_points.dtype == torch.float32 and b_sample.dtype == torch.float32
    B, M, D = b_mask_points.shape
    B2, G, D2 = b_sample.shape
    assert B == B2 and D == D2, "batch/dim mismatch between points and sample"

    b_mask_points = b_mask_points.contiguous()
    b_sample = b_sample.contiguous()

    D_PADDED = triton.next_power_of_2(D)

    def grid(meta):
        return (B, triton.cdiv(G, meta["BLOCK_G"]))

    # Sized for the smallest BLOCK_G in the autotune list so the buffer
    # shape doesn't depend on which config gets picked; unused trailing
    # columns stay -inf and are ignored by the final amax.
    smallest_block_g = 256
    max_g_blocks = triton.cdiv(G, smallest_block_g)
    partial = torch.full(
        (B, M, max_g_blocks), float("-inf"),
        device=b_mask_points.device, dtype=torch.float32,
    )

    _flood_scan_kernel[grid](
        b_mask_points, b_sample, partial,
        b_mask_points.stride(0), b_mask_points.stride(1), b_mask_points.stride(2),
        b_sample.stride(0), b_sample.stride(1), b_sample.stride(2),
        partial.stride(0), partial.stride(1), partial.stride(2),
        M, G,
        D=D, D_PADDED=D_PADDED,
    )

    return partial.amax(dim=2)
