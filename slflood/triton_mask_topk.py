"""
Triton replacement for:

    b_mask_indices = torch.argsort(b_mask, dim=1, descending=True, stable=True)[:, :K]
    or
    b_mask_indices = torch.topk(b_mask, k=b_mask, dim=1, sorted=False)[1]

Algorithm: classic prefix-sum-based stream compaction, done as a single
sequential pass per row over blocks of the N dimension, carrying a running
(true_count_so_far, implicit false_count_so_far) across blocks:

  For row-relative position j, let C[j] = number of True entries strictly
  before j (exclusive prefix sum of the mask up to j).
    - if mask[j] is True:  its destination column is C[j]
    - if mask[j] is False: its destination column is true_count + (j - C[j])
      where true_count is the row's TOTAL true count (known ahead of time -
      reuse the mask_counts you already compute before the batch loop,
      don't recompute here).

  Only destinations < K are ever written (True entries always land below
  K since true_count <= K by construction of K = max over the batch), so
  most False-entry writes for well-covered rows never happen at all.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_N": 2048}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit(do_not_specialize=["K"])
def _compact_kernel(
    mask_ptr, counts_ptr, out_ptr,
    stride_mb, stride_mn,
    stride_cb,
    stride_ob, stride_ok,
    N, K,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    true_count = tl.load(counts_ptr + row * stride_cb)

    carry_true = 0
    n_blocks = tl.cdiv(N, BLOCK_N)

    for blk in range(0, n_blocks):
        n_offsets = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        n_valid = n_offsets < N

        m = tl.load(
            mask_ptr + row * stride_mb + n_offsets * stride_mn,
            mask=n_valid, other=0,
        ).to(tl.int32)

        incl = tl.cumsum(m, axis=0)   # inclusive prefix sum within this block
        excl = incl - m               # exclusive prefix sum within this block

        global_true_rank = carry_true + excl                     # C[j], global
        global_false_rank = n_offsets - global_true_rank          # j - C[j], global

        dest = tl.where(m == 1, global_true_rank, true_count + global_false_rank)
        store_mask = n_valid & (dest < K)

        tl.store(out_ptr + row * stride_ob + dest * stride_ok, n_offsets, mask=store_mask)

        block_true_count = tl.sum(tl.where(n_valid, m, 0), axis=0)
        carry_true += block_true_count


def compact_mask_indices(mask: torch.Tensor, mask_counts: torch.Tensor, k: int) -> torch.Tensor:
    """
    Drop-in replacement for:
        b_mask_indices = torch.argsort(mask, dim=1, descending=True, stable=True)[:, :K]

    mask:        (rows, N) bool, CUDA
    mask_counts: (rows,)   int   - true count per row (mask.sum(1)).
                 Pass the values you've already computed for batch sorting
                 rather than recomputing them here - they're the same
                 numbers.
    k:           output width (b_mask_count_max for this batch)

    Returns (rows, K) int64 tensor of indices, matching stable
    descending-argsort semantics exactly: True positions in ascending
    original order first, then enough False positions (also ascending) to
    fill out to K columns.
    """
    assert mask.is_cuda
    rows, N = mask.shape

    mask_i32 = mask.to(torch.int32).contiguous()
    counts_i32 = mask_counts.to(torch.int32).contiguous()
    out = torch.empty((rows, k), device=mask.device, dtype=torch.int32)

    grid = (rows,)
    _compact_kernel[grid](
        mask_i32, counts_i32, out,
        mask_i32.stride(0), mask_i32.stride(1),
        counts_i32.stride(0),
        out.stride(0), out.stride(1),
        N, k,
    )
    return out.to(torch.int64)
