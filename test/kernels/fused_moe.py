# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.tile as ct
import torch
import torch.nn.functional as F

from kernels.matmul import swizzle_2d

ConstInt = ct.Constant[int]


@ct.kernel
def fused_moe_kernel(
    A,
    B,
    C,
    topk_weights,
    sorted_token_ids,
    sorted_expert_ids,
    num_token_replicas: int,
    mul_routed_weight: bool,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
):
    """
    Fused MoE kernel that multiplies tokens by their assigned expert weights.

    Args:
        A: Input tokens, shape (batch, K).
        B: Expert weights, shape (num_experts, N, K).
        C: Output tensor, shape (num_tokens * topk, N).
        topk_weights: Router weights for each token-expert pair, shape (num_tokens * topk,).
        sorted_token_ids: Token indices sorted by expert assignment, replicated topk times,
            and padded to align with TILE_M.
        sorted_expert_ids: Expert index for each TILE_M, sorted.
        num_token_replicas: Replication factor applied to each token row in A (topk or 1).
        mul_routed_weight: Whether to multiply output by router weights.

    Token ids are sorted and padded to ensure each expert processes a multiple of TILE_M tokens,
    enabling efficient tiled matrix multiplication.
    """

    M = sorted_token_ids.shape[0]
    N = B.shape[1]
    K = B.shape[2]

    GROUP_SIZE_M = 8
    bid_m, bid_n = swizzle_2d(M, N, TILE_M, TILE_N, GROUP_SIZE_M)

    zero_pad = ct.PaddingMode.ZERO

    # Gather replicated/padded token indices handled by this block pair (bid_m, bid_n).
    token_id_indices = bid_m * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    token_ids = ct.gather(sorted_token_ids, token_id_indices)

    # Collapse the replica dimension to recover the source row in A for each entry.
    a_row_indices = token_ids // num_token_replicas

    # Each TILE_M block is homogenous in expert assignment; fetch the expert id once.
    expert_id = ct.load(sorted_expert_ids, index=bid_m, shape=())

    accumulator = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
    for k in range(0, ct.cdiv(K, TILE_K)):
        a_col_indices = k * TILE_K + ct.arange(TILE_K, dtype=ct.int32)
        a = ct.gather(A, (a_row_indices[:, None], a_col_indices[None, :]))

        b = ct.load(B, (expert_id, k, bid_n), shape=(1, TILE_K, TILE_N),
                    order=(0, 2, 1), padding_mode=zero_pad).reshape((TILE_K, TILE_N))

        accumulator = ct.mma(a, b, accumulator)

    if mul_routed_weight:
        moe_weight = ct.gather(topk_weights, token_ids)
        accumulator = accumulator * moe_weight[:, None]

    # Compute the column span this block covers and scatter the tile back into C.
    c_col_indices = bid_n * TILE_N + ct.arange(TILE_N, dtype=ct.int32)
    accumulator = ct.astype(accumulator, C.dtype)
    ct.scatter(C, (token_ids[:, None], c_col_indices[None, :]), accumulator)


# -- PyTorch Utilities --

def silu_and_mul_torch(input: torch.Tensor, out: torch.Tensor):
    gate_result, up_result = input.chunk(2, dim=-1)
    torch.mul(F.silu(gate_result), up_result, out=out)


def moe_align_tile_size_torch(
    topk_ids: torch.Tensor, tile_m: int, num_experts: int
):
    """
    Sort, replicate, and pad token indices by expert so every expert processes a
    TILE_M-aligned tile when launching the fused_moe_kernel.

    Args:
        topk_ids: Router-selected expert ids per token (num_tokens, topk).
        tile_m: Tile size used along the M dimension by the kernel.
        num_experts: Total number of experts present in w1/w2 tensors.

    Returns:
        sorted_token_ids: 1-D tensor containing the flattened token-replica indices
            sorted by expert; remaining slots are filled with a sentinel index
            (num_tokens * topk) for padding.
        sorted_expert_ids: For each block, the expert id that
            owns the corresponding TILE_M slice in `sorted_token_ids`.
    """

    device = topk_ids.device
    num_tokens, topk = topk_ids.shape
    total_tokens = num_tokens * topk

    # Flatten expert ids (num_tokens * topk) and sort by experts.
    flat_expert_ids = topk_ids.reshape(-1)
    sorted_token_indices = torch.argsort(flat_expert_ids, stable=True)

    # Determine how many replicas each expert owns and how many TILE_M blocks we need
    # once padded to TILE_M alignment.
    expert_token_counts = torch.bincount(flat_expert_ids, minlength=num_experts)
    expert_block_counts = (expert_token_counts - 1 + tile_m) // tile_m
    total_blocks = expert_block_counts.sum()

    # Allocate output buffers; fill token ids with sentinel value (total_tokens).
    sorted_token_ids = torch.full((total_blocks * tile_m,), total_tokens,
                                  device=device, dtype=torch.int32)

    sorted_expert_ids = torch.zeros((total_blocks,), device=device,
                                    dtype=torch.int32)

    current_block = 0
    current_token = 0
    for expert_id in range(num_experts):
        token_count = expert_token_counts[expert_id]
        block_count = expert_block_counts[expert_id]

        # Map each TILE_M block with its owning expert id
        sorted_expert_ids[current_block:current_block+block_count] = expert_id

        sorted_token_start = current_block * tile_m
        # Copy the expert's sorted token indices; residual slots remain at the
        # sentinel value for padding.
        sorted_token_ids[sorted_token_start:sorted_token_start+token_count] = (
            sorted_token_indices[current_token:current_token+token_count]
        )

        current_token += token_count
        current_block += block_count

    return sorted_token_ids, sorted_expert_ids
