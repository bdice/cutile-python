# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch
import torch.nn.functional as F
import pytest
import cuda.tile as ct

from conftest import dtype_id, shape_id
from util import estimate_bench_iter
from kernels.fused_moe import fused_moe_kernel, silu_and_mul_torch, moe_align_tile_size_torch


@pytest.fixture(params=[
    (110, 1024, 16, 512, 4),
    (256, 2048, 1408, 32, 6),
], ids=shape_id)
def shape(request):
    """
    (num_tokens, hidden_size, num_experts, intermediate_size, topk)
    """
    return request.param


@pytest.fixture(params=[
    torch.float16, torch.bfloat16, torch.float32
], ids=dtype_id)
def dtype(request):
    return request.param


@pytest.mark.benchmark(group='moe')
def bench_moe(shape, dtype, backend, benchmark):
    num_tokens, hidden_size, num_experts, intermediate_size, topk = shape
    device = "cuda"

    hidden_states = torch.empty(
        num_tokens, hidden_size, device=device, dtype=dtype
    ).normal_(0, 0.5)
    w1 = torch.empty(
        num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
    ).normal_(0, 0.1)
    w2 = torch.empty(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ).normal_(0, 0.1)

    # Unique expert IDs for each token (no repeating elements per row)
    topk_ids = torch.stack([
        torch.randperm(num_experts, device=device)[:topk]
        for _ in range(num_tokens)
    ])
    topk_weights = torch.softmax(
        torch.randn(num_tokens, topk, device=device), dim=-1
    ).to(dtype)

    result = backend(hidden_states, w1, w2, topk_weights, topk_ids)
    ref = torch_moe(hidden_states, w1, w2, topk_weights, topk_ids)

    if dtype == torch.float16:
        rtol, atol = 5e-2, 5e-2
    elif dtype == torch.bfloat16:
        rtol, atol = 1e-1, 1e-1
    else:
        rtol, atol = 1e-2, 1e-2

    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)
    torch.cuda.synchronize()

    warmup_rounds, iterations, rounds = estimate_bench_iter(
        backend, (hidden_states, w1, w2, topk_weights, topk_ids)
    )

    benchmark.pedantic(
        backend, (hidden_states, w1, w2, topk_weights, topk_ids),
        rounds=rounds, warmup_rounds=warmup_rounds, iterations=iterations,
    )


def torch_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor
) -> torch.Tensor:
    """
    hidden_states: (num_tokens, hidden_size)
    w1: (num_experts, intermediate_size * 2, hidden_size)
    w2: (num_experts, hidden_size, intermediate_size)
    topk_weights: (num_tokens, topk)
    topk_ids: (num_tokens, topk)
    """
    gate_proj, up_proj = w1.chunk(2, dim=1)
    down_proj = w2

    num_experts = w1.shape[0]

    final_hidden_states = torch.zeros_like(hidden_states)

    # (num_experts, topk, num_tokens)
    expert_mask = F.one_hot(topk_ids, num_classes=num_experts).permute(2, 1, 0)

    expert_usage = expert_mask.sum(dim=(-1, -2)) > 0
    active_expert_ids = expert_usage.nonzero().squeeze(-1)

    for expert_id in active_expert_ids:
        expert_gate = gate_proj[expert_id]
        expert_up = up_proj[expert_id]
        expert_down = down_proj[expert_id]

        matched_ks, matched_token_ids = torch.where(expert_mask[expert_id])
        matched_tokens = hidden_states[matched_token_ids]

        gate_output = matched_tokens @ expert_gate.T
        up_output = matched_tokens @ expert_up.T
        swiglu_output = F.silu(gate_output) * up_output
        expert_output = swiglu_output @ expert_down.T

        routing_weights = topk_weights[matched_token_ids, matched_ks]
        weighted_output = expert_output * routing_weights.unsqueeze(-1)

        final_hidden_states.index_add_(
            0,
            matched_token_ids,
            weighted_output.to(hidden_states.dtype)
        )

    return final_hidden_states


def cutile_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor
) -> torch.Tensor:

    tile_m = 128
    tile_n = 128
    tile_k = 64

    out_dtype = hidden_states.dtype
    device = hidden_states.device

    num_tokens, hidden_size = hidden_states.shape
    num_experts, _, intermediate_size = w2.shape
    _, topk = topk_ids.shape

    assert w1.shape[1] == intermediate_size * 2

    intermediate_cache1 = torch.zeros(
        (num_tokens, topk, intermediate_size * 2),
        device=device, dtype=out_dtype,
    )
    intermediate_cache2 = torch.zeros(
        (num_tokens * topk, intermediate_size),
        device=device,
        dtype=out_dtype,
    )
    intermediate_cache3 = torch.zeros(
        (num_tokens, topk, hidden_size),
        device=device,
        dtype=out_dtype,
    )

    sorted_token_ids, sorted_expert_ids = moe_align_tile_size_torch(
        topk_ids,
        tile_m,
        num_experts
    )

    invoke_fused_moe_kernel(
        hidden_states,
        w1,
        intermediate_cache1,
        topk_weights,
        sorted_token_ids,
        sorted_expert_ids,
        False,
        topk,
        tile_m,
        tile_n,
        tile_k,
    )

    silu_and_mul_torch(
        intermediate_cache1.view(-1, intermediate_cache1.shape[-1]),
        intermediate_cache2,
    )

    invoke_fused_moe_kernel(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        topk_weights,
        sorted_token_ids,
        sorted_expert_ids,
        True,
        1,
        tile_m,
        tile_n,
        tile_k,
    )

    return torch.sum(intermediate_cache3, dim=1)


def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    mul_routed_weight: bool,
    num_token_replicas: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
):
    m = sorted_token_ids.shape[0]
    n = B.shape[1]

    grid = (math.ceil(m / tile_m) * math.ceil(n / tile_n),)

    topk_weights = topk_weights.view(-1)
    C = C.view(-1, C.shape[2])

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        fused_moe_kernel,
        (
            A,
            B,
            C,
            topk_weights,
            sorted_token_ids,
            sorted_expert_ids,
            num_token_replicas,
            mul_routed_weight,
            tile_m,
            tile_n,
            tile_k,
        )
    )
