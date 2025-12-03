# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cuda.tile as ct


def ifelse_alias(X, Y, cond: int, TILE: ct.Constant[int]):
    if cond:
        alias = Y
    else:
        alias = X
    ct.store(alias, index=(0,), tile=ct.full((TILE,), 3, dtype=X.dtype))


def for_loop_alias(X, Y, n: int, TILE: ct.Constant[int]):
    alias = X
    for i in range(n):
        alias = Y
    ct.store(alias, index=(0,), tile=ct.full((TILE,), 3, dtype=X.dtype))


def while_loop_alias(X, Y, n: int, TILE: ct.Constant[int]):
    alias = X
    i = 0
    while i < n:
        alias = Y
        i += 1
    ct.store(alias, index=(0,), tile=ct.full((TILE,), 3, dtype=X.dtype))


def nested_alias(X, Y, n: int, TILE: ct.Constant[int]):
    alias = X
    if n <= 100:
        for i in range(n):
            alias = Y
    ct.store(alias, index=(0,), tile=ct.full((TILE,), 3, dtype=X.dtype))


def tuple_alias(X, Y, n: int, TILE: ct.Constant[int]):
    alias1, _ = (Y, X)
    ct.store(alias1, index=(0,), tile=ct.full((TILE,), 3, dtype=X.dtype))


def nested_tuple_alias(X, Y, n: int, TILE: ct.Constant[int]):
    t = (X, Y)
    if n <= 100:
        i = 0
        while i < n:
            t = (Y, X)
            i += 1
    ct.store(t[0], index=(0,), tile=ct.full((TILE,), 3, dtype=X.dtype))


def helper(X, Y):
    return Y, X


def helper_alias(X, Y, n: int, TILE: ct.Constant[int]):
    alias_tuple = helper(X, Y)[:1]
    ct.store(alias_tuple[0], index=(0,), tile=ct.full((TILE,), 3, dtype=X.dtype))


@pytest.mark.parametrize(
    "kernel",
    [ifelse_alias, for_loop_alias, while_loop_alias, nested_alias,
     tuple_alias, nested_tuple_alias, helper_alias],
    ids=lambda x: x.__name__,
)
def test_alias(kernel):
    tile = 256
    X = torch.zeros((tile,), dtype=torch.float32, device='cuda')
    Y = torch.zeros_like(X)
    expected_Y = torch.full((tile,), 3, dtype=torch.float32, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), ct.kernel(kernel), (X, Y, 1, tile))
    torch.testing.assert_close(Y, expected_Y)
