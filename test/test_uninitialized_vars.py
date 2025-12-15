# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from math import ceil
import cuda.tile as ct
from cuda.tile import TileTypeError
from cuda.tile._exception import TileSyntaxError


@ct.kernel
def mma_uninitialized_var_both_sides(A, B, C,
                                     tm: ct.Constant[int],
                                     tn: ct.Constant[int],
                                     tk: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    a = ct.load(A, index=(bidx, 0), shape=(tm, tk))
    b = ct.load(B, index=(0, bidy), shape=(tk, tn))
    acc = ct.mma(a, b, acc) # noqa
    acc = ct.astype(acc, C.dtype)
    ct.store(C, index=(bidx, bidy), tile=acc)


@ct.kernel
def mma_uninitialized_var_right_side(A, B, C,
                                     tm: ct.Constant[int],
                                     tn: ct.Constant[int],
                                     tk: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    a = ct.load(A, index=(bidx, 0), shape=(tm, tk))
    b = ct.load(B, index=(0, bidy), shape=(tk, tn))
    acc2 = ct.mma(a, b, acc) # noqa
    acc2 = ct.astype(acc2, C.dtype)
    ct.store(C, index=(bidx, bidy), tile=acc2)


@ct.kernel
def mma_uninitialized_var_right_side_global(A, B, C,
                                            tm: ct.Constant[int],
                                            tn: ct.Constant[int],
                                            tk: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    a = ct.load(A, index=(bidx, 0), shape=(tm, tk))
    b = ct.load(B, index=(0, bidy), shape=(tk, tn))
    sum2 = ct.mma(a, b, sum) # noqa
    sum2 = ct.astype(sum2, C.dtype)
    ct.store(C, index=(bidx, bidy), tile=sum2)


@ct.kernel
def mma_uninitialized_var_in_loop(A, B, C,
                                  tm: ct.Constant[int],
                                  tn: ct.Constant[int],
                                  tk: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    num_tiles = ct.num_tiles(A, axis=1, shape=(tm, tk))
    # num_tiles = A.shape[1]/tk

    # acc is not initialized, so it's undefined.
    for k in range(num_tiles):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk))
        b = ct.load(B, index=(k, bidy), shape=(tk, tn))
        acc = ct.mma(a, b, acc) # noqa

    acc = ct.astype(acc, C.dtype)
    ct.store(C, index=(bidx, bidy), tile=acc)


@ct.kernel
def mma_uninitialized_var_in_if(A, B, C,
                                tm: ct.Constant[int],
                                tn: ct.Constant[int],
                                tk: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    a = ct.load(A, index=(bidx, 0), shape=(tm, tk))
    b = ct.load(B, index=(0, bidy), shape=(tk, tn))
    if bidx == 0:
        acc = ct.full((tm, tn), 0, dtype=C.dtype)
        acc = ct.mma(a, b, acc)
    else:
        acc = ct.mma(a, b, acc)
    acc = ct.astype(acc, C.dtype)
    ct.store(C, index=(bidx, bidy), tile=acc)


@pytest.mark.parametrize("func", [
    mma_uninitialized_var_both_sides,
    mma_uninitialized_var_right_side,
    mma_uninitialized_var_right_side_global,
    mma_uninitialized_var_in_loop,
    mma_uninitialized_var_in_if,
])
def test_uninitialized_vars(func):
    if func is mma_uninitialized_var_right_side_global:
        pytest.xfail("Uninitialized variable with built-in name is not raising expected error yet.")
    m, n, k = 4, 2, 8
    A = torch.randn((m, k), dtype=torch.float32, device="cuda")
    B = torch.randn((k, n), dtype=torch.float32, device=A.device)
    C = torch.zeros((m, n), dtype=torch.float32, device=A.device)
    tm, tn, tk = 2, 2, 2
    grid = (ceil(m / tm), ceil(n / tn), 1)
    with pytest.raises((TileSyntaxError, TileTypeError), match="[Uu]ndefined variable"):
        ct.launch(torch.cuda.current_stream(), grid, func, (A, B, C, tm, tn, tk))
