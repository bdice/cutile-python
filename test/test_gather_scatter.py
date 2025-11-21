# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import math
import re

import pytest
import torch
import numpy as np

from math import ceil
import cuda.tile as ct
from cuda.tile import TileValueError
from cuda.tile._compiler_options import CompilerOptions
from cuda.tile._exception import TileTypeError
from cuda.tile._ir.ops import LoadPointerTokenOrdered, StorePointerTokenOrdered
from cuda.tile._ir.ops_utils import _is_implicit_cast_ok
from cuda.tile._ir.typing_support import to_dtype
from cuda.tile._compile import compile_tile
from util import assert_equal, raises_if
from conftest import float_dtypes, bool_dtypes, int_dtypes, dtype_id
from torch.testing import make_tensor


@ct.kernel
def array_copy_1d(x, y, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    indices = ct.arange(TILE, dtype=np.int64)
    indices += bid*TILE
    tx = ct.gather(x, indices)
    ct.scatter(y, indices, tx)


@pytest.mark.parametrize("shape", [(128,), (225,), (260,)])
@pytest.mark.parametrize("tile", [128, 256])
@pytest.mark.parametrize("x_dtype", float_dtypes+int_dtypes+bool_dtypes, ids=dtype_id)
@pytest.mark.parametrize("y_dtype", float_dtypes+int_dtypes+bool_dtypes, ids=dtype_id)
def test_array_copy_1d(shape, x_dtype, y_dtype, tile):
    x = make_tensor(shape, dtype=x_dtype, device="cuda")
    y = torch.zeros_like(x, dtype=y_dtype)
    grid = (ceil(shape[0] / tile), 1, 1)

    invalid_cast = not _is_implicit_cast_ok(to_dtype(x_dtype), to_dtype(y_dtype))
    msg = "cannot implicitly cast"
    with raises_if(invalid_cast, TileTypeError, match=re.escape(msg)):
        ct.launch(torch.cuda.current_stream(), grid, array_copy_1d, (x, y, tile))
        assert_equal(x.to(y.dtype), y)


@ct.kernel
def array_copy_2d(x, y, TILE_X: ct.Constant[int], TILE_Y: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    ind_x = ct.arange(TILE_X, dtype=ct.int32) + bidx * TILE_X
    ind_y = ct.arange(TILE_Y, dtype=ct.int32) + bidy * TILE_Y
    t = ct.gather(x, (ind_x[:, None], ind_y))
    ct.scatter(y, (ind_x[:, None], ind_y), t)


@pytest.mark.parametrize("shape", [(128, 128), (192, 192), (128, 192)])
@pytest.mark.parametrize("tile", [(64, 64), (128, 32)])
@pytest.mark.parametrize("x_dtype", float_dtypes+int_dtypes+bool_dtypes, ids=dtype_id)
@pytest.mark.parametrize("y_dtype", float_dtypes+int_dtypes+bool_dtypes, ids=dtype_id)
def test_array_copy_2d(shape, x_dtype, y_dtype, tile):
    x = make_tensor(shape, dtype=x_dtype, device="cuda")
    y = torch.zeros_like(x, dtype=y_dtype)
    grid = (*(ceil(i / j) for i, j in zip(shape, tile)), 1)

    invalid_cast = not _is_implicit_cast_ok(to_dtype(x_dtype), to_dtype(y_dtype))
    msg = "cannot implicitly cast"
    with raises_if(invalid_cast, TileTypeError, match=re.escape(msg)):
        ct.launch(torch.cuda.current_stream(), grid, array_copy_2d,
                  (x, y, tile[0], tile[1]))
        assert_equal(x.to(y.dtype), y)


@ct.kernel
def scalar_copy(x, y):
    s = ct.gather(x, 0)
    ct.scatter(y, 0, s)


def test_scalar_copy():
    x = torch.full((1,), 7.0, dtype=torch.float32, device="cuda")
    y = torch.zeros_like(x, dtype=torch.float32)
    ct.launch(torch.cuda.current_stream(), (1,), scalar_copy, (x, y))
    assert y.cpu().item() == 7.0


@ct.kernel
def custom_padding_constant(x, y, pad_val: ct.Constant[int | float]):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(x, ind, padding_value=pad_val)
    ct.scatter(y, ind, t)


@pytest.mark.parametrize("pad_val", [7, 7.0, math.inf, -math.inf])
def test_custom_padding_constant(pad_val):
    x = torch.arange(100, 106, dtype=torch.float32, device="cuda")
    y = torch.zeros(8, dtype=torch.float32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), custom_padding_constant, (x, y, pad_val))
    assert y.cpu().tolist() == [
        100.0, 101.0, 102.0, 103.0, 104.0, 105.0, float(pad_val), float(pad_val)
    ]


def test_padding_value_out_of_range():
    x = torch.arange(100, 106, dtype=torch.int8, device="cuda")
    y = torch.zeros(8, dtype=torch.int32, device="cuda")
    with pytest.raises(TileValueError, match="128 is out of range"):
        ct.launch(torch.cuda.current_stream(), (1,), custom_padding_constant, (x, y, 128))


@ct.kernel
def literal_negative_infinity_padding(x, y):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(x, ind, padding_value=-math.inf)
    ct.scatter(y, ind, t)


def test_literal_negative_infinity_padding():
    x = torch.arange(100, 106, dtype=torch.float32, device="cuda")
    y = torch.zeros(8, dtype=torch.float32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), literal_negative_infinity_padding, (x, y))
    assert y.cpu().tolist() == [
        100.0, 101.0, 102.0, 103.0, 104.0, 105.0, -math.inf, -math.inf
    ]


@ct.kernel
def custom_padding_1d(x, y):
    ind = ct.arange(8, dtype=ct.int32)
    padding_value = ct.arange(8, dtype=ct.int32).astype(ct.float32)
    t = ct.gather(x, ind, padding_value=padding_value)
    ct.scatter(y, ind, t)


def test_custom_padding_1d():
    x = torch.arange(100, 106, dtype=torch.float32, device="cuda")
    y = torch.zeros(8, dtype=torch.float32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), custom_padding_1d, (x, y))
    assert y.cpu().tolist() == [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 6.0, 7.0]


@ct.kernel
def custom_padding_1d_broadcasted_to_2d(x, y):
    # Assuming x has length 5:
    #
    # ind:       gathered val:     bcasted pad:      t:
    # -------    ---------------   ---------------   ---------------
    # 0 2 4 6    100 102 104 pad   0   1   2   3     100 102 104 3
    # 1 3 5 7    101 103 pad pad   0   1   2   3     101 103 2   3
    ind = ct.arange(8, dtype=ct.int32).reshape((4, 2)).transpose()
    padding_value = ct.arange(4, dtype=ct.int32).astype(ct.float32)
    t = ct.gather(x, ind, padding_value=padding_value)
    ct.scatter(y, ind, t)


def test_custom_padding_1d_broadcasted_to_2d():
    x = torch.arange(100, 105, dtype=torch.float32, device="cuda")
    y = torch.zeros(8, dtype=torch.float32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), custom_padding_1d_broadcasted_to_2d, (x, y))
    assert y.cpu().tolist() == [100.0, 101.0, 102.0, 103.0, 104.0, 2.0, 3.0, 3.0]


@ct.kernel
def copy_8(x, y):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(x, ind)
    ct.scatter(y, ind, t)


def test_scatter_bounds_checking():
    x = torch.arange(10, 18, dtype=torch.float32, device="cuda")
    y = torch.arange(100, 108, dtype=torch.float32, device="cuda")
    # Create a view of `y` that only covers the first 5 elements
    y_slice = y[:5]
    ct.launch(torch.cuda.current_stream(), (1,), copy_8, (x, y_slice))

    # The value of `y` not covered but the slice should survive
    assert y.cpu().tolist() == [10.0, 11.0, 12.0, 13.0, 14.0, 105.0, 106.0, 107.0]


@ct.kernel
def copy_8_unchecked(x, y):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(x, ind, check_bounds=False)
    ct.scatter(y, ind, t, check_bounds=False)


def test_unchecked():
    x = torch.arange(10, 18, dtype=torch.float32, device="cuda")
    y = torch.zeros_like(x)
    ct.launch(torch.cuda.current_stream(), (1,), copy_8_unchecked, (x, y))
    assert y.cpu().tolist() == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]


@pytest.mark.parametrize("kernel, expected_mask", [
    (copy_8, True),
    (copy_8_unchecked, False),
], ids=["checked", "unchecked"])
def test_ir_checked_vs_unchecked(kernel, expected_mask):
    x = torch.arange(10, 18, dtype=torch.float32, device="cuda")
    y = torch.zeros_like(x)
    ir = compile_tile(kernel._pyfunc, (x, y), CompilerOptions()).final_ir

    load_ops = [op for op in ir.root_block.traverse() if isinstance(op, LoadPointerTokenOrdered)]
    assert len(load_ops) == 1
    assert (load_ops[0].mask is not None) == expected_mask

    store_ops = [op for op in ir.root_block.traverse() if isinstance(op, StorePointerTokenOrdered)]
    assert len(store_ops) == 1
    assert (store_ops[0].mask is not None) == expected_mask
