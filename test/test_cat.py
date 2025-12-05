# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cuda.tile as ct
from util import assert_equal
from cuda.tile._exception import TileTypeError


@ct.kernel
def cat2(out, axis: ct.Constant[int]):
    tx = ct.ones((2, 2), ct.int32)
    ty = ct.ones((2, 2), ct.int32)
    tz = ct.cat((tx, ty), axis).reshape((8,))
    ct.store(out, (0,), tile=tz)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_cat(axis: int):
    out = torch.zeros(8, dtype=torch.int32, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), cat2, (out, axis))
    ref = torch.ones_like(out)
    assert_equal(out, ref)


@pytest.mark.parametrize("axis", [2, -3])
def test_illegal_axis(axis):
    out = torch.zeros(8, dtype=torch.int32, device='cuda')
    with pytest.raises(TileTypeError):
        ct.launch(torch.cuda.current_stream(), (1,), cat2, (out, axis))


@ct.kernel
def cat3(out):
    tx = ct.ones((2, 2), ct.int32)
    ty = ct.ones((4, 2), ct.int32)
    ty = ct.cat((tx, ty), 0)
    ct.store(out, (0,), tile=ty)


def test_non_power_of_two():
    out = torch.zeros(8, dtype=torch.int32, device='cuda')
    with pytest.raises(TileTypeError, match=r"Result tile shape must be power of 2"):
        ct.launch(torch.cuda.current_stream(), (1,), cat3, (out,))


@ct.kernel
def cat4(out):
    tx = ct.ones((2,), ct.int32)
    ty = ct.cat((tx, tx, tx, tx), 0)
    ct.store(out, (0,), tile=ty)


def test_more_than_two_tiles():
    out = torch.zeros(8, dtype=torch.int32, device='cuda')
    with pytest.raises(TileTypeError, match=r"cat\(\) supports at most 2 tiles, got 4"):
        ct.launch(torch.cuda.current_stream(), (1,), cat4, (out,))


@ct.kernel
def cat_mixed_dtype(out):
    tx = ct.ones((2, 2), ct.int32)
    ty = ct.ones((2, 2), ct.float32)
    tz = ct.cat((tx, ty), 0).reshape((8,))
    ct.store(out, (0,), tile=tz)


def test_mixed_dtype():
    out = torch.zeros(8, dtype=torch.int32, device='cuda')
    with pytest.raises(TileTypeError, match=r"Expected tiles to have the same dtype"):
        ct.launch(torch.cuda.current_stream(), (1,), cat_mixed_dtype, (out,))


@ct.kernel
def cat_mixed_rank(out):
    tx = ct.ones((2, 2), ct.int32)
    ty = ct.ones((2, 2, 1), ct.int32)
    tz = ct.cat((tx, ty), 0).reshape((8,))
    ct.store(out, (0,), tile=tz)


def test_mixed_rank():
    out = torch.zeros(8, dtype=torch.int32, device='cuda')
    with pytest.raises(TileTypeError, match=r"Expected tiles to have the same rank"):
        ct.launch(torch.cuda.current_stream(), (1,), cat_mixed_rank, (out,))


@ct.kernel
def cat_mixed_shape(out):
    tx = ct.ones((2, 2), ct.int32)
    ty = ct.ones((2, 4), ct.int32)
    tz = ct.cat((tx, ty), 0).reshape((8,))
    ct.store(out, (0,), tile=tz)


def test_mixed_shape():
    out = torch.zeros(8, dtype=torch.int32, device='cuda')
    with pytest.raises(TileTypeError, match=r"Expected tiles to have the same shape for non axis dimensions"):  # noqa: E501
        ct.launch(torch.cuda.current_stream(), (1,), cat_mixed_shape, (out,))


@ct.kernel
def cat1(out):
    tx = ct.ones((2, 2), ct.int32)
    tz = ct.cat((tx,), 0).reshape((4,))
    ct.store(out, (0,), tile=tz)


def test_cat_with_one_tile():
    out = torch.zeros(4, dtype=torch.int32, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), cat1, (out,))
    ref = torch.ones_like(out)
    assert_equal(out, ref)
