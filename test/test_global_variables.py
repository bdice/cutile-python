# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from math import ceil
import cuda.tile as ct
from cuda.tile._exception import TileSyntaxError, TileTypeError
from torch.testing import make_tensor
from util import assert_equal


global_int = 128
global_tuple = (128,)


@ct.kernel
def kernel_use_global_variable(x, y, z):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(global_int,))
    ty = ct.load(y, index=(bid,), shape=global_tuple)
    tz = tx + ty
    ct.store(z, index=(bid,), tile=tz)


def test_use_global_variable():
    shape = (128, )
    x = make_tensor(shape, dtype=torch.float32, device='cuda')
    y = make_tensor(shape, dtype=torch.float32, device='cuda')
    z = torch.zeros_like(x)
    grid = (ceil(shape[0] / global_int), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, kernel_use_global_variable, (x, y, z))
    assert_equal(z, x + y)


@ct.kernel
def kernel_read_before_assignment(x, y, z):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(global_int,))
    ty = ct.load(y, index=(bid,), shape=global_tuple)
    # Local assignment makes global_int local, but used before assignment.
    global_int += 1 # noqa
    tz = tx + ty
    ct.store(z, index=(bid,), tile=tz)


def test_kernel_read_before_assignment():
    shape = (128, )
    x = make_tensor(shape, dtype=torch.float32, device='cuda')
    y = make_tensor(shape, dtype=torch.float32, device='cuda')
    z = torch.zeros_like(x)
    grid = (ceil(shape[0] / global_int), 1, 1)
    with pytest.raises(TileSyntaxError, match=r"Undefined variable"):
        ct.launch(torch.cuda.current_stream(), grid, kernel_read_before_assignment, (x, y, z))


global_x = make_tensor((128, ), dtype=torch.float32, device='cuda')


@ct.kernel
def kernel_argument_over_global_variable(global_x, y, z, global_int: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(global_x, index=(bid,), shape=(global_int,))
    ty = ct.load(y, index=(bid,), shape=(global_int,))
    tz = tx + ty
    ct.store(z, index=(bid,), tile=tz)


def test_kernel_argument_over_global_variable():
    shape = (128, )
    y = make_tensor(shape, dtype=torch.float32, device='cuda')
    z = torch.zeros_like(global_x)
    grid = (ceil(shape[0] / global_int), 1, 1)
    half_global_int = global_int // 2
    ct.launch(torch.cuda.current_stream(), grid, kernel_argument_over_global_variable,
              (global_x, y, z, half_global_int))
    # Only the first half elements are used.
    assert_equal(z[:half_global_int], global_x[:half_global_int] + y[:half_global_int])
    assert_equal(z[half_global_int:], 0.)


@ct.kernel
def kernel_argument_using_global_tensor(y, z):
    bid = ct.bid(0)
    tx = ct.load(global_x, index=(bid,), shape=(global_int,))
    ty = ct.load(y, index=(bid,), shape=(global_int,))
    tz = tx + ty
    ct.store(z, index=(bid,), tile=tz)


def test_kernel_argument_using_global_tensor():
    shape = (128, )
    y = make_tensor(shape, dtype=torch.float32, device='cuda')
    z = torch.zeros_like(global_x)
    grid = (ceil(shape[0] / global_int), 1, 1)
    with pytest.raises(TileTypeError,
                       match=r"Cannot create constant from value of type torch.Tensor"):
        ct.launch(torch.cuda.current_stream(), grid, kernel_argument_using_global_tensor, (y, z))
