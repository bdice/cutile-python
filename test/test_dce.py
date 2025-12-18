# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import cuda.tile as ct
from cuda.tile._ir.ops import Loop, Continue, Break
from cuda.tile._ir import ir
from cuda.tile._compile import _get_final_ir
from cuda.tile._cext import default_tile_context


def get_ir(func) -> ir.Block:
    x = torch.zeros(10, device="cuda")
    ir = _get_final_ir(func, (x,), default_tile_context)
    print(ir)
    return ir


def test_unused_loop_var():
    def kernel(x):
        a = 0   # can be pruned
        t = ct.load(x, (0,), (1,))
        for i in range(10):
            a = a + 1   # can be pruned
            t = t + 1
        ct.store(x, (1,), t)

    func_body = get_ir(kernel)
    loop, = [op for op in func_body if isinstance(op, Loop)]
    assert [v.get_original_name() for v in loop.body_vars] == ["t"]


def test_unused_body_var():
    def kernel(x):
        t = ct.load(x, (0,), (1,))   # can be pruned
        i = 0
        while True:
            t = ct.ones((1,), x.dtype)
            if i > ct.bid(0):
                break
            t = t + 1   # can be pruned
            i = i + 1
        ct.store(x, (1,), t)

    func_body = get_ir(kernel)
    loop, = [op for op in func_body if isinstance(op, Loop)]

    # The initial variable should be undefined because it is never used
    t_idx = [v.get_original_name() for v in loop.body.params].index("t")
    assert loop.initial_values[t_idx].is_undefined()

    # The yielded variable should also be undefined because it is never used
    continue_op = loop.body[-1]
    assert isinstance(continue_op, Continue)
    assert continue_op.values[t_idx].is_undefined()


def test_unused_result_var():
    def kernel(x):
        t = ct.load(x, (0,), (1,))
        i = 0
        while True:
            ct.store(x, (1,), t)
            t = t + 1
            if i > ct.bid(0):
                t = t + 2  # can be pruned
                break
            i = i + 1

    func_body = get_ir(kernel)
    loop, = [op for op in func_body if isinstance(op, Loop)]

    # The value yielded by "break" should be undefined
    t_idx = [v.get_original_name() for v in loop.body_vars].index("t")
    break_op, = [op for op in func_body.traverse() if isinstance(op, Break)]
    assert break_op.values[t_idx].is_undefined()
