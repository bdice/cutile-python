# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import cuda.tile as ct
from cuda.tile._ir.ops import Loop
from cuda.tile._compiler_options import CompilerOptions
from cuda.tile._compile import compile_tile

from util import assert_equal


@ct.kernel
def split_ge_kernel(x):
    for i in range(x.shape[0]):
        val = i
        if i >= 3:
            val *= 10
        ct.store(x, i, val)


def test_split_ge():
    x = torch.zeros(10, dtype=torch.int32, device="cuda")
    root_block = compile_tile(split_ge_kernel._pyfunc, (x,), CompilerOptions()).final_ir
    loop_ops = [op for op in root_block.traverse() if isinstance(op, Loop)]
    assert len(loop_ops) == 2

    ct.launch(torch.cuda.current_stream(), (1,), split_ge_kernel, (x,))
    ref = torch.tensor([0, 1, 2, 30, 40, 50, 60, 70, 80, 90], dtype=torch.int32, device="cuda")
    assert_equal(x, ref)
