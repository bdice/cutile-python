# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from .._ir import ir
from .._ir.ops import Assign


def eliminate_assign_ops(root_block: ir.Block):
    def walk(block):
        new_ops = []
        for op in block:
            if isinstance(op, Assign):
                var = orig_var.get(op.value.name, op.value)
                orig_var[op.result_var.name] = var
                mapper.set_var(op.result_var, var)
            else:
                for nested_block in op.nested_blocks:
                    walk(nested_block)
                new_ops.append(op)
        block[:] = new_ops

    mapper = ir.Mapper(root_block.ctx, preserve_vars=True)
    orig_var = dict()
    walk(root_block)
    root_block[:] = [op.clone(mapper) for op in root_block]
