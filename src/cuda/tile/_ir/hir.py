# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


# HIR stands for "High-level Intermediate Representation".
# The HIR is the initial representation that we build from the Python AST (see ast2hir.py).
#
# Unlike the IR, it doesn't have specific Operation definitions. Instead, it uses the concept
# of a "function call" to model all operations, including structured control flow.
# For example, addition like `a + b` is represented as calling `operator.add(a, b)`.
#
# It also has a simpler representation of constants: they can be used directly as arguments
# or as functions to be called (see the `Operand` type alias below).


import enum
from dataclasses import dataclass
from textwrap import indent
from typing import Any, Set, Mapping

from cuda.tile._exception import Loc
from cuda.tile._ir.ir import Var


# An "Operand" is a value that can be used as a function's argument, or as the function itself.
# There are two kinds of Operands: variables and constants. Using a `Var` instance as an Operand
# signals that this Operand is a variable, e.g. a result of a previous call or a kernel parameter.
# An object of any other type means that it is an immediate constant.
Operand = Var | Any


ModuleType = type(enum)


@dataclass
class Call:
    results: tuple[Var, ...]
    callee: Operand
    args: tuple[Operand, ...]
    kwargs: tuple[tuple[str, Operand], ...]
    loc: Loc

    def __str__(self):
        opfmt = _OperandFormatter([])
        loc_str = f"  # Line {self.loc.line}"
        if self.callee is identity:
            return f"{_lhs_var_str(self.results[0])} = {opfmt(self.args[0])}{loc_str}"
        callee_str = opfmt(self.callee)
        results_str = ", ".join(_lhs_var_str(r) for r in self.results)
        lhs_str = f"{results_str} = " if results_str else ""
        args_and_kwargs = (*(opfmt(a) for a in self.args),
                           *(f"{k}={opfmt(v)}" for k, v in self.kwargs))
        args_str = ", ".join(args_and_kwargs)
        blocks_str = "".join(indent(f"\n{b}", "    ") for b in opfmt.blocks)
        return f"{lhs_str}{callee_str}({args_str}){loc_str}{blocks_str}"


def _lhs_var_str(var: Var):
    ty = var.try_get_type()
    if ty is None:
        return var.name
    return f"{var.name}: {ty}"


class Jump(enum.Enum):
    END_BRANCH = "end_branch"
    CONTINUE = "continue"
    BREAK = "break"
    RETURN = "return"


@dataclass
class Block:
    name: str
    params: tuple[Var, ...]
    calls: list[Call]
    results: tuple[Operand, ...]
    jump: Jump | None
    jump_loc: Loc
    stored_names: Set[str]
    loc: Loc

    def __str__(self):
        params_str = ", ".join(p.name for p in self.params)
        calls_str = "".join(f"\n{c}" for c in self.calls)
        if self.jump is not None:
            calls_str += "\n" + self.jump_str()
        calls_str = indent(calls_str, "    ")
        return f"^{self.name}({params_str}):{calls_str}"

    def jump_str(self):
        opfmt = _OperandFormatter([])
        results_str = ",".join(f" {opfmt(r)}" for r in self.results)
        return f"{self.jump._value_}{results_str}  # Line {self.jump_loc.line}"


@dataclass
class Function:
    body: Block
    param_names: tuple[str, ...]
    param_locs: tuple[Loc, ...]
    frozen_globals: Mapping[str, Any]


@dataclass
class _OperandFormatter:
    blocks: list["Block"]

    def __call__(self, x: Operand) -> str:
        if isinstance(x, Var):
            return x.name
        elif isinstance(x, ModuleType):
            return str(f"<mod:{x.__name__}>")
        elif isinstance(x, Block):
            self.blocks.append(x)
            return f"^{x.name}"
        elif callable(x):
            return f"<fn:{x.__name__}>"
        else:
            return f"<{repr(x)}>"


# ==================================
# Special function stubs used in HIR
# ==================================

def if_else(cond, then_block, else_block, /): ...
def loop(body, iterable, /): ...  # infinite if `iterable` is None
def build_tuple(*items): ...  # Makes a tuple (i.e. returns `items`)
def identity(x): ...   # Identity function (i.e. returns `x`)
def store_var(name, value, /): ...  # Store into a named variable
def load_var(name, /): ...  # Load from a named variable
