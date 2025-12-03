# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

from .ast2hir import get_function_hir
from .. import TileTypeError
from .._exception import Loc, TileSyntaxError, TileInternalError, TileError
from .._ir import hir, ir
from .._ir.ir import Var, IRContext, Argument
from .._ir.op_impl import op_implementations
from .._ir.ops import loosely_typed_const, get_bound_self, assign, end_branch, return_, continue_, \
    break_
from .._ir.type import FunctionTy, BoundMethodTy, DTypeConstructor
from .._ir.typing_support import get_signature


def hir2ir(func_body: hir.Block, args: tuple[Argument, ...], ir_ctx: IRContext) -> ir.Block:
    new_params = []
    with ir.Builder(ir_ctx, func_body.loc) as ir_builder:
        mapper = ir.Mapper(ir_ctx, preserve_vars=True)
        for var, arg in zip(func_body.params, args, strict=True):
            if arg.is_const:
                with ir_builder.change_loc(var.loc):
                    const_var = loosely_typed_const(arg.const_value)
                mapper.set_var(const_var, var)
                ir_ctx.copy_type_information(const_var, var)

                unused_param = ir_ctx.make_var_like(var)
                unused_param.set_type(arg.type)
                new_params.append(unused_param)
            else:
                var.set_type(arg.type)
                var.set_loose_type(arg.loose_type)
                new_params.append(var)

        if not mapper.is_empty():
            for i in range(len(ir_builder.ops)):
                ir_builder.ops[i] = ir_builder.ops[i].clone(mapper)

        try:
            _dispatch_hir_block_inner(func_body, ir_builder)
        except Exception as e:
            if 'CUTILEIR' in ir_ctx.tile_ctx.config.log_keys:
                highlight_loc = e.loc if hasattr(e, 'loc') else None
                ir_str = "\n".join(op.to_string(highlight_loc=highlight_loc)
                                   for op in ir_builder.ops)
                print(f"==== Partial cuTile IR ====\n\n{ir_str}\n\n", file=sys.stderr)
            raise

    ret = ir.Block(ir_ctx, new_params, func_body.name, func_body.loc)
    ret.extend(ir_builder.ops)
    return ret


def dispatch_hir_block(block: hir.Block, ignore_jump: bool = False) -> tuple[Var, ...]:
    return _dispatch_hir_block_inner(block, ir.Builder.get_current(), ignore_jump)


@dataclass
class _State:
    done: list[hir.Call]
    current: hir.Call | None
    todo_stack: list[hir.Call]

    @contextmanager
    def next_call(self):
        call = self.current = self.todo_stack.pop()
        yield call
        # Intentionally not in a "finally" block because we want to preserve the state
        # for the debug printout in case of an exception.
        self.current = None
        self.done.append(call)


def _dispatch_hir_block_inner(block: hir.Block, builder: ir.Builder,
                              ignore_jump: bool = False) -> tuple[Var, ...]:
    state = _State([], None, list(reversed(block.calls)))
    try:
        if not _dispatch_hir_calls(state, builder):
            return ()
        result_vars = tuple(_ensure_var_or_block(x) for x in block.results)
        if not ignore_jump:
            with _wrap_exceptions(block.jump_loc), builder.change_loc(block.jump_loc):
                _dispatch_hir_jump(block.jump, result_vars)
        return result_vars
    except Exception:
        if 'CUTILEIR' in builder.ir_ctx.tile_ctx.config.log_keys:
            hir_params = ", ".join(p.name for p in block.params)
            hir_lines = [str(c) for c in state.done]
            cur_idx = len(hir_lines)
            if state.current is not None:
                hir_lines.append(str(state.current))
            hir_lines.extend(str(c) for c in reversed(state.todo_stack))
            hir_lines.append(block.jump_str())
            hir_str = "\n".join("{}{}".format("--> " if i == cur_idx else "    ", c)
                                for i, c in enumerate(hir_lines))
            print(f"==== HIR for ^{block.name}({hir_params}) ====\n{hir_str}\n", file=sys.stderr)
        raise


def _dispatch_hir_jump(jump: hir.Jump,
                       block_results: tuple[Var, ...]):
    match jump:
        case hir.Jump.END_BRANCH:
            end_branch(block_results)
        case hir.Jump.CONTINUE:
            continue_(block_results)
        case hir.Jump.BREAK:
            break_(block_results)
        case hir.Jump.RETURN:
            assert len(block_results) == 1
            return_(block_results[0])
        case _: assert False


def _dispatch_hir_calls(state: _State, cur_builder: ir.Builder) -> bool:
    while len(state.todo_stack) > 0:
        with state.next_call() as call:
            with _wrap_exceptions(call.loc), cur_builder.change_loc(call.loc):
                _dispatch_call(call, cur_builder, state.todo_stack)
            if cur_builder.is_terminated:
                # The current block has been terminated, e.g. by flattening an if-else
                # with a constant condition (`if True: break`). By returning False,
                # we signal that the original jump and block results should be ignored.
                return False
    return True


@contextmanager
def _wrap_exceptions(loc: Loc):
    with loc:
        try:
            yield
        except TileError:
            raise
        except Exception as e:
            raise TileInternalError(str(e)) from e


def _dispatch_call(call: hir.Call, builder: ir.Builder, todo_stack: list[hir.Call]):
    first_idx = len(builder.ops)
    callee_var = _ensure_var_or_block(call.callee)
    callee, self_arg = _get_callee_and_self(callee_var)
    args = (*self_arg, *(_ensure_var_or_block(x) for x in call.args))
    kwargs = {k: _ensure_var_or_block(v) for k, v in call.kwargs}
    arg_list = _bind_args(callee, args, kwargs)

    if callee in op_implementations:
        result = op_implementations[callee](*arg_list)
        if builder.is_terminated:
            # The current block has been terminated, e.g. by flattening an if-else
            # with a constant condition (`if True: break`). Ignore the `result` in this case.
            return

        if result is None:
            result = (loosely_typed_const(None),) if len(call.results) == 1 else ()
        elif not isinstance(result, tuple):
            result = (result,)

        # Remap result variables
        assert len(result) == len(call.results)
        mapper = ir.Mapper(builder.ir_ctx, preserve_vars=True)
        for impl_res, call_res in zip(result, call.results, strict=True):
            builder.ir_ctx.copy_type_information(impl_res, call_res)
            if _is_freshly_defined(impl_res, builder, first_idx):
                # The result is a freshly defined variable.
                # Replace it with the original variable.
                mapper.set_var(impl_res, call_res)
            else:
                # The result is a pre-existing variable.
                # This mainly happens when an operation implementation reduces to a no-op
                # by returning its input. For example, `reshape(x, new_shape)` may return `x`
                # when the new shape is the same as the old one. So we need to replace
                # `y = reshape(x, new_shape)` with `y = assign(x)` to make sure `y` is defined.
                assign(impl_res, call_res)

        if not mapper.is_empty():
            for i in range(first_idx, len(builder.ops)):
                builder.ops[i] = builder.ops[i].clone(mapper)
    else:
        # Callee is a user-defined function.
        _check_recursive_call(call.loc, callee)
        sig = get_signature(callee)
        for param_name, param in sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD):
                raise TileSyntaxError("Variadic parameters in user-defined"
                                      " functions are not supported")
        callee_hir = get_function_hir(callee, builder.ir_ctx, call_site=call.loc)
        for callee_retval, caller_res in zip(callee_hir.results, call.results):
            todo_stack.append(hir.Call((caller_res,), hir.identity, (callee_retval,), (), call.loc))
        todo_stack.extend(reversed(callee_hir.calls))
        for arg, param in zip(arg_list, callee_hir.params, strict=True):
            todo_stack.append(hir.Call((param,), hir.identity, (arg,), (), call.loc))


def _is_freshly_defined(var: Var, builder: ir.Builder, first_idx: int):
    return any(var.name == r.name
               for i in range(first_idx, len(builder.ops))
               for r in builder.ops[i].result_vars)


def _check_recursive_call(call_loc: Loc, callee: Callable):
    while call_loc is not None:
        if call_loc.function is callee:
            raise TileTypeError("Recursive function call detected")
        call_loc = call_loc.call_site


def _get_callee_and_self(callee_var: Var) -> tuple[Any, tuple[()] | tuple[Var]]:
    callee_ty = callee_var.get_type()
    if isinstance(callee_ty, FunctionTy):
        return callee_ty.func, ()
    elif isinstance(callee_ty, BoundMethodTy):
        return callee_ty.func, (get_bound_self(callee_var),)
    elif isinstance(callee_ty, DTypeConstructor):
        return callee_ty.dtype, ()
    else:
        raise TileTypeError(f"Cannot call an object of type {callee_ty}")


def _ensure_var_or_block(x: hir.Operand) -> Var | hir.Block:
    if isinstance(x, Var | hir.Block):
        return x
    else:
        return loosely_typed_const(x)


def _bind_args(sig_func, args, kwargs) -> list[Var]:
    sig = get_signature(sig_func)
    try:
        bound_args = sig.bind(*args, **kwargs)
    except TypeError as e:
        raise TileTypeError(f"{sig_func.__name__}(): {e}")
    ret = []
    for name, param in sig.parameters.items():
        if name in bound_args.arguments:
            ret.append(bound_args.arguments[name])
        elif param.kind == param.VAR_POSITIONAL:
            ret.append(())
        else:
            assert param.default is not param.empty
            ret.append(loosely_typed_const(param.default))
    return ret
