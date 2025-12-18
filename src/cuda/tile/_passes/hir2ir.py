# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Sequence

from .ast2hir import get_function_hir
from .. import TileTypeError
from .._exception import Loc, TileSyntaxError, TileInternalError, TileError, TileRecursionError
from .._ir import hir, ir
from .._ir.ir import Var, IRContext, Argument, Scope, LocalScope, BoundMethodValue
from .._ir.op_impl import op_implementations, impl
from .._ir.ops import loosely_typed_const, assign, end_branch, return_, continue_, \
    break_, flatten_block_parameters
from .._ir.type import FunctionTy, BoundMethodTy, DTypeConstructor
from .._ir.typing_support import get_signature


MAX_RECURSION_DEPTH = 50


def hir2ir(func_hir: hir.Function,
           args: tuple[Argument, ...],
           ir_ctx: IRContext) -> ir.Block:
    scope = _create_scope(func_hir, ir_ctx, call_site=None)
    aggregate_params = [
        scope.local.redefine(param_name, param_loc)
        for param_name, param_loc in zip(func_hir.param_names, func_hir.param_locs, strict=True)
    ]
    preamble = []
    for param_name, var, arg in zip(func_hir.param_names, aggregate_params, args, strict=True):
        var.set_type(arg.type)
        if arg.is_const:
            preamble.append(hir.Call((), hir.store_var, (param_name, arg.const_value), (), var.loc))

    with ir.Builder(ir_ctx, func_hir.body.loc, scope) as ir_builder:
        flat_params = flatten_block_parameters(aggregate_params)
        try:
            _dispatch_hir_block_inner(preamble, func_hir.body, ir_builder)
        except Exception as e:
            if 'CUTILEIR' in ir_ctx.tile_ctx.config.log_keys:
                highlight_loc = e.loc if hasattr(e, 'loc') else None
                ir_str = "\n".join(op.to_string(highlight_loc=highlight_loc)
                                   for op in ir_builder.ops)
                print(f"==== Partial cuTile IR ====\n\n{ir_str}\n\n", file=sys.stderr)
            raise

    all_flat_params = sum(flat_params, ())
    ret = ir.Block(ir_ctx, all_flat_params, func_hir.body.name, func_hir.body.loc)
    ret.extend(ir_builder.ops)
    return ret


def _create_scope(func_hir: hir.Function, ir_ctx: IRContext, call_site: Loc | None) -> Scope:
    local_scope = LocalScope(func_hir.body.stored_names, ir_ctx)
    return Scope(local_scope, func_hir.frozen_globals, call_site)


def dispatch_hir_block(block: hir.Block):
    _dispatch_hir_block_inner((), block, ir.Builder.get_current())


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


def _dispatch_hir_block_inner(preamble: Sequence[hir.Call],
                              block: hir.Block,
                              builder: ir.Builder):
    state = _State([], None, list(reversed(block.calls)) + list(reversed(preamble)))
    try:
        if not _dispatch_hir_calls(state, builder):
            return ()
        result_vars = tuple(_resolve_operand(x) for x in block.results)
        loc = _add_call_site(block.jump_loc, builder)
        with _wrap_exceptions(loc), builder.change_loc(loc):
            _dispatch_hir_jump(block.jump, result_vars)
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
            assert len(block_results) == 0
            continue_()
        case hir.Jump.BREAK:
            assert len(block_results) == 0
            break_()
        case hir.Jump.RETURN:
            assert len(block_results) == 1
            return_(block_results[0])
        case _: assert False


def _dispatch_hir_calls(state: _State, cur_builder: ir.Builder) -> bool:
    while len(state.todo_stack) > 0:
        with state.next_call() as call:
            loc = _add_call_site(call.loc, cur_builder)
            with _wrap_exceptions(loc), cur_builder.change_loc(loc):
                _dispatch_call(call, cur_builder, state.todo_stack)
            if cur_builder.is_terminated:
                # The current block has been terminated, e.g. by flattening an if-else
                # with a constant condition (`if True: break`). By returning False,
                # we signal that the original jump and block results should be ignored.
                return False
    return True


def _add_call_site(loc: Loc, builder: ir.Builder) -> Loc:
    return loc.with_call_site(builder.scope.call_site)


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
    callee_var = _resolve_operand(call.callee)
    callee, self_arg = _get_callee_and_self(callee_var)
    args = (*self_arg, *(_resolve_operand(x) for x in call.args))
    kwargs = {k: _resolve_operand(v) for k, v in call.kwargs}
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
        _check_recursive_call(builder.loc)
        sig = get_signature(callee)
        for param_name, param in sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD):
                raise TileSyntaxError("Variadic parameters in user-defined"
                                      " functions are not supported")
        callee_hir = get_function_hir(callee, builder.ir_ctx, entry_point=False)

        # Since `todo_stack` is a stack, we push things backwards. First, we push identity()
        # calls to assign the temporary return values back to the original result variables.
        for callee_retval, caller_res in zip(callee_hir.body.results, call.results):
            todo_stack.append(hir.Call((caller_res,), hir.identity, (callee_retval,), (), call.loc))

        # Now we create a fresh Scope for the new function and install it on the builder.
        # We need to reset the builder back to the old scope when we return.
        # For this purpose, we push a call to the special _set_scope stub.
        old_scope = builder.scope
        todo_stack.append(hir.Call((), _set_scope, (old_scope,), (), call.loc))
        builder.scope = _create_scope(callee_hir, builder.ir_ctx, call_site=builder.loc)

        # Now push the function body.
        todo_stack.extend(reversed(callee_hir.body.calls))

        # Finally, call store_var() to bind arguments to parameters.
        for arg, param_name, param_loc in zip(arg_list, callee_hir.param_names,
                                              callee_hir.param_locs, strict=True):
            todo_stack.append(hir.Call((), hir.store_var, (param_name, arg), (), param_loc))


def _is_freshly_defined(var: Var, builder: ir.Builder, first_idx: int):
    return any(var.name == r.name
               for i in range(first_idx, len(builder.ops))
               for r in builder.ops[i].result_vars)


def _check_recursive_call(call_loc: Loc):
    depth = 1
    while call_loc is not None:
        depth += 1
        call_loc = call_loc.call_site
    if depth > MAX_RECURSION_DEPTH:
        raise TileRecursionError(f"Maximum recursion depth ({MAX_RECURSION_DEPTH}) reached"
                                 f" while inlining a function call")


def _get_callee_and_self(callee_var: Var) -> tuple[Any, tuple[()] | tuple[Var]]:
    callee_ty = callee_var.get_type()
    if isinstance(callee_ty, FunctionTy):
        return callee_ty.func, ()
    elif isinstance(callee_ty, BoundMethodTy):
        bound_method = callee_var.get_aggregate()
        assert isinstance(bound_method, BoundMethodValue)
        return callee_ty.func, (bound_method.bound_self,)
    elif isinstance(callee_ty, DTypeConstructor):
        return callee_ty.dtype, ()
    else:
        raise TileTypeError(f"Cannot call an object of type {callee_ty}")


def _resolve_operand(x: hir.Operand) -> Var | hir.Block | Scope:
    if isinstance(x, Var | hir.Block | Scope):
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


def _set_scope(scope): ...


@impl(_set_scope)
def _set_scope_impl(scope):
    assert isinstance(scope, Scope)
    builder = ir.Builder.get_current()
    builder.scope = scope
