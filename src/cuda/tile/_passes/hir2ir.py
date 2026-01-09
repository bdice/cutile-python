# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import inspect
import sys
from contextlib import contextmanager
from typing import Any

from .ast2hir import get_function_hir
from .. import TileTypeError
from .._coroutine_util import resume_after, run_coroutine
from .._exception import Loc, TileSyntaxError, TileInternalError, TileError, TileRecursionError
from .._ir import hir, ir
from .._ir.ir import Var, IRContext, Argument, Scope, LocalScope, BoundMethodValue
from .._ir.op_impl import op_implementations
from .._ir.ops import loosely_typed_const, assign, end_branch, return_, continue_, \
    break_, flatten_block_parameters, store_var
from .._ir.type import FunctionTy, BoundMethodTy, DTypeConstructor
from .._ir.typing_support import get_signature


MAX_RECURSION_DEPTH = 1000


def hir2ir(func_hir: hir.Function,
           args: tuple[Argument, ...],
           ir_ctx: IRContext) -> ir.Block:
    # Run as a coroutine using a software stack, so that we don't exceed Python's recursion limit.
    return run_coroutine(_hir2ir_coroutine(func_hir, args, ir_ctx))


async def _hir2ir_coroutine(func_hir: hir.Function, args: tuple[Argument, ...], ir_ctx: IRContext):
    scope = _create_scope(func_hir, ir_ctx, call_site=None)
    aggregate_params = [
        scope.local.redefine(param_name, param_loc)
        for param_name, param_loc in zip(func_hir.param_names, func_hir.param_locs, strict=True)
    ]

    with ir.Builder(ir_ctx, func_hir.body.loc, scope) as ir_builder:
        try:
            for param_name, var, arg in zip(func_hir.param_names, aggregate_params, args,
                                            strict=True):
                var.set_type(arg.type)
                if arg.is_const:
                    var = loosely_typed_const(arg.const_value)
                    store_var(param_name, var, var.loc)
            flat_params = flatten_block_parameters(aggregate_params)

            await _dispatch_hir_block_inner(func_hir.body, ir_builder)
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


async def dispatch_hir_block(block: hir.Block, cur_builder: ir.Builder | None = None):
    if cur_builder is None:
        cur_builder = ir.Builder.get_current()
    await _dispatch_hir_block_inner(block, cur_builder)


async def _dispatch_hir_block_inner(block: hir.Block, builder: ir.Builder):
    cursor = 0  # Pre-initialize to guarantee it's defined in the `except` block
    try:
        for cursor, call in enumerate(block.calls):
            loc = _add_call_site(call.loc, builder)
            with _wrap_exceptions(loc), builder.change_loc(loc):
                await _dispatch_call(call, builder)
            if builder.is_terminated:
                # The current block has been terminated, e.g. by flattening an if-else
                # with a constant condition (`if True: break`).
                return
        cursor = len(block.calls)

        result_vars = tuple(_resolve_operand(x) for x in block.results)
        loc = _add_call_site(block.jump_loc, builder)
        with _wrap_exceptions(loc), builder.change_loc(loc):
            _dispatch_hir_jump(block.jump, result_vars)
    except Exception:
        if 'CUTILEIR' in builder.ir_ctx.tile_ctx.config.log_keys:
            hir_params = ", ".join(p.name for p in block.params)
            hir_lines = [str(c) for c in block.calls]
            hir_lines.append(block.jump_str())
            hir_str = "\n".join("{}{}".format("--> " if i == cursor else "    ", c)
                                for i, c in enumerate(hir_lines))
            print(f"==== HIR for ^{block.name}({hir_params}) ====\n{hir_str}\n", file=sys.stderr)
        raise


def _dispatch_hir_jump(jump: hir.Jump | None, block_results: tuple[Var, ...]):
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
        case None: pass
        case _: assert False


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


async def _dispatch_call(call: hir.Call, builder: ir.Builder):
    first_idx = len(builder.ops)
    callee_var = _resolve_operand(call.callee)
    callee, self_arg = _get_callee_and_self(callee_var)
    args = (*self_arg, *(_resolve_operand(x) for x in call.args))
    kwargs = {k: _resolve_operand(v) for k, v in call.kwargs}
    arg_list = _bind_args(callee, args, kwargs)

    if callee in op_implementations:
        impl = op_implementations[callee]
        result = impl(*arg_list)
        if impl._is_coroutine:
            result = await result

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

        # Activate a fresh Scope.
        new_scope = _create_scope(callee_hir, builder.ir_ctx, call_site=builder.loc)
        with builder.change_scope(new_scope):
            # Call store_var() to bind arguments to parameters.
            for arg, param_name, param_loc in zip(arg_list, callee_hir.param_names,
                                                  callee_hir.param_locs, strict=True):
                store_var(param_name, arg, param_loc)

            # Dispatch the function body. Use resume_after() to break the call stack
            # and make sure we stay within the Python's recursion limit.
            await resume_after(dispatch_hir_block(callee_hir.body, builder))

        for callee_retval, caller_res in zip(callee_hir.body.results, call.results):
            assign(callee_retval, caller_res)


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
