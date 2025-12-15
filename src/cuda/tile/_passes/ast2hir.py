# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import operator
from contextlib import contextmanager
from enum import Enum, auto
from typing import List, Sequence, Optional, Any, Dict, Type, Callable

from cuda.tile import _datatype as datatype
from cuda.tile._exception import TileSyntaxError, Loc
from cuda.tile._ir.ir import IRContext, Var
from cuda.tile._ir import hir


def get_function_hir(pyfunc: Callable,
                     ir_ctx: IRContext,
                     call_site: Optional[Loc]) -> hir.Function:
    # Get the original function from the decorated function if it exists.
    pyfunc = getattr(pyfunc, "__wrapped__", pyfunc)

    source_lines, first_line = inspect.getsourcelines(pyfunc)
    # The source code of our function could be inside a class, an if-else block etc.
    # This means it can have extra indentation on the left. If we try to give it
    # to ast.parse() as is, we will get a parse error. The common workaround
    # suggested on the web is to filter the source through textwrap.dedent() to remove
    # a common amount of indentation. This is not correct though, because lines that
    # only contain spaces and comments, as well as continuation lines, are not required to
    # be indented. For example, this code is valid:
    #
    #     class A:
    #         def foo(self):
    #              return (100 +
    #     200)
    #
    # The "textwrap.dedent" method would fail to remove the extra indent because the
    # continuation line "200)" is not indented.
    #
    # To handle this properly, we resort to a hack: add one level of indentation to our
    # function and wrap it inside an "if True:" block.
    header_line = "if True:\n "
    indented_source = header_line + " ".join(source_lines)
    mod = ast.parse(indented_source)
    assert len(mod.body) == 1
    assert isinstance(mod.body[0], ast.If)
    assert len(mod.body[0].body) == 1
    func_def = mod.body[0].body[0]
    assert isinstance(func_def, ast.FunctionDef)

    func_globals = dict(pyfunc.__builtins__)
    func_globals.update(pyfunc.__globals__)
    # Add closure variables (from freevars)
    if pyfunc.__closure__:
        for name, cell in zip(pyfunc.__code__.co_freevars, pyfunc.__closure__):
            func_globals[name] = cell.cell_contents

    ctx = _Context(inspect.getfile(pyfunc), first_line, call_site, pyfunc, ir_ctx)
    assert isinstance(func_def, ast.FunctionDef)
    body = _ast2hir(func_def, ctx)
    all_ast_args = _get_all_parameters(func_def, ctx)
    param_names = tuple(p.arg for p in all_ast_args)
    param_locs = tuple(ctx.get_loc(p) for p in all_ast_args)
    body.stored_names.update(param_names)
    return hir.Function(body, param_names, param_locs, func_globals)


# Translate the 1-based line number of the chunk we passed to the AST parser
# to the original 1-based line number in the file.
def _get_source_line_no(first_line_no: int, ast_line_no: int):
    # Why -2?
    #    -1 because both first_line_no and ast_line_no are 1-based;
    #    another -1 to account for the "if True" line that we inserted.
    return first_line_no + ast_line_no - 2


class LoopKind(Enum):
    FOR = auto()
    WHILE = auto()


class _Context:
    def __init__(self, filename: str, first_line: int, call_site: Optional[Loc],
                 function: Callable, ir_ctx: IRContext):
        self.filename = filename
        self.first_line = first_line
        self.entry_point = call_site is None
        self.call_site = call_site
        self.function = function
        self.parent_loops: List[LoopKind] = []
        self.current_loc = Loc.unknown()
        self.current_block: Optional[hir.Block] = None
        self.ir_ctx = ir_ctx

    @contextmanager
    def change_loc(self, loc: ast.AST | Loc):
        old = self.current_loc
        self.current_loc = loc if isinstance(loc, Loc) else self.get_loc(loc)
        try:
            yield
        finally:
            self.current_loc = old

    @contextmanager
    def new_block(self, name: str, params: Sequence[Var] = ()):
        name = self.ir_ctx.make_var(f"^{name}", self.current_loc).name.removeprefix("^")
        old = self.current_block
        new_block = hir.Block(name, tuple(params), calls=[], results=(), jump=None,
                              jump_loc=Loc.unknown(), stored_names=set(), loc=self.current_loc)
        self.current_block = new_block
        try:
            yield self.current_block
        finally:
            self.current_block = old

        if old is not None:
            old.stored_names.update(new_block.stored_names)

    def call(self, callee, args, kwargs=()) -> Var:
        res = self.ir_ctx.make_temp(self.current_loc)
        self.current_block.calls.append(hir.Call((res,), callee, args, kwargs, self.current_loc))
        return res

    def call_void(self, callee, args, kwargs=()) -> None:
        self.current_block.calls.append(hir.Call((), callee, args, kwargs, self.current_loc))

    def set_block_jump(self, jump: hir.Jump, results: Sequence[hir.Operand] = ()):
        assert self.current_block.jump is None
        self.current_block.jump = jump
        self.current_block.results = tuple(results)
        self.current_block.jump_loc = self.current_loc

    def store(self, var_name: str, value: hir.Operand):
        self.call_void(hir.store_var, (var_name, value))
        self.current_block.stored_names.add(var_name)

    def load(self, var_name: str) -> Var:
        # TODO: generate more helpful variable names
        return self.call(hir.load_var, (var_name,))

    def get_loc(self, node: ast.AST) -> Loc:
        line_no = _get_source_line_no(self.first_line, node.lineno)
        last_line_no = _get_source_line_no(self.first_line, node.end_lineno)
        # Subtract 1 from the column offset to correct for an extra level
        # of indentation we inserted for the dummy "if True" block.
        return Loc(line_no, node.col_offset - 1, self.filename,
                   last_line_no, node.end_col_offset - 1, self.function,
                   self.call_site)

    def syntax_error(self, message: str, loc=None) -> TileSyntaxError:
        if loc is None:
            loc = self.current_loc
        elif not isinstance(loc, Loc):
            loc = self.get_loc(loc)
        return TileSyntaxError(message, loc)

    def unsupported_syntax(self, loc=None) -> TileSyntaxError:
        return self.syntax_error("Unsupported syntax", loc=loc)


def _register(mapping, klazz):
    def decorate(f):
        mapping[klazz] = f
        return f
    return decorate


# ================================
# Expressions
# ================================
_expr_handlers: Dict[Type[ast.AST], Callable] = {}


@_register(_expr_handlers, ast.Call)
def _call_expr(call: ast.Call, ctx: _Context) -> Var:
    callee = _expr(call.func, ctx)
    args = tuple(_expr(a, ctx) for a in call.args)
    kwargs = tuple((a.arg, _expr(a.value, ctx)) for a in call.keywords)
    return ctx.call(callee, args, kwargs)


@_register(_expr_handlers, ast.Name)
def _name_expr(name: ast.Name, ctx: Any) -> Var:
    if not isinstance(name.ctx, ast.Load):
        raise ctx.unsupported_syntax()
    return ctx.load(name.id)


_unary_map = {ast.Invert: operator.invert, ast.Not: operator.not_,
              ast.UAdd: operator.pos, ast.USub: operator.neg}


@_register(_expr_handlers, ast.UnaryOp)
def _unary_op(unary: ast.UnaryOp, ctx: _Context) -> Var:
    op_func = _unary_map.get(type(unary.op))
    if op_func is None:
        raise ctx.unsupported_syntax()

    operand = _expr(unary.operand, ctx)
    return ctx.call(op_func, (operand,))


_binop_map = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv, ast.Div: operator.truediv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.BitOr: operator.or_, ast.BitXor: operator.xor, ast.BitAnd: operator.and_,
    ast.LShift: operator.lshift, ast.RShift: operator.rshift,
    ast.MatMult: operator.matmul,
}


@_register(_expr_handlers, ast.BinOp)
def _binop_expr(binop: ast.BinOp, ctx: _Context) -> Var:
    op_func = _binop_map.get(type(binop.op))
    if op_func is None:
        raise ctx.unsupported_syntax()
    lhs = _expr(binop.left, ctx)
    rhs = _expr(binop.right, ctx)
    return ctx.call(op_func, (lhs, rhs))


_cmp_map = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Is: operator.is_, ast.IsNot: operator.is_not,
}


@_register(_expr_handlers, ast.Compare)
def _compare_expr(cmp: ast.Compare, ctx: _Context) -> Var:
    """
    cond = left $op0 comparator0 $op1 comparator1 $op2 comparator2
    -->
    c0 = left $op0 comparator0
    c = if c0:
            c1 = comparator0 $op1 comparator1
            c12 = if c1:
                    c2 = comparator1 $op2 comparator2
                    yield c2
                else:
                    yield c1 # False
            yield c12
        else:
            yield c0 # False
    """
    op_func0 = _cmp_map.get(type(cmp.ops[0]))
    if op_func0 is None:
        raise ctx.unsupported_syntax()
    lhs = _expr(cmp.left, ctx)
    rhs = _expr(cmp.comparators[0], ctx)

    cond0 = ctx.call(op_func0, (lhs, rhs))
    if len(cmp.ops) == 1:
        return cond0

    with ctx.new_block("then") as then_block:
        cmp.left = cmp.comparators[0]
        cmp.comparators = cmp.comparators[1:]
        cmp.ops = cmp.ops[1:]
        cond_right = _expr(cmp, ctx)
        ctx.set_block_jump(hir.Jump.END_BRANCH, (cond_right,))

    with ctx.new_block("else") as else_block:
        ctx.set_block_jump(hir.Jump.END_BRANCH, (cond0,))

    return ctx.call(hir.if_else, (cond0, then_block, else_block))


@_register(_expr_handlers, ast.Attribute)
def _attribute_expr(attr: ast.Attribute, ctx: _Context) -> Var:
    value = _expr(attr.value, ctx)
    return ctx.call(getattr, (value, attr.attr))


@_register(_expr_handlers, ast.Constant)
def _constant_expr(node: ast.Constant, ctx: Any) -> Any:
    # We could just return node.value directly here, but we wrap the constant
    # in a `identity` call in order to preserve location info.
    return ctx.call(hir.identity, (node.value,))


@_register(_expr_handlers, ast.Tuple)
def _tuple_expr(tup: ast.Tuple, ctx: _Context) -> Var:
    items = tuple(_expr(x, ctx) for x in tup.elts)
    return ctx.call(hir.build_tuple, items)


@_register(_expr_handlers, ast.Subscript)
def _subscript_expr(subscript: ast.Subscript, ctx: _Context) -> Var:
    value = _expr(subscript.value, ctx)
    index = _expr(subscript.slice, ctx)
    return ctx.call(operator.getitem, (value, index))


@_register(_expr_handlers, ast.Slice)
def _slice_stmt(slice_: ast.Slice, ctx: _Context) -> Var:
    def get_var(x: ast.AST | None):
        return None if x is None else _expr(x, ctx)
    lower, upper, step = map(get_var, (slice_.lower, slice_.upper, slice_.step))
    return ctx.call(slice, (lower, upper, step))


def _unsupported_expr(expr: ast.AST, ctx: _Context):
    raise ctx.unsupported_syntax()


def _expr(expr: ast.AST, ctx: _Context) -> hir.Operand:
    """Dispatch expression node to appropriate handler"""
    handler = _expr_handlers.get(type(expr), _unsupported_expr)
    with ctx.change_loc(expr):
        return handler(expr, ctx)


# ================================
# Statements
# ================================
_stmt_handlers: Dict[Type[ast.AST], Callable] = {}


@_register(_stmt_handlers, ast.Assign)
def _assign_stmt(assign: ast.Assign, ctx: _Context) -> None:
    value = _expr(assign.value, ctx)
    for target in reversed(assign.targets):
        with ctx.change_loc(target):
            if isinstance(target, ast.Name):
                ctx.store(target.id, value)
            elif isinstance(target, ast.Tuple):
                for i, el in enumerate(target.elts):
                    with ctx.change_loc(el):
                        if not isinstance(el, ast.Name):
                            raise ctx.unsupported_syntax()
                        item_var = ctx.call(operator.getitem, (value, i), )
                        ctx.store(el.id, item_var)
            else:
                raise ctx.unsupported_syntax()


@_register(_stmt_handlers, ast.AugAssign)
def _aug_assign_stmt(aug: ast.AugAssign, ctx: _Context):
    if not isinstance(aug.target, ast.Name):
        raise ctx.unsupported_syntax(aug.target)
    op_func = _binop_map.get(type(aug.op))
    if op_func is None:
        raise ctx.unsupported_syntax()
    lhs = ctx.load(aug.target.id)
    rhs = _expr(aug.value, ctx)
    res = ctx.call(op_func, (lhs, rhs))
    ctx.store(aug.target.id, res)


@_register(_stmt_handlers, ast.Expr)
def _expr_stmt(expr: ast.Expr, ctx: _Context):
    _expr(expr.value, ctx)


def _propagate_return(ctx: _Context):
    if ctx.entry_point:
        return
    # In order to propagate an early return, insert the following:
    #    if $returning:
    #        break
    flag = ctx.load("$returning")
    with ctx.new_block("then") as then_block:
        ctx.set_block_jump(hir.Jump.BREAK)
    with ctx.new_block("else") as else_block:
        ctx.set_block_jump(hir.Jump.END_BRANCH)
    ctx.call_void(hir.if_else, (flag, then_block, else_block))


@_register(_stmt_handlers, ast.For)
def _for_stmt(stmt: ast.For, ctx: _Context):
    if len(stmt.orelse) > 0:
        raise ctx.syntax_error("'for-else' is not supported", loc=stmt.orelse[0])

    iterable = _expr(stmt.iter, ctx)
    if not isinstance(stmt.target, ast.Name):
        raise ctx.unsupported_syntax(stmt.target)

    ctx.parent_loops.append(LoopKind.FOR)
    induction_var = ctx.ir_ctx.make_var(stmt.target.id, ctx.get_loc(stmt.target))
    with ctx.new_block("body", params=(induction_var,)) as body_block:
        with ctx.change_loc(induction_var.loc):
            ctx.store(stmt.target.id, induction_var)
        _stmt_list(stmt.body, ctx)
        if body_block.jump is None:
            ctx.set_block_jump(hir.Jump.CONTINUE)
    ctx.parent_loops.pop()

    ctx.call_void(hir.loop, (body_block, iterable))


def _cast_cond_to_bool(cond: Var, ctx: _Context) -> Var:
    with ctx.change_loc(cond.loc):
        return ctx.call(datatype.bool_, (cond,))


@_register(_stmt_handlers, ast.While)
def _while_stmt(stmt: ast.While, ctx: _Context):
    if len(stmt.orelse) > 0:
        raise ctx.syntax_error("'while-else' is not supported", loc=stmt.orelse[0])

    with ctx.new_block("body") as body_block:
        # Add "if cond: pass; else: break"
        cond = _expr(stmt.test, ctx)
        cond = _cast_cond_to_bool(cond, ctx)

        with ctx.new_block("then") as then_block:
            ctx.set_block_jump(hir.Jump.END_BRANCH)

        with ctx.new_block("else") as else_block:
            ctx.set_block_jump(hir.Jump.BREAK)

        ctx.call_void(hir.if_else, (cond, then_block, else_block))

        ctx.parent_loops.append(LoopKind.WHILE)
        _stmt_list(stmt.body, ctx)
        if body_block.jump is None:
            ctx.set_block_jump(hir.Jump.CONTINUE)
        ctx.parent_loops.pop()

    ctx.call_void(hir.loop, (body_block, None))
    _propagate_return(ctx)


@_register(_expr_handlers, ast.BoolOp)
def _boolop_expr(boolop: ast.BoolOp, ctx: _Context) -> Var:
    assert len(boolop.values) >= 2
    cond0 = _expr(boolop.values[0], ctx)
    cond0 = _cast_cond_to_bool(cond0, ctx)

    if isinstance(boolop.op, ast.And):
        """
        cond = cond0() and cond1():
        -->
        c0 = cond0()
        c = if c0:
            c1 = cond1()
            yield c1
        else:
            yield c0 # False
        """
        with ctx.new_block("then") as then_block:
            if len(boolop.values) > 2:
                # Consecutive operations with the same operator, such as a or b or c,
                # are collapsed into one node with several values.
                boolop.values = boolop.values[1:]
                cond1 = _expr(boolop, ctx)
            else:
                cond1 = _expr(boolop.values[1], ctx)
            cond1 = _cast_cond_to_bool(cond1, ctx)
            ctx.set_block_jump(hir.Jump.END_BRANCH, results=(cond1,))

        with ctx.new_block("else") as else_block:
            ctx.set_block_jump(hir.Jump.END_BRANCH, results=(cond0,))

        return ctx.call(hir.if_else, (cond0, then_block, else_block))
    elif isinstance(boolop.op, ast.Or):
        """
        cond = cond0() or cond1():
        -->
        c0 = cond0()
        c = if c0:
            yield c0
        else:
            c1 = cond1()
            yield c1
        """
        with ctx.new_block("then") as then_block:
            ctx.set_block_jump(hir.Jump.END_BRANCH, results=(cond0,))

        with ctx.new_block("else") as else_block:
            if len(boolop.values) > 2:
                boolop.values = boolop.values[1:]
                cond1 = _expr(boolop, ctx)
            else:
                cond1 = _expr(boolop.values[1], ctx)
            cond1 = _cast_cond_to_bool(cond1, ctx)
            ctx.set_block_jump(hir.Jump.END_BRANCH, results=(cond1,))

        return ctx.call(hir.if_else, (cond0, then_block, else_block))
    else:
        raise ctx.unsupported_syntax()


@_register(_expr_handlers, ast.IfExp)
def _ifexp_expr(ifexp: ast.IfExp, ctx: _Context) -> Var:
    cond = _expr(ifexp.test, ctx)
    cond = _cast_cond_to_bool(cond, ctx)

    with ctx.new_block("then") as then_block:
        then_val = _expr(ifexp.body, ctx)
        ctx.set_block_jump(hir.Jump.END_BRANCH, results=(then_val,))

    with ctx.new_block("else") as else_block:
        else_val = _expr(ifexp.orelse, ctx)
        ctx.set_block_jump(hir.Jump.END_BRANCH, results=(else_val,))

    return ctx.call(hir.if_else, (cond, then_block, else_block))


@_register(_stmt_handlers, ast.If)
def _if_stmt(stmt: ast.If, ctx: _Context) -> None:
    cond = _expr(stmt.test, ctx)
    cond = _cast_cond_to_bool(cond, ctx)

    with ctx.new_block("then") as then_block:
        _stmt_list(stmt.body, ctx)
        if then_block.jump is None:
            ctx.set_block_jump(hir.Jump.END_BRANCH)

    with ctx.new_block("else") as else_block:
        _stmt_list(stmt.orelse, ctx)
        if else_block.jump is None:
            ctx.set_block_jump(hir.Jump.END_BRANCH)

    ctx.call_void(hir.if_else, (cond, then_block, else_block))


@_register(_stmt_handlers, ast.Continue)
def _continue_stmt(stmt: ast.Continue, ctx: _Context) -> None:
    ctx.set_block_jump(hir.Jump.CONTINUE)


@_register(_stmt_handlers, ast.Break)
def _break_stmt(stmt: ast.Break, ctx: _Context) -> None:
    if ctx.parent_loops and ctx.parent_loops[-1] is LoopKind.FOR:
        raise ctx.syntax_error("Break in a for loop is not supported")
    ctx.set_block_jump(hir.Jump.BREAK)


@_register(_stmt_handlers, ast.Return)
def _return_stmt(stmt: ast.Return, ctx: _Context) -> None:
    if ctx.parent_loops and ctx.parent_loops[-1] is LoopKind.FOR:
        raise ctx.syntax_error("Returning from a for loop is not supported")

    return_val = None if stmt.value is None else _expr(stmt.value, ctx)
    if ctx.entry_point:
        ctx.set_block_jump(hir.Jump.RETURN, (return_val,))
    else:
        ctx.store("$retval", return_val)
        ctx.store("$returning", True)
        ctx.set_block_jump(hir.Jump.BREAK)


@_register(_stmt_handlers, ast.Pass)
def _pass_stmt(stmt: ast.Pass, ctx: _Context) -> None:
    pass


def _unsupported_stmt(stmt: ast.AST, ctx: _Context) -> None:
    raise ctx.unsupported_syntax()


def _stmt(stmt: ast.AST, ctx: _Context) -> None:
    handler = _stmt_handlers.get(type(stmt), _unsupported_stmt)
    with ctx.change_loc(stmt):
        handler(stmt, ctx)


def _stmt_list(statements: Sequence[ast.stmt], ctx: _Context):
    statements = iter(statements)
    for stmt in statements:
        _stmt(stmt, ctx)
        if ctx.current_block.jump is not None:
            break

    # Process "dead" statements, i.e. the ones after a jump ("continue"/"break"/"return").
    # We still need to look at them in order to figure out the set of local variables.
    # So create a throwaway block to store these into.
    with ctx.new_block("dummy"):
        for stmt in statements:
            _stmt(stmt, ctx)


def _get_all_parameters(func_def: ast.FunctionDef, ctx: _Context) -> List[ast.arg]:
    for a in (func_def.args.vararg, func_def.args.kwarg):
        if a is not None:
            raise ctx.syntax_error(
                "Variadic parameters in user-defined functions are not supported", a)
    all_args = []
    for arg in func_def.args.posonlyargs:
        all_args.append(arg)
    for arg in func_def.args.args:
        all_args.append(arg)
    for arg in func_def.args.kwonlyargs:
        all_args.append(arg)
    return all_args


def _ast2hir(func_def: ast.FunctionDef, ctx: _Context) -> hir.Block:
    with ctx.change_loc(func_def), ctx.new_block(func_def.name) as root_block:
        if ctx.entry_point:
            _stmt_list(func_def.body, ctx)
            # Add a Return jump to the root block if it doesn't have one
            if root_block.jump is None:
                ctx.set_block_jump(hir.Jump.RETURN, results=(None,))
        else:
            # To enable early returns in a helper function, wrap the body in a loop.
            # Thus, we can use "break" to implement the return statement.
            ctx.store("$returning", False)
            with ctx.new_block("wrapped_body") as body_block:
                _stmt_list(func_def.body, ctx)
                if body_block.jump is None:
                    ctx.store("$retval", None)
                    ctx.set_block_jump(hir.Jump.BREAK)

            ctx.call_void(hir.loop, (body_block, None))
            retval = ctx.load("$retval")
            root_block.results = (retval,)
    return root_block
