# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from contextlib import contextmanager
from typing import Dict, Tuple, Sequence, Any, Optional

from cuda.tile import _datatype as datatype
from cuda.tile._datatype import get_signedness
from cuda.tile import DType, RoundingMode, PaddingMode
import cuda.tile._bytecode as bc
from cuda.tile._compiler_options import CompilerOptions
from cuda.tile._exception import TileInternalError, TileError, FunctionDesc
from cuda.tile._ir.ir import Block, Loc, Var, IRContext
from cuda.tile._ir.ops_utils import (
    padding_mode_to_bytecode, rounding_mode_to_bytecode,
    get_default_rounding_mode, get_dtype,
)
from cuda.tile._ir.type import Type, TileTy, PointerTy, TokenTy, TupleTy, ArrayTy, SizeTy


def dtype_typeid(tt: bc.TypeTable, dtype: datatype.DType) -> bc.TypeId:
    return tt.simple(dtype._bytecode_type)


def tensor_view_typeid(tt: bc.TypeTable, array_ty: ArrayTy) -> bc.TypeId:
    dtype = dtype_typeid(tt, array_ty.dtype)
    shape = [x.bytecode_value for x in array_ty.shape]
    strides = [x.bytecode_value for x in array_ty.strides]
    return tt.tensor_view(dtype, shape, strides)


def tensor_view_typeid_for_list(tt: bc.TypeTable, item_size_words: int) -> bc.TypeId:
    shape = [bc.DYNAMIC_SHAPE, item_size_words]
    strides = [item_size_words, 1]
    return tt.tensor_view(tt.I64, shape, strides)


def typeid(tt: bc.TypeTable, ty: Type, wrap_scalars: bool = True) -> bc.TypeId:
    if isinstance(ty, TileTy):
        dtype = typeid(tt, ty.dtype, wrap_scalars=False)
        shape = [x.value for x in ty.shape]
        return tt.tile(dtype, shape)
    elif isinstance(ty, datatype.DType):
        dtype = dtype_typeid(tt, ty)
        return tt.tile(dtype, ()) if wrap_scalars else dtype
    elif isinstance(ty, PointerTy):
        pointee = typeid(tt, ty.pointee_type, wrap_scalars=False)
        return tt.pointer(pointee)
    elif isinstance(ty, TokenTy):
        return tt.Token
    else:
        raise NotImplementedError(f"Lowering type '{ty}' is not supported")


def get_list_item_repr_size_in_words(item_ty: Type) -> int:
    if isinstance(item_ty, ArrayTy):
        # Base pointer + shape + strides
        return 1 + 2 * item_ty.ndim
    else:
        raise NotImplementedError(f"List of type '{item_ty}' are not supported")


def get_list_partition_view_tile_size(item_size_words: int) -> int:
    # Round up the item size to the nearest power of two
    return 1 << (item_size_words - 1).bit_length()


# Encode a single int/float value according to the MLIR "DenseElementsAttr" splat format.
def _constant_to_bytes(value: int | float, dtype: DType) -> bytes:
    if dtype == datatype.bool_:
        # Note that MLIR requires "0xFF" for "True" value.
        return b"\xff" if value else b"\x00"
    elif datatype.is_integral(dtype):
        return int(value).to_bytes((dtype.bitwidth + 7) // 8, "little", signed=value < 0)
    elif datatype.is_float(dtype) or datatype.is_restricted_float(dtype):
        # Note that TF32 is stored as 3 bytes despite the "32" in its name.
        # Its float_bit_size() is 19 bits, which is rounded up to 24 bits.
        bits = bc.float_to_bits(value, dtype._bytecode_type)
        bit_size = bc.float_bit_size(dtype._bytecode_type)
        return bits.to_bytes((bit_size + 7) // 8, "little")
    else:
        raise TypeError(f"Cannot make a constant out of {dtype}")


def _get_type_conversion_encoder(from_dtype: Type, to_dtype: Type):

    def kind(t):
        if datatype.is_float(t) or datatype.is_restricted_float(t):
            return 'f'
        if datatype.is_integral(t) or datatype.is_boolean(t):
            return 'si' if datatype.is_signed(t) else 'ui'
        raise TileInternalError(f'Unsupported dtype: {t}')

    from_kind, to_kind = kind(from_dtype), kind(to_dtype)
    round_to_float = rounding_mode_to_bytecode[get_default_rounding_mode()]
    partial = functools.partial
    match from_kind, to_kind:
        case 'f', 'f': return partial(bc.encode_FToFOp,
                                      rounding_mode=bc.RoundingMode.NEAREST_EVEN)
        case 'f', 'si': return partial(bc.encode_FToIOp,
                                       signedness=bc.Signedness.Signed,
                                       rounding_mode=bc.RoundingMode.NEAREST_INT_TO_ZERO)
        case 'f', 'ui': return partial(bc.encode_FToIOp,
                                       signedness=bc.Signedness.Unsigned,
                                       rounding_mode=bc.RoundingMode.NEAREST_INT_TO_ZERO)
        case 'si', 'f': return partial(bc.encode_IToFOp,
                                       signedness=bc.Signedness.Signed,
                                       rounding_mode=round_to_float)
        case 'ui', 'f': return partial(bc.encode_IToFOp,
                                       signedness=bc.Signedness.Unsigned,
                                       rounding_mode=round_to_float)

    if from_dtype.bitwidth < to_dtype.bitwidth or from_dtype is datatype.bool_:
        assert from_kind in ("si", "ui")
        return partial(bc.encode_ExtIOp, signedness=get_signedness(from_dtype))
    elif from_dtype.bitwidth > to_dtype.bitwidth:
        return partial(bc.encode_TruncIOp, overflow=bc.IntegerOverflow.NONE)
    elif from_kind in ("si", "ui") and to_kind in ("si", "ui"):
        # Signed-to-unsigned or unsigned-to-signed conversion without changing bitwidth is a no-op
        return lambda _builder, _type, val: val
    raise NotImplementedError(f"Type coversion from {from_dtype} to {to_dtype} not implemented")


def convert_dtype(ctx: "BytecodeContext", val: bc.Value,
                  fromty: Type, toty: Type) -> bc.Value:
    from_dtype = fromty.dtype if isinstance(fromty, TileTy) else fromty
    to_dtype = toty.dtype if isinstance(toty, TileTy) else toty
    toty_id = typeid(ctx.type_table, toty)
    if to_dtype == datatype.bool_ and datatype.is_integral(from_dtype):
        # TruncIOp is not doing pytorch style boolean casting (x != 0)
        # We have to use CmpIOp instead
        zero = ctx.constant(0, fromty)
        return bc.encode_CmpIOp(
            ctx.builder,
            result_type=toty_id,
            lhs=val,
            rhs=zero,
            comparison_predicate=bc.ComparisonPredicate.NOT_EQUAL,
            signedness=datatype.get_signedness(from_dtype))
    else:
        encoder = _get_type_conversion_encoder(from_dtype, to_dtype)
        return encoder(ctx.builder, toty_id, val)


def _broadcast_shape(ctx: "BytecodeContext",
                     val: bc.Value, fromty: TileTy, toty: TileTy):
    if len(fromty.shape) < len(toty.shape):
        # prepend 1s if input_shape have fewer dimensions
        diff = len(toty.shape) - len(fromty.shape)
        new_shape = TupleTy(tuple([SizeTy(1)] * diff) + tuple(fromty.shape))
        reshaped_ty = TileTy(fromty.dtype, new_shape)
        reshaped_ty_id = typeid(ctx.type_table, reshaped_ty)
        val = bc.encode_ReshapeOp(ctx.builder, reshaped_ty_id, val)
        fromty = reshaped_ty

    if fromty.shape != toty.shape:
        broadcasted_ty = TileTy(fromty.dtype, toty.shape)
        broadcasted_ty_id = typeid(ctx.type_table, broadcasted_ty)
        val = bc.encode_BroadcastOp(ctx.builder, broadcasted_ty_id, val)
        fromty = broadcasted_ty
    return val, fromty


Limits = Tuple[float, float] | Tuple[int, int] | Tuple[bool, bool]


def _get_min_max(dtype: datatype.DType) -> Limits:
    use_float = datatype.is_float(dtype)
    if use_float:
        if dtype in [datatype.float16, datatype.bfloat16, datatype.float32, datatype.float64]:
            return -float("inf"), float("inf")
        else:
            raise NotImplementedError(f"Unsupported float dtype: {dtype}")
    elif datatype.is_signed(dtype):
        return -(1 << (dtype.bitwidth-1)), (1 << (dtype.bitwidth-1)) - 1
    else:
        return 0, (1 << dtype.bitwidth) - 1


def _get_reduce_element_type_and_attr(
        reduce_fn: str, input_ty: Type, tt: bc.TypeTable,
) -> Tuple[bc.TypeId, bc.TaggedAttribute]:
    # Identity serves as the initial accumulator value.
    # Note identity value is not used with the current power-of-2 assumption.
    if reduce_fn in ["max", "argmax"]:
        id_val = _get_min_max(input_ty.dtype)[0]
    elif reduce_fn in ["min", "argmin"]:
        id_val = _get_min_max(input_ty.dtype)[1]
    elif reduce_fn == "add":
        id_val = -0.0
    elif reduce_fn == "mul":
        id_val = 1.0
    else:
        raise NotImplementedError(f"Unsupported reduction: {reduce_fn}")

    input_dtype_bc = input_ty.dtype._bytecode_type
    input_dtype_id = tt.simple(input_dtype_bc)
    if datatype.is_float(input_ty.dtype):
        attr = bc.Float(float(id_val), input_dtype_bc, tt)
    else:
        attr = bc.Integer(input_dtype_id, input_ty.dtype.bitwidth, int(id_val))
    return input_dtype_id, attr


def lower_reduce(ctx: "BytecodeContext", x: bc.Value, input_ty: Type, normalized_axis: int,
                 output_ty: Type, reduce_fn: str,
                 rounding_mode: Optional[RoundingMode], flush_to_zero: bool) -> bc.Value:
    tt = ctx.type_table
    if rounding_mode is None:
        rounding_mode = get_default_rounding_mode()
    rounding_mode_bc = rounding_mode_to_bytecode[rounding_mode]

    element_typeid, identity_attr = _get_reduce_element_type_and_attr(
            reduce_fn, input_ty, ctx.type_table)
    reduce_output_typeid = typeid(tt, output_ty)
    nested_builder = bc.encode_ReduceOp(
        ctx.builder,
        result_types=[reduce_output_typeid],
        operands=[x],
        dim=normalized_axis,
        identities=[identity_attr]
    )

    kind = "float" if datatype.is_float(input_ty.dtype) else "int"
    tensor_element_typeid = ctx.type_table.tile(element_typeid, ())

    with nested_builder.new_block((tensor_element_typeid, tensor_element_typeid)) as (a, b):
        match reduce_fn, kind:
            case "add", "int":
                res = bc.encode_AddIOp(ctx.builder, tensor_element_typeid, a, b,
                                       overflow=bc.IntegerOverflow.NONE)
            case "add", "float":
                res = bc.encode_AddFOp(ctx.builder, tensor_element_typeid, a, b,
                                       rounding_mode=rounding_mode_bc,
                                       flush_to_zero=flush_to_zero)
            case "max", "int":
                res = bc.encode_MaxIOp(ctx.builder, tensor_element_typeid, a, b,
                                       signedness=datatype.get_signedness(output_ty.dtype))
            case "max", "float":
                res = bc.encode_MaxFOp(ctx.builder, tensor_element_typeid, a, b,
                                       propagate_nan=False,
                                       flush_to_zero=flush_to_zero)
            case "min", "int":
                res = bc.encode_MinIOp(ctx.builder, tensor_element_typeid, a, b,
                                       signedness=datatype.get_signedness(output_ty.dtype))
            case "min", "float":
                res = bc.encode_MinFOp(ctx.builder, tensor_element_typeid, a, b,
                                       propagate_nan=False,
                                       flush_to_zero=flush_to_zero)
            case "mul", "int":
                res = bc.encode_MulIOp(ctx.builder, tensor_element_typeid, a, b,
                                       overflow=bc.IntegerOverflow.NONE)
            case "mul", "float":
                res = bc.encode_MulFOp(ctx.builder, tensor_element_typeid, a, b,
                                       rounding_mode=rounding_mode_bc,
                                       flush_to_zero=flush_to_zero)
            case _:
                raise NotImplementedError(f"Unsupported reduction: {reduce_fn}")
        bc.encode_YieldOp(ctx.builder, [res])

    [reduce_res] = nested_builder.done()
    return reduce_res


def _get_reduce_indices(
    ctx: "BytecodeContext", input_shape: Tuple[SizeTy, ...], output_ty: TileTy,
    normalized_axis: int,
) -> bc.Value:
    tt = ctx.type_table
    # iota
    indices_ty = TileTy(
        output_ty.dtype, TupleTy(tuple([input_shape[normalized_axis]]))
    )
    indices = bc.encode_IotaOp(ctx.builder, typeid(tt, indices_ty))

    # prepend and append 1 until normalized_axis is at the right dimension.
    new_shape = [SizeTy(1)] * len(input_shape)
    new_shape[normalized_axis] = input_shape[normalized_axis]
    indices_ty = TileTy(
        output_ty.dtype, TupleTy(new_shape)
    )
    indices = bc.encode_ReshapeOp(ctx.builder, typeid(tt, indices_ty), indices)
    # broadcast to input_shape
    to_indices_ty = TileTy(output_ty.dtype, TupleTy(input_shape))
    res, _ = _broadcast_shape(ctx, indices, indices_ty, to_indices_ty)
    return res


def encode_comparison(builder: bc.CodeBuilder, fn: str, lhs: bc.Value, rhs: bc.Value,
                      dtype: Type, result_typeid: bc.TypeId) -> bc.Value:
    match fn:
        case "eq": pred = bc.ComparisonPredicate.EQUAL
        case "ne": pred = bc.ComparisonPredicate.NOT_EQUAL
        case "ge": pred = bc.ComparisonPredicate.GREATER_THAN_OR_EQUAL
        case "gt": pred = bc.ComparisonPredicate.GREATER_THAN
        case "le": pred = bc.ComparisonPredicate.LESS_THAN_OR_EQUAL
        case "lt": pred = bc.ComparisonPredicate.LESS_THAN

    if datatype.is_float(dtype):
        return bc.encode_CmpFOp(builder,
                                result_type=result_typeid,
                                comparison_predicate=pred,
                                comparison_ordering=bc.ComparisonOrdering.ORDERED,
                                lhs=lhs, rhs=rhs)
    elif datatype.is_integral(dtype) or datatype.is_boolean(dtype):
        return bc.encode_CmpIOp(builder,
                                result_type=result_typeid,
                                comparison_predicate=pred,
                                signedness=datatype.get_signedness(dtype),
                                lhs=lhs, rhs=rhs)
    else:
        raise TileInternalError(f'Unexpected dtype: {dtype}')


def lower_reduce_argmax_argmin(
    ctx: "BytecodeContext", x: bc.Value, input_ty: Type,
    normalized_axis: Optional[int],
    output_ty: Type, reduce_fn: str
) -> bc.Value:
    tt = ctx.type_table
    element_typeid, identity_attr = _get_reduce_element_type_and_attr(reduce_fn, input_ty, tt)
    index_typeid = typeid(tt, output_ty.dtype, wrap_scalars=False)

    if normalized_axis is None:
        # Get reduce output type and reduce indices type
        reduce_value_typeid = tt.tile(element_typeid, ())
        reduce_output_typeid = tt.tile(index_typeid, ())
        # Get reduce element value and reduce indices value in flattened form
        flattened_x_typeid = tt.tile(element_typeid, (input_ty.numel,))
        x = bc.encode_ReshapeOp(ctx.builder, flattened_x_typeid, x)
        indices_typeid = tt.tile(index_typeid, (input_ty.numel,))
        indices = bc.encode_IotaOp(ctx.builder, indices_typeid)
        normalized_axis = 0
    else:
        # Get reduce output type and reduce indices type
        reduce_value_typeid = tt.tile(element_typeid, output_ty.shape_value)
        reduce_output_typeid = typeid(tt, output_ty)
        # Get reduce indices value
        indices = _get_reduce_indices(ctx, input_ty.shape, output_ty, normalized_axis)

    index_identity_attr = bc.Integer(index_typeid, output_ty.dtype.bitwidth, 0)

    # Reduce with x and indices
    nested_builder = bc.encode_ReduceOp(
        ctx.builder,
        result_types=[reduce_value_typeid, reduce_output_typeid],
        operands=[x, indices],
        dim=normalized_axis,
        identities=[identity_attr, index_identity_attr])

    element_0d_typeid = tt.tile(element_typeid, ())
    index_0d_typeid = tt.tile(index_typeid, ())
    block_arg_typeids = [element_0d_typeid, element_0d_typeid, index_0d_typeid, index_0d_typeid]

    with nested_builder.new_block(block_arg_typeids) as (a, b, a_idx, b_idx):
        # When the values are not equal, yield the larger value for argmax,
        # and the smaller value for argmin.
        # When the values are equal, yield the smaller index.
        fn = "gt" if reduce_fn == "argmax" else "lt"
        bool_0d_typeid = tt.tile(tt.I1, ())
        cmp_strict_gt_or_lt_res = encode_comparison(
                ctx.builder, fn, a, b, input_ty.dtype, bool_0d_typeid)
        cmp_equal_res = encode_comparison(
                ctx.builder, "eq", a, b, input_ty.dtype, bool_0d_typeid)
        cmp_index_res = encode_comparison(
                ctx.builder, "lt", a_idx, b_idx, output_ty.dtype, bool_0d_typeid)
        cmp_equal_res = bc.encode_AndIOp(ctx.builder, bool_0d_typeid, cmp_equal_res, cmp_index_res)
        cmp_res = bc.encode_OrIOp(
                ctx.builder, bool_0d_typeid, cmp_strict_gt_or_lt_res, cmp_equal_res)
        res = bc.encode_SelectOp(ctx.builder, element_0d_typeid, cmp_res, a, b)
        index = bc.encode_SelectOp(ctx.builder, index_0d_typeid, cmp_res, a_idx, b_idx)
        bc.encode_YieldOp(ctx.builder, (res, index))

    _, reduce_res = nested_builder.done()
    return reduce_res


def lower_scan(ctx: "BytecodeContext", x: bc.Value, input_ty: Type,
               normalized_axis: int, reverse: bool,
               output_ty: Type, scan_fn: str, rounding_mode: Optional[RoundingMode],
               flush_to_zero: bool) -> bc.Value:
    use_float = True if datatype.is_float(output_ty.dtype) else False
    if scan_fn == "mul":
        id_val = 1.0
    elif scan_fn == "add":
        id_val = -0.0
    else:
        raise NotImplementedError(f"Unsupported scan function: {scan_fn}")
    element_dtype = get_dtype(input_ty)
    tt = ctx.type_table
    element_type_id = typeid(tt, element_dtype, wrap_scalars=False)
    if use_float:
        identity_attr = bc.Float(id_val, element_dtype._bytecode_type, tt)
    else:
        identity_attr = bc.Integer(element_type_id, element_dtype.bitwidth, int(id_val))

    scan_output_typeid = typeid(tt, output_ty)
    nested_builder = bc.encode_ScanOp(ctx.builder,
                                      result_types=[scan_output_typeid],
                                      operands=[x],
                                      dim=normalized_axis,
                                      reverse=reverse,
                                      identities=[identity_attr])

    element_tile_typeid = tt.tile(element_type_id, ())
    with nested_builder.new_block((element_tile_typeid, element_tile_typeid)) as (a, b):
        rounding_mode_bc = rounding_mode_to_bytecode[rounding_mode]
        match scan_fn, use_float:
            case "add", True:
                res = bc.encode_AddFOp(ctx.builder, element_tile_typeid, a, b,
                                       rounding_mode=rounding_mode_bc,
                                       flush_to_zero=flush_to_zero)
            case "add", False:
                res = bc.encode_AddIOp(ctx.builder, element_tile_typeid, a, b,
                                       overflow=bc.IntegerOverflow.NONE)
            case "mul", True:
                res = bc.encode_MulFOp(ctx.builder, element_tile_typeid, a, b,
                                       rounding_mode=rounding_mode_bc,
                                       flush_to_zero=flush_to_zero)
            case "mul", False:
                res = bc.encode_MulIOp(ctx.builder, element_tile_typeid, a, b,
                                       overflow=bc.IntegerOverflow.NONE)
            case _:
                raise NotImplementedError(f"Unsupported scan function {scan_fn}")
        bc.encode_YieldOp(ctx.builder, [res])
    scan_res, = nested_builder.done()
    return scan_res


class DebugAttrMap:
    def __init__(self, debug_attr_table: bc.DebugAttrTable, linkage_name: str, anonymize: bool):
        self._subprogram_cache = {}
        self._debug_attr_table = debug_attr_table
        self._linkage_name = linkage_name
        self._anonymize = anonymize

    def get_subprogram(self, func_desc: FunctionDesc) -> bc.DebugAttrId:
        try:
            return self._subprogram_cache[func_desc]
        except KeyError:
            pass

        func_dirname, func_basename = os.path.split(func_desc.filename)
        file_attr = self._debug_attr_table.file(func_basename, func_dirname)
        compile_unit_attr = self._debug_attr_table.compile_unit(file_attr)
        ret = self._debug_attr_table.subprogram(
            file=file_attr,
            line=func_desc.line,
            name=func_desc.name,
            linkage_name=self._linkage_name,
            compile_unit=compile_unit_attr,
            scope_line=func_desc.line,
        )
        self._subprogram_cache[func_desc] = ret
        return ret

    def get_debugattr(self, loc: Loc) -> bc.DebugAttrId:
        if self._anonymize:
            return bc.MISSING_DEBUG_ATTR_ID

        subprogram = self.get_subprogram(loc.function)
        attr = self._debug_attr_table.loc(subprogram, loc.filename, loc.line, loc.col)
        if loc.call_site is not None:
            caller_loc = self.get_debugattr(loc.call_site)
            attr = self._debug_attr_table.call_site(attr, caller_loc)
        return attr


class BytecodeContext:
    def __init__(self,
                 builder: bc.CodeBuilder,
                 type_table: bc.TypeTable,
                 debug_attr_map: DebugAttrMap,
                 global_section: bc.GlobalSection,
                 ir_ctx: IRContext,
                 sm_arch: str) -> None:
        self.builder = builder
        self.type_table = type_table
        self._debug_attr_map = debug_attr_map
        self.global_section = global_section
        self._typemap: Dict[str, Type] = ir_ctx.typemap
        self._constants: Dict[str, Any] = ir_ctx.constants
        self._value_map: Dict[str, bc.Value] = {}
        self._array_base_ptr: Dict[str, bc.Value] = {}
        self._list_partition_views: Dict[str, bc.Value] = {}
        self.sm_arch = sm_arch
        self.innermost_loop = None

    @contextmanager
    def loc(self, loc: Loc):
        debug_attr_id = self._debug_attr_map.get_debugattr(loc)
        with loc, self.builder.debug_attr(debug_attr_id):
            yield

    @contextmanager
    def enter_loop(self, loop):
        old = self.innermost_loop
        self.innermost_loop = loop
        try:
            yield
        finally:
            self.innermost_loop = old

    def typeof(self, var: Var) -> Type:
        return self._typemap[var.name]

    def typeid_of(self, var: Var) -> bc.TypeId:
        return typeid(self.type_table, self.typeof(var))

    def is_constant(self, var: Var) -> bool:
        return var.name in self._constants

    def get_constant(self, var: Var):
        return self._constants[var.name]

    def get_constant_or_default(self, var: Var, default=None):
        return self._constants.get(var.name, default)

    def get_value(self, var: Var) -> bc.Value:
        return self._value_map[var.name]

    def get_value_allow_undefined(self, var: Var, ty: Type) -> bc.Value:
        return self.undefined_value(ty) if var.is_undefined() else self.get_value(var)

    def get_optional_value(self, var: Var) -> Optional[bc.Value]:
        if var.name in self._constants and self._constants[var.name] is None:
            return None
        else:
            return self.get_value(var)

    def set_value(self, var: Var, value: bc.Value) -> None:
        name = var.name
        if name in self._value_map:
            raise ValueError(f"Variable {name} is already in the value map")
        self._value_map[name] = value

    def cast(self, val: bc.Value, fromty: Type, toty: Type) -> bc.Value:
        if fromty == toty:
            return val
        if isinstance(fromty, datatype.DType):
            fromty = TileTy(fromty, TupleTy([]))
        if isinstance(toty, datatype.DType):
            toty = TileTy(toty, TupleTy([]))
        if fromty.shape != toty.shape:
            val, fromty = _broadcast_shape(self, val, fromty, toty)
        if fromty.dtype != toty.dtype:
            val = convert_dtype(self, val, fromty, toty)
        return val

    def bitcast(self, value: bc.Value, fromty: Type, toty: Type) -> bc.Value:
        if fromty == toty:
            return value
        if isinstance(fromty, datatype.DType):
            fromty = TileTy(fromty, TupleTy([]))
        if isinstance(toty, datatype.DType):
            toty = TileTy(toty, TupleTy([]))
        if fromty.shape != toty.shape:
            value, fromty = _broadcast_shape(self, value, fromty, toty)
        if fromty.dtype != toty.dtype:
            value = bc.encode_BitcastOp(self.builder, typeid(self.type_table, toty), value)
        return value

    def constant(self, value: int | float, ty: Type) -> bc.Value:
        if isinstance(ty, TileTy):
            dtype = ty.dtype
        elif isinstance(ty, datatype.DType):
            dtype = ty
        else:
            raise TypeError(f"Cannot make a constant tuple out of {ty}")

        data = _constant_to_bytes(value, dtype)
        return bc.encode_ConstantOp(self.builder, typeid(self.type_table, ty), data)

    def constant_tuple(self, value, ty: Type) -> Tuple[bc.Value, ...]:
        if isinstance(ty, TupleTy):
            return sum((self.constant_tuple(item_val, item_ty)
                        for item_ty, item_val in zip(ty.value_types, value, strict=True)), ())
        return self.constant(value, ty),

    def undefined_value(self, ty: Type) -> bc.Value:
        if isinstance(ty, TokenTy):
            return bc.encode_MakeTokenOp(self.builder, typeid(self.type_table, ty))

        if isinstance(ty, TileTy) and isinstance(ty.dtype, PointerTy):
            const = self.constant(0, TileTy(dtype=datatype.int64, shape=ty.shape))
            return bc.encode_IntToPtrOp(self.builder, typeid(self.type_table, ty), const)

        return self.constant(0, ty)

    def make_partition_view(self,
                            array: Var,
                            order: Sequence[int],
                            tile_shape: Sequence[int],
                            padding_mode: PaddingMode) -> bc.Value:
        padding_value = padding_mode_to_bytecode[padding_mode]
        array_ty = self.typeof(array)
        assert isinstance(array_ty, ArrayTy)
        view_ty_id = tensor_view_typeid(self.type_table, array_ty)
        partition_ty_id = self.type_table.partition_view(
                tile_shape, view_ty_id, order, padding_value)
        view = self.get_value(array)
        return bc.encode_MakePartitionViewOp(self.builder, partition_ty_id, view)


def generate_bytecode_for_block(ctx: BytecodeContext, block: Block):
    for op in block.operations:
        with ctx.loc(op.loc):
            try:
                result_values = op.generate_bytecode(ctx)
                if isinstance(result_values, bc.Value):
                    result_values = (result_values,)

                for result_var, val in zip(op.result_vars, result_values, strict=True):
                    assert isinstance(val, bc.Value)
                    ctx.set_value(result_var, val)
            except TileError:
                raise
            except Exception as e:
                raise TileInternalError(f"Internal error: {e}") from e


def generate_bytecode_for_kernel(root_block: Block,
                                 compiler_options: CompilerOptions,
                                 sm_arch: str,
                                 writer: bc.BytecodeWriter,
                                 anonymize_debug_attr: bool):
    target_options = compiler_options.specialize_for_target(sm_arch)
    entry_hints = bc.EntryHints(num_cta_in_cga=target_options.num_ctas,
                                occupancy=target_options.occupancy)

    param_type_ids = [typeid(writer.type_table, p.get_type()) for p in root_block.params]
    debug_attr_map = DebugAttrMap(writer.debug_attr_table, root_block.name, anonymize_debug_attr)
    func_debug_attr = debug_attr_map.get_debugattr(root_block.loc)

    with writer.function(name=root_block.name,
                         parameter_types=param_type_ids,
                         result_types=(),
                         entry_point=True,
                         hints={sm_arch: entry_hints},
                         debug_attr=func_debug_attr) as (builder, param_values):
        ctx = BytecodeContext(builder=builder,
                              type_table=writer.type_table,
                              debug_attr_map=debug_attr_map,
                              global_section=writer.global_section,
                              ir_ctx=root_block.ctx,
                              sm_arch=sm_arch)

        for var, value in zip(root_block.params, param_values, strict=True):
            ctx.set_value(var, value)

        generate_bytecode_for_block(ctx, root_block)
