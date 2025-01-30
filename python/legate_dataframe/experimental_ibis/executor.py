# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from collections import defaultdict
from functools import singledispatch
from typing import List, Tuple

import cudf  # for scalar
import ibis.expr.operations as ops
from ibis.common.exceptions import OperationNotDefinedError, UnboundExpressionError

import legate_dataframe.lib.join as ldf_join
from legate_dataframe import LogicalTable
from legate_dataframe.experimental_ibis.schema import to_plc_type
from legate_dataframe.lib.binaryop import binary_operation, binary_operator
from legate_dataframe.lib.groupby_aggregation import (
    AggregationKind,
    groupby_aggregation,
)
from legate_dataframe.lib.stream_compaction import apply_boolean_mask
from legate_dataframe.lib.unaryop import unary_operation, unary_operator


@singledispatch
def execute(expr, **_):
    raise NotImplementedError(
        f"Operation {expr!r} is not implemented for the legate backend"
    )


@execute.register
def operation(op: ops.Node, **_):
    raise OperationNotDefinedError(f"No translation rule for {type(op)}")


@execute.register
def literal(op: ops.Literal, **_):
    # TODO(seberg): See if a pylibcudf scalar is easy to create, then
    # allow legate-dataframe ops to digest them directly (or only).
    from cudf._lib.types import PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES

    dtype = PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[to_plc_type(op.dtype).id()]
    return cudf.Scalar(op.value, dtype)


@execute.register
def column(op: ops.Field, **kw):
    table = execute(op.rel, **kw)
    return table[op.name]


@execute.register
def project(op: ops.Project, **kw):
    # Note: Will fail for scalars (and unnesting), which is fine, though.
    values = op.values.values()
    names = op.values.keys()

    columns = [execute(col, **kw) for col in values]
    return LogicalTable(columns, names)


@execute.register
def filter(op: ops.Filter, **kw):
    tbl = execute(op.parent, **kw)
    if not op.predicates:
        return tbl

    def _refine_mask(mask, other):
        other = execute(other, **kw)
        return binary_operation(mask, other, binary_operator.LOGICAL_AND, mask.dtype())

    mask = execute(op.predicates[0], **kw)
    mask = functools.reduce(_refine_mask, op.predicates[1:], mask)

    return apply_boolean_mask(tbl, mask)


_unaryops = {
    ops.Abs: unary_operator.ABS,
    ops.Acos: unary_operator.ARCCOS,
    ops.Asin: unary_operator.ARCSINH,
    ops.Atan: unary_operator.ARCTAN,
    ops.Ceil: unary_operator.CEIL,
    ops.Cos: unary_operator.COS,
    # ops.Cot:
    # ops.DayOfWeekIndex:
    ops.Exp: unary_operator.EXP,
    ops.Floor: unary_operator.FLOOR,
    # ops.IsInf:
    # ops.IsNan:
    # ops.IsNull:
    ops.Ln: unary_operator.LOG,
    # ops.Log10:
    # ops.Log2:
    # ops.Negate:
    ops.Not: unary_operator.NOT,
    # ops.NotNull:
    ops.Sin: unary_operator.SIN,
    ops.Sqrt: unary_operator.SQRT,
    ops.Tan: unary_operator.TAN,
    ops.BitwiseNot: unary_operator.BIT_INVERT,
}


@execute.register
def unaryop(op: ops.Unary | ops.Comparison, **_):
    # automatically pick the correct kernel based on the operand types
    typ = type(op)

    unaryop = _unaryops.get(typ)
    if unaryop is not None:
        col = execute(op.arg, **_)
        return unary_operation(col, unaryop)

    raise OperationNotDefinedError(f"Operation {typ} not yet defined.")


_binops = {
    # math ops:
    ops.Atan2: binary_operator.ATAN2,
    ops.Add: binary_operator.ADD,
    ops.And: binary_operator.LOGICAL_AND,
    ops.Divide: binary_operator.TRUE_DIV,
    ops.FloorDivide: binary_operator.DIV,
    # ops.Modulus: (which rules to follow?)
    ops.Multiply: binary_operator.MUL,
    ops.Subtract: binary_operator.SUB,
    # logical/bitwise
    ops.And: binary_operator.LOGICAL_AND,
    ops.Or: binary_operator.LOGICAL_OR,
    # ops.Xor:
    ops.BitwiseOr: binary_operator.BITWISE_OR,
    ops.BitwiseXor: binary_operator.BITWISE_XOR,
    ops.BitwiseAnd: binary_operator.BITWISE_AND,
    # Comparisons:
    ops.Equals: binary_operator.EQUAL,
    ops.Less: binary_operator.LESS,
    ops.LessEqual: binary_operator.LESS_EQUAL,
    ops.Greater: binary_operator.GREATER,
    ops.GreaterEqual: binary_operator.GREATER_EQUAL,
    ops.NotEquals: binary_operator.NOT_EQUAL,
    ops.IdenticalTo: binary_operator.NULL_EQUALS,
    # Times (mostly probably identical as above):
    # ops.DateAdd: operator.add,
    # ops.DateSub: operator.sub,
    # ops.DateDiff: operator.sub,
    # ops.TimestampAdd: operator.add,
    # ops.TimestampSub: operator.sub,
    # ops.IntervalSubtract: operator.sub,
}


@execute.register
def binaryop(op: ops.Binary | ops.Comparison, **_):
    # automatically pick the correct kernel based on the operand types
    typ = type(op)
    dtype = to_plc_type(op.dtype)

    binop = _binops.get(typ)
    if binop is not None:
        lhs = execute(op.left, **_)
        rhs = execute(op.right, **_)
        return binary_operation(lhs, rhs, binop, dtype)

    raise OperationNotDefinedError(f"Operation {typ} not yet defined.")


reductions: dict[type, AggregationKind] = {
    ops.Min: AggregationKind.MIN,
    ops.Max: AggregationKind.MAX,
    ops.Sum: AggregationKind.SUM,
    ops.Mean: AggregationKind.MEAN,
    ops.Count: AggregationKind.COUNT_VALID,
    # ops.Mode:
    ops.Any: AggregationKind.ANY,
    ops.All: AggregationKind.ALL,
    ops.Median: AggregationKind.MEDIAN,
    # ops.ApproxMedian:
    # ops.BitAnd:
    # ops.BitOr:
    # ops.BitXor:
    # ops.Arbitrary: arbitrary,
    ops.CountDistinct: AggregationKind.NUNIQUE,
    # ops.ApproxCountDistinct:
}


@execute.register
def reduction(op: ops.Reduction, **_):
    raise OperationNotDefinedError(
        "Reductions are not yet implemented (only groupby-aggs)"
    )


@execute.register
def aggregation(op: ops.Aggregate, **kw):
    unique_cols = {expr: name for name, expr in op.groups.items()}
    keys = list(op.groups.keys())
    aggs: List[Tuple[str, AggregationKind, str]] = []
    reduce_col_counter = 0  # to generate unique names for reduce expressions

    if not op.groups:
        raise NotImplementedError("select is not implemented.")

    for res_name, v in op.metrics.items():
        # TODO: Do we need to allow other kinds of nesting?  Do we need to
        #       reject certain other things (i.e. `order_by`).
        if not isinstance(v, ops.Reduction):
            raise NotImplementedError("Groupby must be an aggregation.")
        if v.where is not None:
            # Note: could probably just as well push the error into execution.
            # Although, it may be plausible to push some where statements down.
            # (A group-by intermediate table could also make sense for this, though...)
            raise NotImplementedError("Where unsupported for groupby aggregations")

        col = v.arg
        if (red := reductions.get(type(v))) is None:
            raise NotImplementedError(f"reduction {type(v)} is not implemented.")

        # Use existing name or generate a new temporary one for the reduction:
        if (name := unique_cols.get(col)) is None:
            while (name := f"reduce_col_{reduce_col_counter}") in unique_cols.values():
                reduce_col_counter += 1

            unique_cols[col] = name

        aggs.append((name, red, res_name))  # type: ignore

    columns = [execute(c, **kw) for c in unique_cols]
    tmp_table = LogicalTable(columns, unique_cols.values())

    return groupby_aggregation(tmp_table, keys, aggs)


_join_types = {
    "left": ldf_join.JoinType.LEFT,
    "inner": ldf_join.JoinType.INNER,
    "outer": ldf_join.JoinType.FULL,
}


def _rename_columns(table: LogicalTable, renames: dict[str, str]):
    """Rename columns from `table` to `renames={old_name: new_name}`.

    Parameters
    ----------
    table : Table to rename
    renames : dict[str, str]
        The new names mapped to the old names `{old_name: new_name}`
    """
    cols = [table.get_column(i) for i in range(table.num_columns())]
    names = [renames.get(n, n) for n in table.get_column_names()]

    return LogicalTable(cols, names)


def _apply_single_join(left_info, link, *, renames, **kw):
    left_table, left_refs = left_info
    right_ref = link.table

    if link.how == "asof":
        raise NotImplementedError("asof joining is not implemented in legate.")

    left_on = []
    right_on = []
    drop_names = []
    # Predicates are e.g. left_ref == right_ref.
    for pred in link.predicates:
        if type(pred) is not ops.Equals:
            raise NotImplementedError(
                "Only equality join predicates are supported in legate."
            )

        # The predicate should refer to columns in left and right.
        if right_ref == pred.right.rel and pred.left.rel in left_refs:
            left, right = pred.left, pred.right
        elif right_ref == pred.left.rel and pred.right.rel in left_refs:
            left, right = pred.right, pred.left
        else:
            raise TypeError(f"Unsupported join predicate {pred}")

        left_name = renames[left.rel].get(left.name, left.name)
        right_name = renames[right.rel].get(right.name, right.name)
        left_on.append(left_name)
        right_on.append(right_name)
        if left_name == right_name:
            drop_names.append(left_on[-1])

    right_table = execute(right_ref.parent, **kw)
    right_table = _rename_columns(right_table, renames[right_ref])

    if link.how == "left":
        join_type = ldf_join.JoinType.LEFT
    elif link.how == "right":
        join_type = ldf_join.JoinType.LEFT
        left_table, right_table = right_table, left_table
        left_on, right_on = right_on, left_on
    elif link.how == "inner":
        join_type = ldf_join.JoinType.INNER
    else:
        # TODO: Need to at least add right join (by swapping and using left)
        raise NotImplementedError(f"join type {link.how} not implemented by legate.")

    rhs_out_columns = right_table.get_column_names()
    for name in drop_names:
        rhs_out_columns.remove(name)

    res = ldf_join.join(
        left_table,
        right_table,
        lhs_keys=left_on,
        rhs_keys=right_on,
        join_type=join_type,
        rhs_out_columns=rhs_out_columns,
    )

    left_refs.add(right_ref)  # Now all columns part of new table
    return res, left_refs


@execute.register
def join(op: ops.JoinChain, **kw):
    """Ibis creates a join chain to allow optimization.  This function is
    designed to evaluate the join left to right (without any optimizations).

    The polars backend splits this join (and column renames) into individual
    nodes.
    """
    # TODO(seberg): this rename stuff is weird, maybe the rewriting solves it?
    renames: dict[ops.Node, dict[str, str]] = defaultdict(dict)
    for out_name, field in op.values.items():
        ref = field.rel
        name = field.name
        renames[ref][name] = out_name

    first_table = execute(op.first.parent, **kw)
    first_table = _rename_columns(first_table, renames[op.first])

    _apply = functools.partial(_apply_single_join, renames=renames, **kw)
    res, _ = functools.reduce(_apply, op.rest, (first_table, {op.first}))
    return res


@execute.register
def table(op: ops.DatabaseTable, **_):
    tables = op.source._tables
    name = op.name
    try:
        return tables[name][0]
    except KeyError:
        raise UnboundExpressionError(
            f"{name} is not a table in the {op.source.name!r} backend, you "
            "probably tried to execute an expression without a data source"
        )


@execute.register(ops.InMemoryTable)
def in_memory_table(op, *, backend, **_):
    return backend._convert_object(op)
