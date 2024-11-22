# Copyright (c) 2023-2024: int NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from numpy.typing import DTypeLike

from legate_dataframe.lib.core.column import LogicalColumn
from legate_dataframe.lib.core.scalar import ScalarLike

class binary_operator(Enum):
    ADD: int  # operator +
    SUB: int  # operator -
    MUL: int  # operator *
    DIV: int  # operator / using common type of lhs and rhs
    TRUE_DIV: int  # operator / after promoting type to floating point
    BITWISE_AND: int  # operator &
    BITWISE_OR: int  # operator |
    BITWISE_XOR: int  # operator ^
    LOGICAL_AND: int  # operator &&
    LOGICAL_OR: int  # operator ||
    EQUAL: int  # operator ==
    NOT_EQUAL: int  # operator !=
    LESS: int  # operator <
    GREATER: int  # operator >
    LESS_EQUAL: int  # operator <=
    GREATER_EQUAL: int  # operator >=
    MOD: int  # operator %

    # operator //
    # integer division rounding towards negative
    # infinity if both arguments are integral;
    # floor division for floating types (using C++ type
    # promotion for mixed integral/floating arguments)
    # If different promotion semantics are required, it
    # is the responsibility of the caller to promote
    # manually before calling in to this function.
    FLOOR_DIV: int
    # positive modulo operator
    # If remainder is negative, this returns
    # (remainder + divisor) % divisor else, it returns
    # (dividend % divisor)
    PMOD: int
    # operator % but following Python's sign rules for negatives
    PYMOD: int
    # lhs ^ rhs
    POW: int
    # int ^ int, used to avoid floating point precision loss.
    # Returns 0 for negative exponents.
    INT_POW: int
    # logarithm to the base
    LOG_BASE: int
    # 2-argument arctangent
    ATAN2: int
    # operator <<
    SHIFT_LEFT: int
    # operator >>
    SHIFT_RIGHT: int
    # operator >>> (from Java)
    # Logical right shift. Casts to an unsigned value before shifting.
    SHIFT_RIGHT_UNSIGNED: int
    # Returns true when both operands are null; false when one is null; the
    # result of equality when both are non-null
    NULL_EQUALS: int
    # Returns max of operands when both are non-null; returns the non-null
    # operand when one is null; or invalid when both are null
    NULL_MAX: int
    # Returns min of operands when both are non-null; returns the non-null
    # operand when one is null; or invalid when both are null
    NULL_MIN: int
    # generic binary operator to be generated with input
    # ptx code
    GENERIC_BINARY: int
    # operator && with Spark rules: (null, null) is null, (null, true) is null,
    # (null, false) is false, and (valid, valid) == LOGICAL_AND(valid, valid)
    NULL_LOGICAL_AND: int
    # operator || with Spark rules: (null, null) is null, (null, true) is true,
    # (null, false) is null, and (valid, valid) == LOGICAL_OR(valid, valid)
    NULL_LOGICAL_OR: int
    # invalid operation
    INVALID_BINARY: int

def binary_operation(
    lhs: LogicalColumn | ScalarLike,
    rhs: LogicalColumn | ScalarLike,
    op: binary_operator,
    output_type: DTypeLike,
) -> LogicalColumn: ...
