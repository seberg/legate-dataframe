/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <legate.h>

#include <cudf/binaryop.hpp>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/scalar.hpp>

namespace legate::dataframe {

/**
 * @brief Performs a binary operation between two columns.
 *
 * The output contains the result of `op(lhs[i], rhs[i])` for all `0 <= i < lhs.size()`
 *
 * Regardless of the operator, the validity of the output value is the logical
 * AND of the validity of the two operands except NullMin and NullMax (logical OR).
 *
 * @param lhs         The left operand column
 * @param rhs         The right operand column
 * @param op          The binary operator
 * @param output_type The desired data type of the output column
 * @return            Output column of `output_type` type containing the result of
 *                    the binary operation
 * @throw cudf::logic_error if @p lhs and @p rhs are different sizes
 * @throw cudf::logic_error if @p output_type dtype isn't boolean for comparison and logical
 * operations.
 * @throw cudf::logic_error if @p output_type dtype isn't fixed-width
 * @throw cudf::data_type_error if the operation is not supported for the types of @p lhs and @p rhs
 */
LogicalColumn binary_operation(const LogicalColumn& lhs,
                               const LogicalColumn& rhs,
                               cudf::binary_operator op,
                               cudf::data_type output_type);

/**
 * @brief Performs a binary operation between a column and a scalar.
 *
 * The output contains the result of `op(lhs[i], rhs)` for all `0 <= i < lhs.size()`
 * The column elements are the left operand and the scalar is the right operand.
 * This distinction is significant in case of non-commutative binary operations
 *
 * Regardless of the operator, the validity of the output value is the logical
 * AND of the validity of the two operands except NullMin and NullMax (logical OR).
 *
 * @param lhs         The left operand column
 * @param rhs         The right operand scalar
 * @param op          The binary operator
 * @param output_type The desired data type of the output column
 * @return            Output column of `output_type` type containing the result of
 *                    the binary operation
 * @throw cudf::logic_error if @p output_type dtype isn't fixed-width
 * @throw cudf::logic_error if @p output_type dtype isn't boolean for comparison and logical
 * operations.
 * @throw cudf::data_type_error if the operation is not supported for the types of @p lhs and @p rhs
 */
LogicalColumn binary_operation(const LogicalColumn& lhs,
                               const ScalarArg& rhs,
                               cudf::binary_operator op,
                               cudf::data_type output_type);

/**
 * @brief Performs a binary operation between a scalar and a column.
 *
 * The output contains the result of `op(lhs, rhs[i])` for all `0 <= i < rhs.size()`
 * The scalar is the left operand and the column elements are the right operand.
 * This distinction is significant in case of non-commutative binary operations
 *
 * Regardless of the operator, the validity of the output value is the logical
 * AND of the validity of the two operands except NullMin and NullMax (logical OR).
 *
 * @param lhs         The left operand scalar
 * @param rhs         The right operand column
 * @param op          The binary operator
 * @param output_type The desired data type of the output column
 * @return            Output column of `output_type` type containing the result of
 *                    the binary operation
 * @throw cudf::logic_error if @p output_type dtype isn't fixed-width
 * @throw cudf::logic_error if @p output_type dtype isn't boolean for comparison and logical
 * operations.
 * @throw cudf::data_type_error if the operation is not supported for the types of @p lhs and @p rhs
 */
LogicalColumn binary_operation(const ScalarArg& lhs,
                               const LogicalColumn& rhs,
                               cudf::binary_operator op,
                               cudf::data_type output_type);

}  // namespace legate::dataframe
