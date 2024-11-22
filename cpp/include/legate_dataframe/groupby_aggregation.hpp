/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <string>
#include <vector>

#include <cudf/aggregation.hpp>

#include <legate_dataframe/core/table.hpp>

namespace legate::dataframe {

/**
 * @brief Help function to make `cudf::groupby_aggregation` object from a kind.
 *
 * Notice, only aggregations with a default constructor is supported.
 *
 * @param kind Aggregation kind
 * @return Groupby aggregation corresponding to `kind`
 */
std::unique_ptr<cudf::groupby_aggregation> make_groupby_aggregation(cudf::aggregation::Kind kind);

/**
 * @brief Help function to make `cudf::groupby_aggregation` objects from kinds.
 *
 * Notice, only aggregations with a default constructor are supported.
 *
 * @param kinds Aggregation kinds
 * @return Vector of groupby aggregations corresponding to each kind in `kinds`
 */
std::vector<std::unique_ptr<cudf::groupby_aggregation>> make_groupby_aggregations(
  const std::vector<cudf::aggregation::Kind>& kinds);

/**
 * @brief Perform a groupby and aggregation in a single operation.
 *
 * WARN: non-default cudf::aggregation arguments are ignored. The default constructor is used
 *       always. This also means that we only support aggregations that have a default constructor!
 *
 * @param table The table to group and aggregate.
 * @param keys The names of the columns whose rows act as the groupby keys.
 * @param column_aggregations A vector of column aggregations to perform. Each column aggregation
 * produces a column in the output table by performing an aggregation-kind on a column in `table`.
 * It consist of a tuple: `(<input-column-name>, <aggregation-kind>, <output-column-name>)`. E.g.
 * `("x", SUM, "sum-of-x")}` will produce a column named "sum-of-x" in the output table, which, for
 * each groupby key, has a row that contains the sum of the values in the column "x". Multiple
 * column aggregations can share the same input column but all output columns must be unique and not
 * conflict with the name of the key columns.
 * @return A new logical table that contains the key columns and the aggregated columns using the
 * output column names and order specified in `column_aggregations`.
 */
LogicalTable groupby_aggregation(
  const LogicalTable& table,
  const std::vector<std::string>& keys,
  const std::vector<std::tuple<std::string, cudf::aggregation::Kind, std::string>>&
    column_aggregations);

}  // namespace legate::dataframe
