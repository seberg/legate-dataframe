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

#include <cudf/types.hpp>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/table.hpp>

namespace legate::dataframe {

/**
 * @brief Write table to csv files.
 *
 * Each partition will be written to a separate file.
 *
 * Files will be created in the specified output directory using the convention ``part.0.csv``,
 * ``part.1.csv``, ``part.2.csv``, ... and so on for each partition in the table:
 *
 *      /path/to/output/
 *          ├── part.0.csv
 *          ├── part.1.csv
 *          ├── part.2.csv
 *          └── ...
 *
 * @param tbl The table to write.
 * @param path Destination directory for data.
 * @param delimiter The field delimiter.
 */
void csv_write(LogicalTable& tbl, const std::string& dirpath, char delimiter = ',');

/**
 * @brief Read csv files into a LogicalTable
 *
 * Files are currently read into N partitions where N is the number of GPU workers used.
 * The partitions are split by row, meaning that each reads approximately the
 * same number of rows (possibly over multiple files).
 * If the number of rows does not split evenly, the first partitions will
 * contain one additional row.
 *
 * Note that file order is currently glob/string sorted.
 *
 * TODO: This function does currently not support passing csv reader
 *       parameters.
 *
 * @param glob_string The glob string to specify the csv files. All glob matches must be valid
 * csv files and have the same layout. See <https://linux.die.net/man/7/glob>.
 * @param dtypes The cudf type for each column (must match usecols).
 * @param na_filter Whether to detect missing values, set to false to improve performance.
 * @param delimiter The field delimiter.
 * @param usecols The column names to read from the file, if not passed reads all columns.
 * @return The read LogicalTable.
 */
LogicalTable csv_read(const std::string& glob_string,
                      const std::vector<cudf::data_type>& dtypes,
                      bool na_filter                                         = true,
                      char delimiter                                         = ',',
                      const std::optional<std::vector<std::string>>& usecols = std::nullopt);

}  // namespace legate::dataframe
