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

#include <numeric>

#include <legate.h>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <legate_dataframe/filling.hpp>

using namespace legate::dataframe;

TEST(FillingTest, int64)
{
  for (int i = -100; i < 100; i += 10) {
    std::vector<int64_t> data(10);
    std::iota(data.begin(), data.end(), i);

    cudf::test::fixed_width_column_wrapper<int64_t> expect(data.begin(), data.end());
    LogicalColumn res = sequence(data.size(), i);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(res.get_cudf()->view(), expect);
  }
}
