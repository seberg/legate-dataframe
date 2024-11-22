/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <legate.h>

#include <cudf/strings/convert/convert_datetime.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/timestamps.hpp>

using namespace legate::dataframe;

template <typename T>
struct TimestampsTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(TimestampsTest, cudf::test::TimestampTypes);

TYPED_TEST(TimestampsTest, ToTimestamps)
{
  cudf::test::strings_column_wrapper strings(
    {"2010-06-19T13:55", "2011-06-19T13:55", "", "2010-07-19T13:55"}, {1, 1, 0, 0});
  cudf::data_type timestamp_type{cudf::type_to_id<TypeParam>()};
  std::string format{"%Y-%m-%dT%H:%M:%SZ"};

  auto expect =
    cudf::strings::to_timestamps(cudf::strings_column_view(strings), timestamp_type, format);

  LogicalColumn input(strings);
  auto result = to_timestamps(input, timestamp_type, format);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.get_cudf()->view(), expect->view());
}
