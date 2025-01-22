/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include <string>

#include <cudf/datetime.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/timestamps.hpp>

namespace legate::dataframe {
namespace task {

class ToTimestampsTask : public Task<ToTimestampsTask, OpCode::ToTimestamps> {
 public:
  static void gpu_variant(legate::TaskContext context)
  {
    GPUTaskContext ctx{context};
    const auto format = argument::get_next_scalar<std::string>(ctx);
    const auto input  = argument::get_next_input<PhysicalColumn>(ctx);
    auto output       = argument::get_next_output<PhysicalColumn>(ctx);

    std::unique_ptr<cudf::column> ret = cudf::strings::to_timestamps(
      input.column_view(), output.cudf_type(), format, ctx.stream(), ctx.mr());
    output.move_into(std::move(ret));
  }
};

class ExtractTimestampComponentTask
  : public Task<ExtractTimestampComponentTask, OpCode::ExtractTimestampComponent> {
 public:
  static void gpu_variant(legate::TaskContext context)
  {
    GPUTaskContext ctx{context};
    const auto component = argument::get_next_scalar<cudf::datetime::datetime_component>(ctx);
    const auto input     = argument::get_next_input<PhysicalColumn>(ctx);
    auto output          = argument::get_next_output<PhysicalColumn>(ctx);

    std::unique_ptr<cudf::column> ret;
    ret = cudf::datetime::extract_datetime_component(
      input.column_view(), component, ctx.stream(), ctx.mr());

    output.move_into(std::move(ret));
  }
};

}  // namespace task

LogicalColumn to_timestamps(const LogicalColumn& input,
                            cudf::data_type timestamp_type,
                            std::string format)
{
  auto runtime          = legate::Runtime::get_runtime();
  auto ret              = LogicalColumn::empty_like(timestamp_type, input.nullable());
  legate::AutoTask task = runtime->create_task(get_library(), task::ToTimestampsTask::TASK_ID);
  argument::add_next_scalar(task, std::move(format));
  argument::add_next_input(task, input);
  argument::add_next_output(task, ret);
  runtime->submit(std::move(task));
  return ret;
}

LogicalColumn extract_timestamp_component(const LogicalColumn& input,
                                          cudf::datetime::datetime_component component)
{
  if (!cudf::is_timestamp(input.cudf_type())) {
    throw std::invalid_argument("extract_timestamp_component() input must be timestamp");
  }
  auto runtime = legate::Runtime::get_runtime();
  auto ret     = LogicalColumn::empty_like(cudf::data_type{cudf::type_id::INT16}, input.nullable());
  legate::AutoTask task =
    runtime->create_task(get_library(), task::ExtractTimestampComponentTask::TASK_ID);
  argument::add_next_scalar(
    task, static_cast<std::underlying_type_t<cudf::datetime::datetime_component>>(component));
  argument::add_next_input(task, input);
  argument::add_next_output(task, ret);
  runtime->submit(std::move(task));
  return ret;
}

}  // namespace legate::dataframe

namespace {

void __attribute__((constructor)) register_tasks()
{
  legate::dataframe::task::ToTimestampsTask::register_variants();
  legate::dataframe::task::ExtractTimestampComponentTask::register_variants();
}

}  // namespace
