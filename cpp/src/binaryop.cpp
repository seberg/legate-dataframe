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

#include <legate.h>

#include <cudf/binaryop.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <legate_dataframe/binaryop.hpp>
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>

namespace legate::dataframe::task {

class BinaryOpColColTask : public Task<BinaryOpColColTask, OpCode::BinaryOpColCol> {
 public:
  static void gpu_variant(legate::TaskContext context)
  {
    GPUTaskContext ctx{context};
    auto op                           = argument::get_next_scalar<cudf::binary_operator>(ctx);
    const auto lhs                    = argument::get_next_input<PhysicalColumn>(ctx);
    const auto rhs                    = argument::get_next_input<PhysicalColumn>(ctx);
    auto output                       = argument::get_next_output<PhysicalColumn>(ctx);
    std::unique_ptr<cudf::column> ret = cudf::binary_operation(
      lhs.column_view(), rhs.column_view(), op, output.cudf_type(), ctx.stream(), ctx.mr());
    output.move_into(std::move(ret));
  }
};

class BinaryOpColScalarTask : public Task<BinaryOpColScalarTask, OpCode::BinaryOpColScalar> {
 public:
  static void gpu_variant(legate::TaskContext context)
  {
    GPUTaskContext ctx{context};
    auto op        = argument::get_next_scalar<cudf::binary_operator>(ctx);
    const auto lhs = argument::get_next_input<PhysicalColumn>(ctx);
    const auto rhs = argument::get_next_scalar<ScalarArg>(ctx);
    auto output    = argument::get_next_output<PhysicalColumn>(ctx);
    std::unique_ptr<cudf::column> ret =
      cudf::binary_operation(lhs.column_view(),
                             *rhs.get_cudf(ctx.stream(), ctx.mr()),
                             op,
                             output.cudf_type(),
                             ctx.stream(),
                             ctx.mr());
    output.move_into(std::move(ret));
  }
};

class BinaryOpScalarColTask : public Task<BinaryOpScalarColTask, OpCode::BinaryOpScalarCol> {
 public:
  static void gpu_variant(legate::TaskContext context)
  {
    GPUTaskContext ctx{context};
    auto op        = argument::get_next_scalar<cudf::binary_operator>(ctx);
    const auto lhs = argument::get_next_scalar<ScalarArg>(ctx);
    const auto rhs = argument::get_next_input<PhysicalColumn>(ctx);
    auto output    = argument::get_next_output<PhysicalColumn>(ctx);
    std::unique_ptr<cudf::column> ret =
      cudf::binary_operation(*lhs.get_cudf(ctx.stream(), ctx.mr()),
                             rhs.column_view(),
                             op,
                             output.cudf_type(),
                             ctx.stream(),
                             ctx.mr());
    output.move_into(std::move(ret));
  }
};

}  // namespace legate::dataframe::task

namespace {

void __attribute__((constructor)) register_tasks()
{
  legate::dataframe::task::BinaryOpColColTask::register_variants();
  legate::dataframe::task::BinaryOpColScalarTask::register_variants();
  legate::dataframe::task::BinaryOpScalarColTask::register_variants();
}

}  // namespace

namespace legate::dataframe {

LogicalColumn binary_operation(const LogicalColumn& lhs,
                               const LogicalColumn& rhs,
                               cudf::binary_operator op,
                               cudf::data_type output_type)
{
  auto runtime          = legate::Runtime::get_runtime();
  bool nullable         = lhs.nullable() || rhs.nullable();
  auto ret              = LogicalColumn::empty_like(std::move(output_type), nullable);
  legate::AutoTask task = runtime->create_task(get_library(), task::BinaryOpColColTask::TASK_ID);
  argument::add_next_scalar(task, static_cast<std::underlying_type_t<cudf::binary_operator>>(op));
  argument::add_next_input(task, lhs);
  argument::add_next_input(task, rhs);
  argument::add_next_output(task, ret);
  runtime->submit(std::move(task));
  return ret;
}

LogicalColumn binary_operation(const LogicalColumn& lhs,
                               const ScalarArg& rhs,
                               cudf::binary_operator op,
                               cudf::data_type output_type)
{
  auto runtime          = legate::Runtime::get_runtime();
  bool nullable         = lhs.nullable() || rhs.is_null();
  auto ret              = LogicalColumn::empty_like(std::move(output_type), nullable);
  legate::AutoTask task = runtime->create_task(get_library(), task::BinaryOpColScalarTask::TASK_ID);
  argument::add_next_scalar(task, static_cast<std::underlying_type_t<cudf::binary_operator>>(op));
  argument::add_next_input(task, lhs);
  argument::add_next_scalar(task, rhs);
  argument::add_next_output(task, ret);
  runtime->submit(std::move(task));
  return ret;
}

LogicalColumn binary_operation(const ScalarArg& lhs,
                               const LogicalColumn& rhs,
                               cudf::binary_operator op,
                               cudf::data_type output_type)
{
  auto runtime          = legate::Runtime::get_runtime();
  bool nullable         = lhs.is_null() || rhs.nullable();
  auto ret              = LogicalColumn::empty_like(std::move(output_type), nullable);
  legate::AutoTask task = runtime->create_task(get_library(), task::BinaryOpScalarColTask::TASK_ID);
  argument::add_next_scalar(task, static_cast<std::underlying_type_t<cudf::binary_operator>>(op));
  argument::add_next_scalar(task, lhs);
  argument::add_next_input(task, rhs);
  argument::add_next_output(task, ret);
  runtime->submit(std::move(task));
  return ret;
}

}  // namespace legate::dataframe
