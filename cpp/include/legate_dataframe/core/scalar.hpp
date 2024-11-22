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

#include <optional>
#include <string>

#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <legate.h>

#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/utils.hpp>

namespace legate::dataframe {

/**
 * @brief Scalar task argument
 *
 * Use this to pass scalar arguments to tasks using `add_next_scalar` and
 * get_next_scalar. It extend `legate::Scalar` by support cudf data types such
 * as TIMESTAMP_DAYS.
 *
 * Contrary to Columns and Tables, which have both a Logical and Physical part,
 * a ScalarArg can be accessed both from the user code and from within tasks.
 */
class ScalarArg {
 public:
  /**
   * @brief Create a NULL scalar with EMPTY cudf_dtype
   *
   * This ctor is mainly here because of Cython.
   */
  ScalarArg() = default;

  /**
   * @brief Create a ScalarArg from a legate scalar
   *
   * @param scalar A legate scalar
   * @param cudf_type The cudf data type of the column. If `EMPTY` (default), the cudf data type is
   * derived from the data type of `scalar`.
   */
  ScalarArg(legate::Scalar scalar,
            cudf::data_type cudf_type = cudf::data_type{cudf::type_id::EMPTY})
    : scalar_{std::move(scalar)}
  {
    if (cudf_type.id() == cudf::type_id::EMPTY) {
      cudf_type_ = cudf::data_type{to_cudf_type_id(scalar.type().code())};
    } else {
      cudf_type_ = std::move(cudf_type);
    }
  }

  /**
   * @brief Create a ScalarArg from a local cudf scalar
   *
   * @param cudf_scalar The local cuDF scalar to copy
   * @param stream CUDA stream used for device memory operations
   */
  ScalarArg(const cudf::scalar& cudf_scalar,
            rmm::cuda_stream_view stream = cudf::get_default_stream());

 public:
  ScalarArg(const ScalarArg& other)            = default;
  ScalarArg& operator=(const ScalarArg& other) = default;
  ScalarArg(ScalarArg&& other)                 = default;
  ScalarArg& operator=(ScalarArg&& other)      = default;

 public:
  /**
   * @brief Check if the scalar is NULL (NA)
   *
   * If the scalar is NULL, the legate value is not set.
   *
   * @return true if a NULL value else false.
   */
  bool is_null() const { return scalar_.type().code() == legate::Type::Code::NIL; }

  /**
   * @brief Return the legate representation of the scalar
   *
   * This function raises if the scalar is NULL.
   *
   * @return Legate scalar.
   */
  legate::Scalar get_legate_scalar() const { return scalar_; }

  /**
   * @brief Copy the scalar into a local cudf scalar
   *
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for all device memory allocations.
   * @return cudf scalar, which owns the data
   */

  std::unique_ptr<cudf::scalar> get_cudf(rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr) const;

  /**
   * @brief Get the data type of the underlying scalar
   *
   * @return The legate data type
   */
  legate::Type type() const { return scalar_.type(); }

  /**
   * @brief Get the cudf data type of the scalar
   *
   * @return The cudf data type
   */
  cudf::data_type cudf_type() const { return cudf_type_; }

 private:
  // In order to support a default ctor (used by Cython),
  // we make the legate scalar optional.
  legate::Scalar scalar_;
  cudf::data_type cudf_type_{cudf::type_id::EMPTY};
};

namespace argument {

/**
 * @brief Add a scalar to the next input task argument
 *
 * This should match a call to `get_next_scalar<ScalarArg>()` by a legate task.
 *
 * NB: the order of "add_next_*" calls must match the order of the
 * corresponding "get_next_*" calls.
 *
 * @param task The legate task to add the argument.
 * @param scalar The scalar to add as the next task argument.
 */
void add_next_scalar(legate::AutoTask& task, const ScalarArg& scalar);

template <>
inline ScalarArg get_next_scalar(GPUTaskContext& ctx)
{
  cudf::type_id cudf_type_id = static_cast<cudf::type_id>(
    ctx.get_next_scalar_arg().value<std::underlying_type_t<cudf::type_id>>());
  legate::Scalar value = ctx.get_next_scalar_arg();
  return ScalarArg(std::move(value), cudf::data_type{cudf_type_id});
}

}  // namespace argument

}  // namespace legate::dataframe
