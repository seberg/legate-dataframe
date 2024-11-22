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

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/device_uvector.hpp>

#include <legate_dataframe/core/scalar.hpp>
#include <legate_dataframe/utils.hpp>

namespace legate::dataframe {

namespace {

struct scalar_from_cudf {
  template <typename T, std::enable_if_t<cudf::is_rep_layout_compatible<T>()>* = nullptr>
  legate::Scalar operator()(const cudf::scalar& cudf_scalar, rmm::cuda_stream_view stream)
  {
    if (!cudf_scalar.is_valid()) {
      throw std::invalid_argument("Cannot convert NULL cudf scalar to legate scalar.");
    }
    using ScalarType                = cudf::scalar_type_t<T>;
    auto p_scalar                   = static_cast<ScalarType const*>(&cudf_scalar);
    T scalar_value                  = p_scalar->value(stream);
    const legate::Type legate_dtype = to_legate_type(cudf_scalar.type().id());
    return legate::Scalar(legate_dtype, &scalar_value, true);
  }

  template <typename T, std::enable_if_t<!cudf::is_rep_layout_compatible<T>()>* = nullptr>
  legate::Scalar operator()(const cudf::scalar& cudf_scalar, rmm::cuda_stream_view stream)
  {
    // TODO: support strings and lists, and maybe more.
    throw std::invalid_argument("ScalarArg: cudf type not (yet) supported.");
  }
};

}  // namespace

ScalarArg::ScalarArg(const cudf::scalar& cudf_scalar, rmm::cuda_stream_view stream)
{
  this->cudf_type_ = cudf_scalar.type();
  if (!cudf_scalar.is_valid()) { return; }
  this->scalar_ =
    cudf::type_dispatcher(cudf_scalar.type(), scalar_from_cudf{}, cudf_scalar, stream);
}

namespace {

struct scalar_to_cudf {
  template <typename T, std::enable_if_t<cudf::is_rep_layout_compatible<T>()>* = nullptr>
  std::unique_ptr<cudf::scalar> operator()(const ScalarArg* scalar,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
  {
    if (scalar->is_null()) {
      // A scalar is by default invalid, which we interpret as null.
      return cudf::make_default_constructed_scalar(scalar->cudf_type(), stream, mr);
    }
    return std::make_unique<cudf::scalar_type_t<T>>(
      scalar->get_legate_scalar().value<T>(), true, stream, mr);
  }

  template <typename T, std::enable_if_t<!cudf::is_rep_layout_compatible<T>()>* = nullptr>
  std::unique_ptr<cudf::scalar> operator()(const ScalarArg* scalar,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
  {
    // TODO: support strings and lists, and maybe more.
    throw std::invalid_argument("Can't convert scalar to cudf as of now.");
  }
};

}  // namespace

std::unique_ptr<cudf::scalar> ScalarArg::get_cudf(rmm::cuda_stream_view stream,
                                                  rmm::mr::device_memory_resource* mr) const
{
  return cudf::type_dispatcher(cudf_type_, scalar_to_cudf{}, this, stream, mr);
}

namespace argument {

/*
 * Overloads to allow getting and setting our Scalars for legate tasks.
 */
void add_next_scalar(legate::AutoTask& task, const ScalarArg& scalar)
{
  add_next_scalar(task,
                  static_cast<std::underlying_type_t<cudf::type_id>>(scalar.cudf_type().id()));
  task.add_scalar_arg(scalar.get_legate_scalar());
}

}  // namespace argument

}  // namespace legate::dataframe
