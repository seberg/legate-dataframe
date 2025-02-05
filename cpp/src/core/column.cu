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

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

#include <legate/cuda/cuda.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/device_uvector.hpp>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/null_mask.hpp>
#include <legate_dataframe/core/print.hpp>
#include <legate_dataframe/core/ranges.hpp>
#include <legate_dataframe/utils.hpp>

namespace legate::dataframe {

namespace {

/**
 * @brief Help function to create a LogicalStore from device memory (copying)
 */
[[nodiscard]] legate::LogicalStore create_store(const size_t nelem,
                                                const legate::Type& dtype,
                                                const void* buffer,
                                                cudaStream_t stream)
{
  legate::Runtime* runtime = legate::Runtime::get_runtime();

  if (nelem == 0) { return runtime->create_store({nelem}, dtype); }

  // For now, we copy the device buffer to host memory and hand it over to legate with the
  // `std::free()` function.
  // TODO: avoid this copy: <https://github.com/rapidsai/legate-dataframe/issues/80>
  const size_t nbytes = nelem * dtype.size();
  void* host_buffer   = std::malloc(nbytes);
  if (host_buffer == nullptr) { throw std::bad_alloc(); }
  LEGATE_CHECK_CUDA(cudaMemcpyAsync(host_buffer, buffer, nbytes, cudaMemcpyDeviceToHost, stream));
  LEGATE_CHECK_CUDA(cudaStreamSynchronize(stream));
  auto alloc = legate::ExternalAllocation::create_sysmem(host_buffer, nbytes, std::free);
  auto ret   = runtime->create_store({nelem}, dtype, alloc);
  return ret;
}

[[nodiscard]] legate::LogicalStore logical_store_from_cudf(const cudf::column_view& col,
                                                           cudaStream_t stream)
{
  const legate::Type dtype = to_legate_type(col.type().id());
  const void* src          = static_cast<const uint8_t*>(col.head()) + col.offset() * dtype.size();
  return create_store(col.size(), dtype, src, stream);
}

// String columns need to use `chars_size()` so deal with them explicitly
[[nodiscard]] legate::LogicalStore logical_store_from_cudf(const cudf::strings_column_view& str_col,
                                                           cudaStream_t stream)
{
  if (str_col.offsets().offset() != 0) {
    throw std::runtime_error("string column seems sliced, which is currently not supported.");
  }

  const legate::Type dtype = legate::int8();
  const void* src          = str_col.chars_begin(stream);
  return create_store(str_col.chars_size(stream), dtype, src, stream);
}

legate::LogicalArray from_cudf(const cudf::column_view& col, rmm::cuda_stream_view stream)
{
  if (cudf::type_id::STRING == col.type().id()) {
    cudf::strings_column_view str_col{col};
    if (str_col.chars_size(stream) == 0) {
      return legate::Runtime::get_runtime()->create_string_array(
        legate::Runtime::get_runtime()->create_array({0}, legate::rect_type(1)),
        legate::Runtime::get_runtime()->create_array({0}, legate::int8()));
    }

    // Convert cudf offsets to legate ranges
    cudf::column_view offsets = str_col.offsets();
    rmm::device_uvector<legate::Rect<1>> ranges(offsets.size() - 1, stream);
    cudf_offsets_to_local_ranges(ranges.size(), ranges.data(), offsets, stream);

    if (col.nullable()) {
      // Convert cudf's bit mask to legate's bool mask.
      rmm::device_uvector<bool> bools(ranges.size(), stream);
      null_mask_bits_to_bools(ranges.size(), bools.data(), col.null_mask(), stream);

      return legate::Runtime::get_runtime()->create_string_array(
        legate::LogicalArray(
          create_store(ranges.size(), legate::rect_type(1), ranges.data(), stream),
          create_store(ranges.size(), legate::bool_(), bools.data(), stream)),
        legate::LogicalArray(logical_store_from_cudf(str_col, stream)));
    }
    return legate::Runtime::get_runtime()->create_string_array(
      legate::LogicalArray(
        create_store(ranges.size(), legate::rect_type(1), ranges.data(), stream)),
      legate::LogicalArray(logical_store_from_cudf(str_col, stream)));
  }
  if (col.num_children() > 0) {
    throw std::invalid_argument("non-string column with children isn't supported");
  }
  if (col.nullable()) {
    // Convert cudf's bit mask to legate's bool mask.
    rmm::device_uvector<bool> bools(col.size(), stream);
    null_mask_bits_to_bools(col.size(), bools.data(), col.null_mask(), stream);

    return legate::LogicalArray(logical_store_from_cudf(col, stream),
                                create_store(col.size(), legate::bool_(), bools.data(), stream));
  }
  return legate::LogicalArray(logical_store_from_cudf(col, stream));
}

}  // namespace

LogicalColumn::LogicalColumn(cudf::column_view cudf_col, rmm::cuda_stream_view stream)
  : LogicalColumn{from_cudf(cudf_col, stream), cudf_col.type()}
{
}

namespace {

/**
 * @brief Since Legate's get_physical_array() doesn't support device memory, use this function to
 * copy a physical array to device.
 * TODO: If `get_physical_array()` supports device memory this can be replaced.
 */
[[nodiscard]] rmm::device_buffer copy_physical_array_to_device(const PhysicalArray& physical_array,
                                                               cudaStream_t stream)
{
  auto host_ary_nbytes = physical_array.shape<1>().volume() * physical_array.type().size();
  auto ret             = rmm::device_buffer(host_ary_nbytes, stream);
  LEGATE_CHECK_CUDA(cudaMemcpyAsync(ret.data(),
                                    read_accessor_as_1d_bytes(physical_array.data()),
                                    host_ary_nbytes,
                                    cudaMemcpyHostToDevice,
                                    stream));
  return ret;
}

}  // namespace

std::unique_ptr<cudf::column> LogicalColumn::get_cudf(rmm::cuda_stream_view stream,
                                                      rmm::mr::device_memory_resource* mr) const
{
  if (array_->nested()) {
    if (array_->type().code() == legate::Type::Code::STRING) {
      const legate::StringPhysicalArray a = array_->get_physical_array().as_string_array();
      const legate::PhysicalArray chars   = a.chars();
      const auto num_chars                = chars.data().shape<1>().volume();

      // Copy and convert the physical array of ranges to a new cudf column
      std::unique_ptr<cudf::column> cudf_offsets = global_ranges_to_cudf_offsets(
        a.ranges(), num_chars, legate::Memory::Kind::SYSTEM_MEM, stream, mr);

      // Copy the physical array of chars to a new cudf column
      auto chars_buf = copy_physical_array_to_device(chars, stream);
      rmm::device_buffer null_mask{};
      cudf::size_type null_count{0};
      if (a.nullable()) {
        null_mask =
          null_mask_bools_to_bits(a.null_mask(), legate::Memory::Kind::SYSTEM_MEM, stream, mr);
        null_count =
          cudf::null_count(static_cast<const cudf::bitmask_type*>(null_mask.data()), 0, num_rows());
      }
      // Create a new string column from ranges and chars
      return cudf::make_strings_column(num_rows(),
                                       std::move(cudf_offsets),
                                       std::move(chars_buf),
                                       null_count,
                                       std::move(null_mask));
    } else {
      throw std::invalid_argument("nested dtype " + array_->type().to_string() +
                                  " isn't supported");
    }
  }
  rmm::device_buffer null_mask{};
  cudf::size_type null_count{0};
  if (array_->nullable()) {
    legate::PhysicalArray ary = array_->get_physical_array();
    null_mask =
      null_mask_bools_to_bits(ary.null_mask(), legate::Memory::Kind::SYSTEM_MEM, stream, mr);
    null_count =
      cudf::null_count(static_cast<const cudf::bitmask_type*>(null_mask.data()), 0, num_rows());
  }
  return std::make_unique<cudf::column>(
    cudf_type_,
    num_rows(),
    copy_physical_array_to_device(array_->get_physical_array(), stream),
    std::move(null_mask),
    null_count);
}

std::string LogicalColumn::repr(size_t max_num_items) const
{
  std::stringstream ss;
  ss << "LogicalColumn(";
  if (unbound()) {
    ss << "data=unbound ";
    if (array_->nullable()) { ss << "null_mask=unbound "; }
    ss << "dtype=" << array_->type() << ")";
  } else {
    legate::PhysicalArray ary = array_->get_physical_array();

    // Notice, `get_physical_array()` returns host memory always
    ss << legate::dataframe::repr(
            ary, max_num_items, legate::Memory::Kind::SYSTEM_MEM, cudaStream_t{0})
       << ")";
  }
  return ss.str();
}

namespace task {

std::string PhysicalColumn::repr(legate::Memory::Kind mem_kind,
                                 cudaStream_t stream,
                                 size_t max_num_items) const
{
  std::stringstream ss;
  ss << "PhysicalColumn(";
  ss << legate::dataframe::repr(array_, max_num_items, mem_kind, stream) << ")";
  return ss.str();
}

cudf::column_view PhysicalColumn::column_view() const
{
  if (unbound()) {
    throw std::runtime_error(
      "Cannot call `.column_view()` on a unbound LogicalColumn, please bind it using "
      "`.move_into()`");
  }

  const void* data                    = nullptr;
  const cudf::bitmask_type* null_mask = nullptr;
  cudf::size_type null_count          = 0;
  cudf::size_type offset              = 0;
  std::vector<cudf::column_view> children;

  if (array_.nested()) {
    if (array_.type().code() == legate::Type::Code::STRING) {
      const legate::StringPhysicalArray a = array_.as_string_array();
      const legate::PhysicalArray chars   = a.chars();
      const auto num_chars                = chars.data().shape<1>().volume();

      std::unique_ptr<cudf::column> cudf_offsets = global_ranges_to_cudf_offsets(
        a.ranges(), num_chars, legate::Memory::Kind::GPU_FB_MEM, ctx_->stream(), ctx_->mr());

      // To keep the offsets alive beyond this function, we push it to temporaries before
      // adding it as the first child.
      tmp_cols_.push_back(std::move(cudf_offsets));
      children.push_back(tmp_cols_.back()->view());

      // The second child is the character column
      data = read_accessor_as_1d_bytes(chars.data());
    } else {
      throw std::invalid_argument("nested dtype " + array_.type().to_string() + " isn't supported");
    }
  } else {
    data = read_accessor_as_1d_bytes(array_.data());
  }
  if (array_.nullable()) {
    tmp_null_masks_.push_back(null_mask_bools_to_bits(
      array_.null_mask(), legate::Memory::Kind::GPU_FB_MEM, ctx_->stream(), ctx_->mr()));
    null_mask  = static_cast<const cudf::bitmask_type*>(tmp_null_masks_.back().data());
    null_count = cudf::null_count(null_mask, 0, num_rows(), ctx_->stream());
  }
  return cudf::column_view(cudf_type_, num_rows(), data, null_mask, null_count, offset, children);
}

namespace {

struct move_into_fn {
  template <typename T, std::enable_if_t<cudf::is_rep_layout_compatible<T>() ||
                                         cudf::is_fixed_width<T>()>* = nullptr>
  void operator()(GPUTaskContext* ctx,
                  legate::PhysicalArray& array,
                  std::unique_ptr<cudf::column> column,
                  cudaStream_t stream)
  {
    const auto num_rows = column->size();
    const auto cudf_col = column->view();
    if (array.nullable()) {
      if (column->nullable()) {
        auto null_mask = array.null_mask().create_output_buffer<bool, 1>(legate::Point<1>(num_rows),
                                                                         /* bind_buffer = */ true);
        null_mask_bits_to_bools(num_rows, null_mask.ptr(0), cudf_col.null_mask(), stream);
      } else {
        auto null_mask = array.null_mask().create_output_buffer<bool, 1>(legate::Point<1>(num_rows),
                                                                         true /* bind_buffer */);
        LEGATE_CHECK_CUDA(cudaMemsetAsync(
          null_mask.ptr(0), std::numeric_limits<bool>::max(), num_rows * sizeof(bool), stream));
      }
    }

    MemAlloc mem_alloc = ctx->mr()->release_buffer(cudf_col);
    // std::cout << "mem_alloc.valid: " << mem_alloc.valid()
    //           << ", mem_alloc.nbytes: " << mem_alloc.nbytes()
    //           << ", cudf_col.nbytes: " << cudf_col.size() * sizeof(T) << std::endl;
    if (mem_alloc.valid()) {
      array.data().bind_untyped_data(mem_alloc.buffer(), cudf_col.size());
    } else {
      auto out =
        array.data().create_output_buffer<T, 1>(legate::Point<1>(num_rows), true /* bind_buffer */);
      auto cudf_ptr = static_cast<char const*>(cudf_col.head()) + (cudf_col.offset() * cudf::size_of(cudf_col.type()));
      LEGATE_CHECK_CUDA(cudaMemcpyAsync(out.ptr(0),
                                        cudf_ptr,
                                        cudf_col.size() * cudf::size_of(cudf_col.type()),
                                        cudaMemcpyDeviceToDevice,
                                        stream));
    }
  }

  template <typename T, std::enable_if_t<std::is_same_v<T, cudf::string_view>>* = nullptr>
  void operator()(GPUTaskContext* ctx,
                  legate::PhysicalArray& array,
                  std::unique_ptr<cudf::column> column,
                  cudaStream_t stream)
  {
    const auto num_rows = column->size();
    const auto cudf_col = column->view();
    if (array.nullable()) {
      if (column->nullable()) {
        auto null_mask = array.null_mask().create_output_buffer<bool, 1>(legate::Point<1>(num_rows),
                                                                         /* bind_buffer = */ true);
        null_mask_bits_to_bools(num_rows, null_mask.ptr(0), cudf_col.null_mask(), stream);
      } else {
        auto null_mask = array.null_mask().create_output_buffer<bool, 1>(legate::Point<1>(num_rows),
                                                                         true /* bind_buffer */);
        LEGATE_CHECK_CUDA(cudaMemsetAsync(
          null_mask.ptr(0), std::numeric_limits<bool>::max(), num_rows * sizeof(bool), stream));
      }
    }

    cudf::strings_column_view str_col{*column};
    legate::StringPhysicalArray ary = array.as_string_array();

    if (str_col.size() == 0) {
      ary.ranges().data().bind_empty_data();
      ary.chars().data().bind_empty_data();
      return;
    }

    auto ranges_size = str_col.offsets().size() - 1;
    auto ranges      = ary.ranges().data().create_output_buffer<legate::Rect<1>, 1>(
      ranges_size, true /* bind_buffer */);

    cudf_offsets_to_local_ranges(ranges_size, ranges.ptr(0), str_col.offsets(), ctx->stream());

    if (str_col.offsets().offset() != 0) {
      throw std::runtime_error("string column seems sliced, which is currently not supported.");
    }
    // TODO: maybe attach the chars data instead of copying.
    auto nbytes = str_col.chars_size(stream);
    auto chars = ary.chars().data().create_output_buffer<int8_t, 1>(nbytes, true /* bind_buffer */);
    LEGATE_CHECK_CUDA(cudaMemcpyAsync(
      chars.ptr(0), str_col.chars_begin(stream), nbytes, cudaMemcpyDeviceToDevice, stream));
  }

  template <typename T,
            std::enable_if_t<!(cudf::is_fixed_width<T>() ||
                               std::is_same_v<T, cudf::string_view> ||
                               cudf::is_fixed_width<T>())>* = nullptr>
  void operator()(GPUTaskContext* ctx,
                  legate::PhysicalArray& array,
                  std::unique_ptr<cudf::column> column,
                  cudaStream_t stream)
  {
    // TODO: support lists
    throw std::invalid_argument("move_into(): type not supported");
  }
};

}  // namespace

void PhysicalColumn::move_into(std::unique_ptr<cudf::column> column)
{
  if (!unbound()) { throw std::invalid_argument("Cannot call `.move_into()` on a bound column"); }
  // NOTE(seberg): In some cases (replace nulls) we expect no nulls, but
  //     seem to get a nullable column.  So also check `has_nulls()`.
  if (column->nullable() && !array_.nullable() && column->has_nulls()) {
    throw std::invalid_argument(
      "move_into(): the cudf column is nullable while the PhysicalArray isn't");
  }
  cudf::type_dispatcher(
    column->type(), move_into_fn{}, ctx_, array_, std::move(column), ctx_->stream());
}

void PhysicalColumn::bind_empty_data() const
{
  if (!unbound()) {
    throw std::invalid_argument("Cannot call `.bind_empty_data()` on a bound column");
  }

  if (array_.nullable()) { array_.null_mask().bind_empty_data(); }
  if (array_.nested()) {
    legate::StringPhysicalArray ary = array_.as_string_array();
    ary.ranges().data().bind_empty_data();
    ary.chars().data().bind_empty_data();
  } else {
    array_.data().bind_empty_data();
  }
}

}  // namespace task

namespace argument {

legate::Variable add_next_input(legate::AutoTask& task, const LogicalColumn& col, bool broadcast)
{
  add_next_scalar(task, col.cudf_type());
  add_next_scalar(task, col.unbound() ? -1 : static_cast<int64_t>(col.num_rows()));
  auto arr      = col.get_logical_array();
  auto variable = task.add_input(arr);
  if (broadcast) { task.add_constraint(legate::broadcast(variable, {0})); }
  return variable;
}

legate::Variable add_next_output(legate::AutoTask& task, const LogicalColumn& col)
{
  add_next_scalar(task, col.cudf_type());
  add_next_scalar(task, col.unbound() ? -1 : static_cast<int64_t>(col.num_rows()));
  return {task.add_output(col.get_logical_array())};
}

}  // namespace argument

}  // namespace legate::dataframe
