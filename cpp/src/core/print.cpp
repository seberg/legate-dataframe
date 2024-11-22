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
#include <stdexcept>

#include <legate/cuda/cuda.h>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/print.hpp>

namespace legate {
namespace dataframe {

namespace {

/*
 * Hopefully temporary custom implementation of << for __half
 */
std::ostream& operator<<(std::ostream& os, const __half& val) { return os << (float)val; }

template <typename VAL>
std::string repr(const VAL* ptr, legate::Memory::Kind mem_kind, cudaStream_t stream)
{
  VAL val;
  if (is_device_mem(mem_kind)) {
    LEGATE_CHECK_CUDA(cudaMemcpyAsync(&val, ptr, sizeof(VAL), cudaMemcpyDeviceToHost, stream));
  } else {
    val = *ptr;
  }
  std::stringstream ss;
  ss << val;
  return ss.str();
}

template <typename VAL>
std::string repr(const AccessorRO<VAL, 1>& acc,
                 const legate::PointInRectIterator<1>& it,
                 legate::Memory::Kind mem_kind,
                 cudaStream_t stream)
{
  const VAL* ptr = acc.ptr(*it);
  return repr<VAL>(ptr, mem_kind, stream);
}

template <typename VAL>
std::string repr(const AccessorRO<VAL, 1>& acc,
                 const Rect<1>& shape,
                 size_t max_num_items,
                 legate::Memory::Kind mem_kind,
                 cudaStream_t stream)
{
  const auto nelem = shape.volume();
  if (nelem == 0) { return "[]"; }
  std::stringstream ss;
  ss << "[";
  if (nelem < 10 || nelem <= max_num_items) {
    for (legate::PointInRectIterator<1> it(shape); it.valid(); ++it) {
      ss << repr<VAL>(acc, it, mem_kind, stream) << ", ";
    }
  } else {
    legate::Rect<1> first_3_items(shape.lo, shape.lo + 2);
    for (legate::PointInRectIterator<1> it(first_3_items); it.valid(); ++it) {
      ss << repr<VAL>(acc, it, mem_kind, stream) << ", ";
    }
    ss << "..., ";
    legate::Rect<1> last_3_items(shape.hi - 2, shape.hi);
    for (legate::PointInRectIterator<1> it(last_3_items); it.valid(); ++it) {
      ss << repr<VAL>(acc, it, mem_kind, stream) << ", ";
    }
  }
  ss << "\b\b]";  // use two ANSI backspace characters '\b' to overwrite the final ','
  return ss.str();
}

struct repr_1d_store_fn {
  template <legate::Type::Code CODE, std::enable_if_t<!legate::is_complex<CODE>::value>* = nullptr>
  std::string operator()(const legate::PhysicalStore& store,
                         size_t max_num_items,
                         legate::Memory::Kind mem_kind,
                         cudaStream_t stream)
  {
    using VAL = legate::type_of<CODE>;
    if (store.dim() != 1) { throw std::invalid_argument("only 1-dim Store supported"); }
    auto acc   = store.read_accessor<VAL, 1>();
    auto shape = store.shape<1>();
    return repr<VAL>(acc, shape, max_num_items, mem_kind, stream);
  }

  template <legate::Type::Code CODE, std::enable_if_t<legate::is_complex<CODE>::value>* = nullptr>
  std::string operator()(const legate::PhysicalStore& store,
                         size_t max_num_items,
                         legate::Memory::Kind mem_kind,
                         cudaStream_t stream)
  {
    // CCCL had a bug with printing complex values, can probably work around but
    // we don't use printing a lot, so disable complex.
    throw std::invalid_argument("Complex repr disabled as a quick fix for a CCCL issue.");
  }
};

}  // namespace

std::string repr(const legate::PhysicalStore& store,
                 size_t max_num_items,
                 legate::Memory::Kind mem_kind,
                 cudaStream_t stream)
{
  return legate::type_dispatch(
    store.code(), repr_1d_store_fn{}, store, max_num_items, mem_kind, stream);
}

std::string repr_ranges(const legate::PhysicalStore& store,
                        size_t max_num_items,
                        legate::Memory::Kind mem_kind,
                        cudaStream_t stream)
{
  auto acc   = store.read_accessor<Rect<1>, 1>();
  auto shape = store.shape<1>();
  return repr(acc, shape, max_num_items, mem_kind, stream);
}

std::string repr(const legate::PhysicalArray& ary,
                 size_t max_num_items,
                 legate::Memory::Kind mem_kind,
                 cudaStream_t stream)
{
  auto nelem = ary.shape<1>().volume();
  std::stringstream ss;
  ss << "Array(";
  if (nelem > max_num_items) { ss << "len=" << nelem << " "; }
  if (ary.nested()) {
    if (ary.type().code() == legate::Type::Code::STRING) {
      const legate::StringPhysicalArray a = ary.as_string_array();
      ss << "ranges=" << repr_ranges(a.ranges().data(), max_num_items, mem_kind, stream) << " ";
      ss << "chars=" << repr(a.chars().data(), max_num_items, mem_kind, stream) << " ";
    } else {
      throw std::invalid_argument(ary.type().to_string() + " isn't supported");
    }
  } else {
    ss << "data=" << repr(ary.data(), max_num_items, mem_kind, stream) << " ";
  }
  if (ary.nullable()) {
    ss << "null_mask=" << repr(ary.null_mask(), max_num_items, mem_kind, stream) << " ";
  }
  ss << "dtype=" << ary.type() << ")";
  return ss.str();
}

}  // namespace dataframe
}  // namespace legate
