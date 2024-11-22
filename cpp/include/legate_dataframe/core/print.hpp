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

#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <legate.h>

namespace legate {
namespace dataframe {

std::string repr(const legate::PhysicalStore& store,
                 size_t max_num_items,
                 legate::Memory::Kind mem_kind,
                 cudaStream_t stream);

std::string repr_ranges(const legate::PhysicalStore& store,
                        size_t max_num_items,
                        legate::Memory::Kind mem_kind,
                        cudaStream_t stream);

std::string repr(const legate::PhysicalArray& ary,
                 size_t max_num_items,
                 legate::Memory::Kind mem_kind,
                 cudaStream_t stream);

template <typename T>
std::string repr(const std::vector<T>& vec)
{
  if (vec.empty()) { return "[]"; }
  std::stringstream ss;
  ss << "[";
  for (const auto& item : vec) {
    ss << item << ", ";
  }
  ss << "\b\b]";  // use two ANSI backspace characters '\b' to overwrite the final ','
  return ss.str();
}

template <typename KeyT, typename ValueT>
std::string repr(const std::map<KeyT, ValueT>& mapping)
{
  if (mapping.empty()) { return "{}"; }
  std::stringstream ss;
  ss << "{";
  for (const auto& [key, value] : mapping) {
    ss << key << ": " << value << ", ";
  }
  ss << "\b\b}";  // use two ANSI backspace characters '\b' to overwrite the final ','
  return ss.str();
}

}  // namespace dataframe
}  // namespace legate
