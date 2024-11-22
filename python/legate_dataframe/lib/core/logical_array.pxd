# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

cdef extern from "<legate.h>" nogil:
    cdef cppclass cpp_LogicalArray "legate::LogicalArray":
        pass
