# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from legate.core import AutoTask


cdef extern from "legate.h" nogil:
    cdef cppclass cpp_AutoTask "legate::AutoTask":
        pass

cdef cpp_AutoTask *get_auto_task_handle(task: AutoTask)
