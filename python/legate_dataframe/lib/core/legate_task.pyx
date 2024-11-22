# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libc.stdint cimport uintptr_t
from legate.core import AutoTask


cdef cpp_AutoTask *get_auto_task_handle(task: AutoTask):
    if not isinstance(task, AutoTask):
        raise ValueError(f"expects a `AutoTask` instance got a {type(task)}")
    cdef uintptr_t raw_handle = task.raw_handle
    return <cpp_AutoTask*> raw_handle
