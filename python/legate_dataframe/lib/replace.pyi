# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from legate_dataframe.lib.core.column import LogicalColumn
from legate_dataframe.lib.core.scalar import ScalarLike

def replace_nulls(
    col: LogicalColumn,
    replacement: ScalarLike,
) -> LogicalColumn: ...
