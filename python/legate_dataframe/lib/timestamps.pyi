# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numpy.typing import DTypeLike

from legate_dataframe.lib.core.column import LogicalColumn

def to_timestamps(
    col: LogicalColumn,
    timestamp_type: DTypeLike,
    format_pattern: str,
) -> LogicalColumn: ...
def extract_timepart(
    col: LogicalColumn,
    resolution: str,
) -> LogicalColumn: ...
