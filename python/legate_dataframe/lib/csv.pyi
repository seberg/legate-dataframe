# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
from typing import Iterable

from numpy.typing import DTypeLike
from pylibcudf.types import DataType

from legate_dataframe.lib.core.table import LogicalTable

__all__ = ["csv_read", "csv_write"]

def csv_write(
    tbl: LogicalTable, path: pathlib.Path | str, delimiter: str = ","
) -> None: ...
def csv_read(
    glob_string: pathlib.Path | str,
    *,
    dtypes: Iterable[DTypeLike | DataType],
    na_filter: bool = False,
    delimiter: str = ",",
    usecols: Iterable[str] | None = None,
) -> LogicalTable: ...
