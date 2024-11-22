# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf
import pytest
from cudf.testing.testing import assert_column_equal

from legate_dataframe import LogicalColumn, LogicalTable
from legate_dataframe.testing import assert_frame_equal, std_dataframe_set


@pytest.mark.parametrize("cudf_col", [cudf.DataFrame({"a": range(10)})._columns[0]])
def test_column_round_trip(cudf_col):
    col = LogicalColumn.from_cudf(cudf_col)
    cudf_res = col.to_cudf()
    assert_column_equal(cudf_col, cudf_res)


@pytest.mark.parametrize("df", std_dataframe_set())
def test_table_round_trip(df):
    tbl = LogicalTable.from_cudf(df)
    res = tbl.to_cudf()
    assert_frame_equal(df, res)
