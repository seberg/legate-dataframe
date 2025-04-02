# Copyright (c) 2023-2025, NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob

import cudf
import cupy
import dask_cudf
import pytest
from legate.core import get_legate_runtime

from legate_dataframe import LogicalColumn, LogicalTable
from legate_dataframe.lib.parquet import parquet_read, parquet_read_array, parquet_write
from legate_dataframe.testing import assert_frame_equal, std_dataframe_set


@pytest.mark.parametrize("df", std_dataframe_set())
def test_write(tmp_path, df):
    print(tmp_path)
    tbl = LogicalTable.from_cudf(df)

    parquet_write(tbl, path=tmp_path)
    get_legate_runtime().issue_execution_fence(block=True)

    res = dask_cudf.read_parquet(tmp_path).compute().reset_index(drop=True)
    assert_frame_equal(res, df)


@pytest.mark.parametrize("columns", [None, ["b"], ["a", "b"], ["b", "a"], []])
@pytest.mark.parametrize("df", std_dataframe_set())
def test_read(tmp_path, df, columns, npartitions=2, glob_string="/*"):
    ddf = dask_cudf.from_cudf(df, npartitions=npartitions)
    ddf.to_parquet(path=tmp_path, index=False)

    has_cols = columns is None or all(c in df.keys() for c in columns)
    if not has_cols:
        with pytest.raises(ValueError):
            parquet_read(str(tmp_path) + glob_string, columns=columns)
    else:
        tbl = parquet_read(str(tmp_path) + glob_string, columns=columns)
        if columns is not None:
            df = df.loc[:, columns]

        if tbl.get_column_names():
            assert_frame_equal(tbl, df)
        else:
            # Table is empty (and has no rows). cudf has an index, though.
            assert len(df.keys()) == 0


def test_read_single_rows(tmp_path, glob_string="/*"):
    df = cudf.DataFrame({"a": cupy.arange(1, dtype="int64")})
    ddf = dask_cudf.from_cudf(df, npartitions=1)
    ddf.to_parquet(path=tmp_path, index=False)
    tbl = parquet_read(str(tmp_path) + glob_string)
    assert_frame_equal(tbl, df)


def test_read_many_files_per_rank(tmp_path, glob_string="/*"):
    # Use uneven number to test splitting
    df = cudf.DataFrame({"a": cupy.arange(983, dtype="int64")})
    npartitions = 100
    ddf = dask_cudf.from_cudf(df, npartitions=npartitions)
    ddf.to_parquet(path=tmp_path, index=False)
    # Test that we really have many files hoped for:
    assert len(glob.glob(str(tmp_path) + glob_string)) == npartitions
    tbl = parquet_read(str(tmp_path) + glob_string)

    # NOTE: Right now the C-code does not attempt to "natural" sort parquet
    #       files.  So more with more than 10 files the order of rows is not
    #       preserved at this time.
    assert_frame_equal(tbl.to_cudf().sort_values(by="a"), df)


def test_read_array(tmp_path, npartitions=2, glob_string="/*"):
    cn = pytest.importorskip("cupynumeric")

    c = cudf.Series(cupy.arange(2, 10002, dtype="float32"))
    c = c.mask([True, False] * 5000)
    df = cudf.DataFrame(
        {
            "a": cupy.arange(10000, dtype="float32"),
            "b": cupy.arange(1, 10001, dtype="float32"),
            "c": c,  # only column with masked values
            "d": cupy.arange(3, 10003, dtype="float32"),
        }
    )
    ddf = dask_cudf.from_cudf(df, npartitions=npartitions)
    ddf.to_parquet(path=tmp_path, index=False)

    tbl = parquet_read(str(tmp_path) + glob_string, columns=["b", "a", "d"])
    arr_from_tbl = tbl.to_array()
    arr = parquet_read_array(
        str(tmp_path) + glob_string, columns=["b", "a", "d"], nullable=False
    )
    arr = cn.asarray(arr)

    assert arr.dtype == arr_from_tbl.dtype
    assert cn.array_equal(arr_from_tbl, arr)

    # Check if the mask behavior seems right for column "c"
    arr = parquet_read_array(str(tmp_path) + glob_string, columns=["c"], nullable=True)
    col_from_arr = LogicalColumn(arr.project(1, 0))
    col = parquet_read(str(tmp_path) + glob_string, columns=["c"])["c"]

    assert_frame_equal(col_from_arr, col)


def test_read_array_large(tmp_path, npartitions=1, glob_string="/*"):
    cn = pytest.importorskip("cupynumeric")

    # Create an array so large, that we should chunk (should be fine for testing)
    df = cudf.DataFrame({"a": cupy.ones(2**26, dtype="uint8")})
    ddf = dask_cudf.from_cudf(df, npartitions=npartitions)
    ddf.to_parquet(path=tmp_path, index=False)
    del df, ddf

    arr = parquet_read_array(str(tmp_path) + glob_string, columns=["a"], nullable=False)
    assert cn.asarray(arr).sum() == 2**26
