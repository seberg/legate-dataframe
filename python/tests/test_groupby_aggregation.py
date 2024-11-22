# Copyright (c) 2024, NVIDIA CORPORATION
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


from typing import Iterable, List, Tuple

import cudf
import pytest

from legate_dataframe import LogicalTable
from legate_dataframe.lib.groupby_aggregation import (
    AggregationKind,
    groupby_aggregation,
)
from legate_dataframe.testing import assert_frame_equal


def kind_to_cudf_agg(kind: AggregationKind) -> str:
    if kind == AggregationKind.PRODUCT:
        return "prod"
    return kind.name.lower()


def cudf_groupby(
    df: cudf.DataFrame,
    keys: List[str],
    column_aggregations: Iterable[Tuple[str, AggregationKind, str]],
) -> cudf.DataFrame:
    """Help function that perform cudf groupby using the legate syntax"""
    ret = {}
    group = df.groupby(keys, as_index=False)
    for in_col, kind, out_col in column_aggregations:
        res = group.agg([kind_to_cudf_agg(kind)])
        for key in keys:
            # We assume the key row order is stable between cudf groupby runs
            ret[key] = res[key]._columns[0]
        ret[out_col] = res[in_col]._columns[0]
    return cudf.DataFrame(ret)


@pytest.mark.parametrize(
    "keys,df",
    [
        (
            ["k1"],
            cudf.DataFrame(
                {
                    "k1": ["x", "x", "y", "y", "z"],
                    "d1": [1, 2, 0, 4, 1],
                    "d2": [3, 2, 4, 5, 1],
                }
            ),
        ),
        (
            ["k1", "k2"],
            cudf.DataFrame(
                {
                    "k1": ["x", "y", "y", "y", "x"],
                    "d1": [1, 2, 0, 4, 1],
                    "k2": ["y", "x", "y", "x", "y"],
                    "d2": [3, 2, 4, 5, 1],
                }
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "aggs",
    [
        [("d1", AggregationKind.SUM, "sum")],
        [("d1", AggregationKind.MIN, "min"), ("d1", AggregationKind.MAX, "max")],
        [("d2", AggregationKind.MEAN, "mean"), ("d1", AggregationKind.PRODUCT, "prod")],
    ],
)
def test_aggregation(df, keys, aggs):
    expect = cudf_groupby(df, keys, aggs)

    tbl = LogicalTable.from_cudf(df)
    result = groupby_aggregation(tbl, keys, aggs)

    assert_frame_equal(result, expect, ignore_row_order=True)
