# Copyright (c) 2023-2024, NVIDIA CORPORATION
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


import cudf
import cupy
import pytest

from legate_dataframe import LogicalTable
from legate_dataframe.lib.join import BroadcastInput, JoinType, join
from legate_dataframe.testing import assert_frame_equal


def make_param():
    """Parameters for 'cudf_lhs,cudf_rhs,left_on,right_on'"""

    for a_num_rows in range(60, 100, 20):
        for b_num_rows in range(60, 100, 20):
            a = cupy.arange(a_num_rows, dtype="int64")
            b = cupy.arange(b_num_rows, dtype="int64")
            cupy.random.shuffle(a)
            cupy.random.shuffle(b)
            yield (
                cudf.DataFrame({"a": a, "payload_a": cupy.arange(a.size)}),
                cudf.DataFrame({"b": b, "payload_b": cupy.arange(b.size) * -1}),
                ["a"],
                ["b"],
            )
            yield (
                cudf.DataFrame({"a": a, "payload_a": [str(i) for i in range(a.size)]}),
                cudf.DataFrame(
                    {"b": b, "payload_b": [str(i * -1) for i in range(b.size)]}
                ),
                ["a"],
                ["b"],
            )
    yield (
        cudf.DataFrame({"a": [1, 2, 3, 4, 5], "payload_a": cupy.arange(5)}),
        cudf.DataFrame({"b": [1, 1, 2, 2, 5, 6], "payload_b": cupy.arange(6) * -1}),
        ["a"],
        ["b"],
    )


def to_cudf_how(how: JoinType) -> str:
    if how == JoinType.FULL:
        return "outer"
    return how.name.lower()


@pytest.mark.parametrize(
    "how",
    (
        JoinType.INNER,
        JoinType.LEFT,
        JoinType.FULL,
    ),
)
@pytest.mark.parametrize(
    "broadcast",
    (BroadcastInput.AUTO, BroadcastInput.LEFT, BroadcastInput.RIGHT),
)
@pytest.mark.parametrize("cudf_lhs,cudf_rhs,left_on,right_on", make_param())
def test_basic(how: JoinType, cudf_lhs, cudf_rhs, left_on, right_on, broadcast):
    lg_lhs = LogicalTable.from_cudf(cudf_lhs)
    lg_rhs = LogicalTable.from_cudf(cudf_rhs)

    if (how == JoinType.FULL and broadcast != BroadcastInput.AUTO) or (
        how == JoinType.LEFT and broadcast == BroadcastInput.LEFT
    ):
        # In these cases we don't support broadcasting (at least for now)
        with pytest.raises(RuntimeError):
            res = join(
                lg_lhs,
                lg_rhs,
                lhs_keys=left_on,
                rhs_keys=right_on,
                join_type=how,
                broadcast=broadcast,
            )
        return

    expect = cudf.merge(
        cudf_lhs,
        cudf_rhs,
        left_on=left_on,
        right_on=right_on,
        how=to_cudf_how(how),
    )

    res = join(
        lg_lhs,
        lg_rhs,
        lhs_keys=left_on,
        rhs_keys=right_on,
        join_type=how,
        broadcast=broadcast,
    )
    assert_frame_equal(res, expect, ignore_row_order=True)


def test_column_names():
    lhs = LogicalTable.from_cudf(cudf.DataFrame({"key": [1, 2, 3], "data0": [1, 2, 3]}))
    rhs = LogicalTable.from_cudf(cudf.DataFrame({"key": [3, 2, 1], "data1": [1, 2, 3]}))

    res = join(
        lhs,
        rhs,
        lhs_keys=["key"],
        rhs_keys=["key"],
        join_type=JoinType.INNER,
        lhs_out_columns=["data0", "key"],
        rhs_out_columns=["data1"],
    )
    assert_frame_equal(
        res,
        cudf.DataFrame({"data0": [1, 2, 3], "key": [1, 2, 3], "data1": [3, 2, 1]}),
        ignore_row_order=True,
    )
