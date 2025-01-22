# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
from time import perf_counter_ns

# Use selected test queries as defined for ibis (not official)
# Query ten includes a limit, so skip for now.
from ibis.backends.tests.tpc.h.test_queries import test_01  # , test_10
from legate.core import get_legate_runtime

from legate_dataframe.experimental_ibis import LegateBackend


def run_test(test, file_glob):
    test = test.__wrapped__

    con = LegateBackend().connect()
    names = inspect.signature(test).parameters.keys()

    # Starting test (this reads data already)
    start_time = perf_counter_ns()
    args = (con.read_parquet(file_glob.format(name=name)) for name in names)
    # If the parquet files include an "ignore" column, drop it:
    args = (t.drop("ignore") if "ignore" in t.columns else t for t in args)
    # The test is nicely wrapped up, unwrap it
    result = test(*args)

    res = con.to_legate(result)
    get_legate_runtime().issue_execution_fence(block=True)
    return res, (perf_counter_ns() - start_time) / 1e9


if __name__ == "__main__":
    import sys

    file_glob = sys.argv[1]

    for test in (test_01,):  # test_01,
        print("Running test", test)
        res, time = run_test(test, file_glob=file_glob)
        print("    Duration in seconds:", time)
