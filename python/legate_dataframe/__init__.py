# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import legate.core

legate.core.get_legate_runtime()  # noqa: F401

from legate_dataframe._version import __version__  # noqa: F401, E402
from legate_dataframe.lib.core.column import LogicalColumn  # noqa: F401, E402
from legate_dataframe.lib.core.table import LogicalTable  # noqa: F401, E402
