# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from pylibcudf.types import DataType, TypeId

from legate_dataframe import LogicalTable

_to_plc_type_id = {
    dt.Boolean: TypeId.BOOL8,
    # dt.Null: ,
    dt.String: TypeId.STRING,
    # dt.Binary: ,
    # dt.Date: type_id.TIMESTAMP_DAYS,  # not sure about these times/dates
    # dt.Time: TypeId.TIMESTAMP_NANOSECONDS,
    dt.Int8: TypeId.INT8,
    dt.Int16: TypeId.INT16,
    dt.Int32: TypeId.INT32,
    dt.Int64: TypeId.INT64,
    dt.UInt8: TypeId.UINT8,
    dt.UInt16: TypeId.UINT16,
    dt.UInt32: TypeId.UINT32,
    dt.UInt64: TypeId.UINT64,
    dt.Float32: TypeId.FLOAT32,
    dt.Float64: TypeId.FLOAT64,
}


_to_plc_type = {i_t: DataType(plc_tn) for i_t, plc_tn in _to_plc_type_id.items()}

_from_legate_df_type = {v: k for k, v in _to_plc_type.items()}


def to_plc_type(ibis_type):
    # TODO: Should we take nullability into account? (and if, how?)
    return _to_plc_type[type(ibis_type)]


def to_ibis_type(plc_type):
    return _from_legate_df_type[plc_type]


# TODO: Do I need the full Schema/Datamapper? (likely reasonable abstractions to add)
def infer_schema_from_logical_table(ldf: LogicalTable) -> sch.Schema:
    info = []

    for col_name in ldf.get_column_names():
        col = ldf[col_name]
        ibis_dtype = _from_legate_df_type[col.dtype()]

        info.append((col_name, ibis_dtype))

    return sch.Schema.from_tuples(info)


def get_names_dtypes_from_schema(
    schema: sch.Schema,
) -> Tuple[List[str], List[DataType]]:
    names = schema.names
    types = [_to_plc_type[type(t)] for t in schema.types]
    return names, types
