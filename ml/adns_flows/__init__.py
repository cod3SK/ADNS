"""
adns_flows — canonical flow-extraction module for ADNS.

Importable from both agent/ and api/. See README.md for usage.
"""
from .schema import (
    FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    Flow,
    SchemaError,
    canonicalize_orientation,
    flow_to_row,
    orientation_key,
    validate_matrix,
)
from .nfstream_config import NFSTREAM_FEATURE_PARAMS, make_nfstream_kwargs
from .extract_nfstream import extract_flows_nfstream, flows_to_dataframe_nfstream

__all__ = [
    "FEATURE_COLUMNS",
    "IDENTITY_COLUMNS",
    "Flow",
    "SchemaError",
    "canonicalize_orientation",
    "flow_to_row",
    "orientation_key",
    "validate_matrix",
    "NFSTREAM_FEATURE_PARAMS",
    "make_nfstream_kwargs",
    "extract_flows_nfstream",
    "flows_to_dataframe_nfstream",
]
