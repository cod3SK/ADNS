# adns_flows — canonical flow-extraction module

Single source of truth for turning packet captures into ADNS model features.
Used identically for live capture and offline PCAP processing to eliminate
train/serve skew.

## Feature contract

`schema.py` defines two ordered tuples:

| Constant | Purpose |
|---|---|
| `IDENTITY_COLUMNS` | ts, src_ip, dst_ip, src_port, dst_port — joins/labeling only, never fed to the model |
| `FEATURE_COLUMNS` | 21 ordered model inputs (see below) |

```
proto, duration, src_bytes, dst_bytes, total_bytes, src_pkts, dst_pkts,
total_pkts, bytes_ratio, pkts_ratio, src_mean_pkt_size, dst_mean_pkt_size,
bytes_per_sec, pkts_per_sec, dst_port_bucket,
syn_count, ack_count, rst_count, fin_count, psh_count, urg_count
```

All features are directly observed from NFStream or arithmetic of observed values.
No Zeek-only fields, no hashing into invented ranges.

## Orientation rule

**Orientation is part of the feature contract**, as load-bearing as the column
list. `bytes_ratio`, `pkts_ratio`, `src_mean_pkt_size`, `dst_mean_pkt_size`,
`src_bytes`, `dst_bytes`, `src_pkts`, and `dst_pkts` all depend on which
endpoint is called "src" and which is "dst". If live scoring and training use
different rules, these features are mirrored relative to each other — silent
poison for the model.

### The rule (implemented in `schema.canonicalize_orientation`)

**Default rule:** `src = the endpoint with the lower (ip, port) tuple** under
Python's lexicographic string ordering of the IP address, then numeric port
comparison for tie-breaking. This ordering is:

- **Total** — every pair has a definite answer, including same-IP/different-port
- **Stable** — same inputs always produce the same output
- **Deterministic** — no randomness or per-run state

Tie-break when both endpoints are identical (degenerate, should not occur in
real traffic): the `endpoint_a` argument is returned as src.

**`prefer_src` override:** corpus builders can pass a known attacker IP as
`prefer_src` to `build_flows()` or `extract_flows()`. If the IP matches one
endpoint, that endpoint is pinned as src regardless of the default rule. If it
matches neither, the default rule applies. Live capture always omits
`prefer_src` so both paths use the same deterministic rule.

### Why NFStream's initiator-side is not the canonical rule

NFStream assigns `src`/`dst` based on which endpoint sent the first packet
(initiator = src). This is non-deterministic across captures of the same flow
in different directions. `extract_nfstream.py` therefore calls
`canonicalize_orientation()` immediately after receiving each flow and swaps
the directional counts if needed — the canonical rule always wins over
NFStream's initiator assignment.

## Extraction — pcap file

```bash
# from the project root (ml/ on PYTHONPATH)
python -m adns_flows --pcap capture.pcap --out flows.csv

# with multiple NFStream meter workers (default 1)
python -m adns_flows --pcap capture.pcap --out flows.csv --n-meters 2
```

## Python API

```python
from adns_flows.extract_nfstream import extract_flows_nfstream, flows_to_dataframe_nfstream
from adns_flows.schema import validate_matrix, FEATURE_COLUMNS, IDENTITY_COLUMNS

# from a pcap
flows = extract_flows_nfstream("capture.pcap")
df = flows_to_dataframe_nfstream(flows)   # columns: IDENTITY_COLUMNS + FEATURE_COLUMNS

# corpus building — pin attacker IP as src
flows = extract_flows_nfstream("attack.pcap", prefer_src="10.0.0.5")

# validate before scoring (raises SchemaError on column drift)
validate_matrix(df[list(FEATURE_COLUMNS)], FEATURE_COLUMNS)
```

## Schema validation

`validate_matrix(data, columns)` raises `SchemaError` if the column list does
not exactly match `FEATURE_COLUMNS` in name and order. Called before every
`model.predict()` in `serving_nfstream.NfstreamScorer.score_matrix()`.

## Running the tests

```bash
# from the project root
pytest ml/adns_flows/tests/ -v
```

All 131 tests run without tshark. NFStream is required (installed via
`requirements-desktop.txt` or `api/requirements.txt`).
