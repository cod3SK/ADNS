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

All features are directly observed from tshark or arithmetic of observed values.
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

### Why tshark's A-side is not the rule

tshark's conv output lists endpoints in `A <-> B` order based on which appeared
first in the capture, not on any semantic property of the flow. The same
physical flow processed in two different captures may produce opposite A-sides.
`parse_conv_output()` therefore returns **neutral** endpoint names (`ep_a`,
`ep_b`, `bytes_ab`, `bytes_ba`, `pkts_ab`, `pkts_ba`) with no src/dst naming.
`assemble._make_flow()` calls `canonicalize_orientation()` once to assign src
and dst, then assigns the directional byte/packet counts accordingly.

Both tshark passes (conv stats and tcp.flags) join on `orientation_key` — the
unordered `(min_ep, max_ep)` pair — so the join succeeds regardless of capture
order.

## Extraction — pcap file

```bash
# from the project root (ml/ on PYTHONPATH)
python -m adns_flows --pcap capture.pcap --out flows.csv

# with an explicit tshark path
python -m adns_flows --pcap capture.pcap --out flows.csv \
    --tshark "C:\Program Files\Wireshark\tshark.exe"
```

## Extraction — live interface

```bash
python -m adns_flows --iface eth0 --window 60 --out flows.csv
```

`--window` is the capture duration in seconds (default 60). Two tshark passes
run over the same window:

- **Pass A** — `tshark -q -z conv,tcp -z conv,udp` → bidirectional byte/packet counts per conversation (neutral `ep_a`/`ep_b` names)
- **Pass B** — `tshark -T fields -Y "ip && tcp" … -e tcp.flags` → per-flow TCP flag counts (keyed on `orientation_key`, direction-agnostic)

## Python API

```python
from adns_flows import extract_flows, flows_to_dataframe, find_tshark, validate_matrix

tshark = find_tshark()

# from a pcap
flows = extract_flows(tshark, pcap="capture.pcap")
df = flows_to_dataframe(flows)          # columns: IDENTITY_COLUMNS + FEATURE_COLUMNS

# corpus building — pin attacker IP as src
flows = extract_flows(tshark, pcap="attack.pcap", prefer_src="10.0.0.5")

# validate before scoring
from adns_flows import FEATURE_COLUMNS
validate_matrix(df, list(df.columns[len(IDENTITY_COLUMNS):]))   # raises SchemaError on drift
```

## Schema validation

`validate_matrix(data, columns)` raises `SchemaError` if the column list does
not exactly match `FEATURE_COLUMNS` in name and order. This replaces
`MetaFeatureBuilder._match_shape`'s silent pad/truncate — the new pipeline
fails loud so schema drift is caught at integration time.

## Running the tests

```bash
# from the project root
pytest ml/adns_flows/tests/ -v

# pure-Python parsing and orientation tests only (no tshark required)
pytest ml/adns_flows/tests/ -v -k "not tshark"
```

tshark-dependent tests are automatically skipped when tshark is not on PATH
and `TSHARK_BIN` is not set. The parsing, orientation, and schema validation
tests run without tshark.

## tshark binary resolution

The module probes in this order (same as `api/app.py`):

1. `sys._MEIPASS/tshark/tshark.exe` (PyInstaller bundle)
2. `TSHARK_BIN` environment variable
3. `C:\Program Files\Wireshark\tshark.exe` (Windows default)
4. `shutil.which("tshark")` (PATH)
