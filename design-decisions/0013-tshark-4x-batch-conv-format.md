# 0013 — tshark 4.x batch conv output format compatibility

- **Status:** Accepted
- **Phase:** 3 — Desktop packaging and distribution

## Context

The batch capture pipeline (`_BatchCaptureAgent` in `api/app.py`) processes each
15-second pcap with two tshark passes. Pass 1 runs `tshark -z conv,tcp -z conv,udp`
to extract per-flow directional byte and packet counts. The original `_BATCH_CONV_RE`
regex was written against an older tshark conv output format that used pipe
characters (`|`) as column separators in data rows and raw integer byte values:

```
192.168.1.1:443  <->  10.0.0.1:52345  |  5  4095  |  |  7  1234  |  |  12  5329  |  0.000000  |  0.234567
```

tshark 4.x (Wireshark 4.0+) emits a different format: **no pipe characters in
data rows**, and byte values are human-readable with SI units (`bytes`, `kB`,
`MB`, `GB`):

```
10.18.0.20:54617  <->  160.79.104.10:443    12 1530 bytes    58 85 kB    70 87 kB    1.588544000    0.0476
```

Because the regex never matched, `_run_conv_stats` always returned an empty list.
`_batches_processed` never incremented and no batch flows were ever written to the
database, making the Batch Analysis tab permanently empty.

## Decision

Replace `_BATCH_CONV_RE` with a regex that matches the tshark 4.x format:

```python
_BATCH_CONV_RE = re.compile(
    r"(\S+):(\d+)\s+<->\s+(\S+):(\d+)\s+"
    r"(\d+)\s+([\d.]+)\s+(\S+)\s+"   # frames_ba  bytes_ba_val  bytes_ba_unit
    r"(\d+)\s+([\d.]+)\s+(\S+)\s+"   # frames_ab  bytes_ab_val  bytes_ab_unit
    r"\d+\s+[\d.]+\s+\S+\s+"         # total (skip)
    r"([\d.]+)\s+"                    # rel_start
    r"([\d.]+)"                       # duration
)
```

Add `_parse_tshark_bytes(value, unit) -> int` to convert the human-readable byte
value + unit string to an integer (handling `bytes`, `kB`/`kib`, `MB`/`mib`,
`GB`/`gib`). Update `_run_conv_stats` to unpack the new groups and call this
helper.

The `(\S+):(\d+)` address pattern handles both IPv4 (`10.18.0.20:54617`) and IPv6
(`fe80::463:4c2e:f13c:ab17:5353`) correctly via greedy backtracking.

## Consequences

- Batch flows are now parsed and ingested correctly; the Batch Analysis tab
  populates after the first 15-second capture window.
- The regex is tied to tshark 4.x output. If the format changes again in a future
  major version, `_run_conv_stats` will silently return empty flows (same symptom
  as the original bug). The `batch.batches_processed` counter in `/capture_status`
  is the canary: if it stays at 0 while the agent is running and the interface has
  traffic, the conv regex has drifted.
- Byte values from human-readable units are approximate at the `kB`/`MB` scale
  (nearest integer after multiplying by 1024). The precision is more than
  sufficient for anomaly scoring, which uses total byte volume as a coarse feature.
