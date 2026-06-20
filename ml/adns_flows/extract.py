"""
tshark invocation and output parsing for ADNS flow extraction.

Two passes over the same source (pcap file or live interface):
  Pass A — `tshark -q -z conv,tcp -z conv,udp`
            Yields per-conversation raw endpoint pairs and directional byte/
            packet counts. Output uses NEUTRAL names (ep_a, ep_b, bytes_ab,
            bytes_ba, pkts_ab, pkts_ba) — NO src/dst assignment yet. Naming
            happens once in assemble.py after canonicalize_orientation().
  Pass B — `tshark -T fields -Y "ip && tcp" -e ip.src ... -e tcp.flags`
            Aggregates TCP flag counts keyed by orientation_key (unordered),
            so both passes always join on the same symmetric key.

The conv parser (parse_conv_output / parse_tshark_bytes) is extracted here so
api/app.py can import it in a future task instead of maintaining its own copy.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

from .schema import orientation_key


# ── tshark binary resolution ───────────────────────────────────────────────
# Mirrors api/app.py:_find_tshark so both paths use identical probe order:
# bundled (PyInstaller), TSHARK_BIN env, Windows default install, PATH.

def find_tshark() -> str | None:
    if hasattr(sys, "_MEIPASS"):
        bundled = os.path.join(sys._MEIPASS, "tshark", "tshark.exe")
        if os.path.isfile(bundled):
            return bundled
    for candidate in [
        os.environ.get("TSHARK_BIN", ""),
        r"C:\Program Files\Wireshark\tshark.exe",
        shutil.which("tshark") or "",
    ]:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _tshark_env(tshark_bin: str) -> dict:
    """Prepend tshark dir to PATH so bundled DLLs (Npcap, Qt) are found."""
    env = os.environ.copy()
    tshark_dir = os.path.dirname(os.path.abspath(tshark_bin))
    env["PATH"] = tshark_dir + os.pathsep + env.get("PATH", "")
    env.setdefault("WIRESHARK_RUN_FROM_BUILD_DIRECTORY", "0")
    return env


def _popen_kwargs(tshark_bin: str) -> dict:
    kw: dict = dict(
        cwd=os.path.dirname(os.path.abspath(tshark_bin)),
        env=_tshark_env(tshark_bin),
    )
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kw


# ── tshark 4.x conv output parser ─────────────────────────────────────────
#
# Format (tshark 4.x, no pipe chars in data rows, human-readable byte units):
#   10.18.0.20:54617  <->  160.79.104.10:443   12 1530 bytes   58 85 kB   70 87 kB   1.588   0.0476
#
# The `(\S+):(\d+)` pattern handles IPv6 via greedy backtracking: the regex
# engine backtracks \S+ until the trailing `:(\d+)` can match, so
# `fe80::463:4c2e:f13c:ab17:5353` parses as IP=`fe80::463:4c2e:f13c:ab17`
# port=5353. See design-decisions/0013-tshark-4x-batch-conv-format.md.

_CONV_RE = re.compile(
    r"(\S+):(\d+)\s+<->\s+(\S+):(\d+)\s+"
    r"(\d+)\s+([\d.]+)\s+(\S+)\s+"   # frames_ba  bytes_ba_val  bytes_ba_unit
    r"(\d+)\s+([\d.]+)\s+(\S+)\s+"   # frames_ab  bytes_ab_val  bytes_ab_unit
    r"\d+\s+[\d.]+\s+\S+\s+"         # total frame/byte cols (skip)
    r"([\d.]+)\s+"                    # rel_start
    r"([\d.]+)"                       # duration
)


def parse_tshark_bytes(value: str, unit: str) -> int:
    """Convert a tshark human-readable byte value+unit to integer bytes.

    Handles: bytes, kB/kib (×1024), MB/mib (×1024²), GB/gib (×1024³).
    Returns 0 on parse failure.
    """
    try:
        n = float(value)
        u = unit.lower()
        if u in ("kb", "kib"):
            return int(n * 1024)
        if u in ("mb", "mib"):
            return int(n * 1024 * 1024)
        if u in ("gb", "gib"):
            return int(n * 1024 * 1024 * 1024)
        return int(n)  # "bytes" or unknown unit — treat as raw integer
    except (ValueError, TypeError):
        return 0


def parse_conv_output(text: str) -> list[dict]:
    """Parse raw tshark -z conv,tcp/-z conv,udp stdout into conv-flow dicts.

    Returns dicts with NEUTRAL endpoint names — no src/dst assignment yet:
        proto        — "TCP" or "UDP"
        ep_a         — (ip, port) tuple for tshark's A-side (left of <->)
        ep_b         — (ip, port) tuple for tshark's B-side (right of <->)
        bytes_ab     — bytes in the A→B direction
        bytes_ba     — bytes in the B→A direction
        pkts_ab      — frames in the A→B direction
        pkts_ba      — frames in the B→A direction
        duration     — flow duration in seconds
        rel_start    — flow start relative to capture start

    src/dst orientation is NOT assigned here. Call canonicalize_orientation()
    in assemble.py to determine which endpoint is src vs dst.
    """
    flows: list[dict] = []
    current_proto = "TCP"
    for line in text.splitlines():
        if "TCP Conversations" in line:
            current_proto = "TCP"
        elif "UDP Conversations" in line:
            current_proto = "UDP"
        m = _CONV_RE.search(line)
        if not m:
            continue
        (a_ip, a_port, b_ip, b_port,
         frames_ba, bytes_ba_val, bytes_ba_unit,
         frames_ab, bytes_ab_val, bytes_ab_unit,
         rel_start, duration) = m.groups()
        flows.append({
            "proto": current_proto,
            "ep_a": (a_ip, int(a_port)),
            "ep_b": (b_ip, int(b_port)),
            "bytes_ab": parse_tshark_bytes(bytes_ab_val, bytes_ab_unit),
            "bytes_ba": parse_tshark_bytes(bytes_ba_val, bytes_ba_unit),
            "pkts_ab": int(frames_ab),
            "pkts_ba": int(frames_ba),
            "duration": float(duration),
            "rel_start": float(rel_start),
        })
    return flows


# ── Pass B: TCP flag aggregation ──────────────────────────────────────────

# Standard TCP flag bits in the flags byte
_FLAG_BITS: dict[str, int] = {
    "fin": 0x01,
    "syn": 0x02,
    "rst": 0x04,
    "psh": 0x08,
    "ack": 0x10,
    "urg": 0x20,
}


def parse_flag_lines(lines: list[str]) -> dict[tuple, dict[str, int]]:
    """Aggregate TCP flag counts from tshark -T fields flag output lines.

    Input lines are tab-separated: ip.src, ip.dst, tcp.srcport, tcp.dstport,
    tcp.flags (hex, e.g. 0x00000002).

    Returns a dict keyed by orientation_key (unordered) → {flag_name: count}.
    Counts aggregate across both directions — the same key is produced for
    packets in either direction, so both passes always join correctly.
    """
    counts: dict[tuple, dict[str, int]] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        src_ip, dst_ip, src_port_s, dst_port_s, flags_s = parts[:5]
        try:
            src_port = int(src_port_s)
            dst_port = int(dst_port_s)
            flags = int(flags_s, 0)  # 0 base auto-detects 0x prefix
        except (ValueError, TypeError):
            continue
        key = orientation_key(src_ip, src_port, dst_ip, dst_port)
        bucket = counts.setdefault(key, {f: 0 for f in _FLAG_BITS})
        for flag_name, bit in _FLAG_BITS.items():
            if flags & bit:
                bucket[flag_name] += 1
    return counts


# ── Two-pass tshark invocation ────────────────────────────────────────────

def _source_args(
    pcap: str | None,
    iface: str | None,
    window_sec: int,
) -> list[str]:
    if pcap:
        return ["-r", pcap]
    if not iface:
        raise ValueError("one of pcap or iface must be provided")
    return ["-i", iface, "-a", f"duration:{window_sec}"]


def run_pass_a(
    tshark_bin: str,
    *,
    pcap: str | None = None,
    iface: str | None = None,
    window_sec: int = 60,
) -> list[dict]:
    """Run tshark conv stats and return parsed conv-flow dicts (neutral names)."""
    src = _source_args(pcap, iface, window_sec)
    cmd = [tshark_bin] + src + ["-q", "-z", "conv,tcp", "-z", "conv,udp"]
    result = subprocess.run(
        cmd,
        capture_output=True,
        timeout=max(window_sec + 30, 90),
        **_popen_kwargs(tshark_bin),
    )
    text = result.stdout.decode("utf-8", errors="replace")
    return parse_conv_output(text)


def run_pass_b(
    tshark_bin: str,
    *,
    pcap: str | None = None,
    iface: str | None = None,
    window_sec: int = 60,
) -> dict[tuple, dict[str, int]]:
    """Run tshark TCP-flags field pass and return aggregated flag counts.

    Keyed by orientation_key (unordered) so the join in assemble.py is
    direction-agnostic — matching regardless of which side tshark listed as A.
    """
    src = _source_args(pcap, iface, window_sec)
    cmd = (
        [tshark_bin] + src
        + ["-T", "fields", "-Y", "ip && tcp"]
        + ["-e", "ip.src", "-e", "ip.dst",
           "-e", "tcp.srcport", "-e", "tcp.dstport", "-e", "tcp.flags"]
        + ["-E", "separator=\t", "-E", "header=n"]
    )
    result = subprocess.run(
        cmd,
        capture_output=True,
        timeout=max(window_sec + 30, 90),
        **_popen_kwargs(tshark_bin),
    )
    lines = result.stdout.decode("utf-8", errors="replace").splitlines()
    return parse_flag_lines(lines)
