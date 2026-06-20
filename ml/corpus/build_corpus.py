"""
PCAP-to-labeled-corpus pipeline for UNSW-NB15 and Gotham Dataset 2025.

Reads raw pcap files, extracts flows using the canonical adns_flows extractor
(same code as the live scoring path), labels each flow, and writes a labeled
parquet corpus.

Supported datasets
------------------
  --dataset unsw   (default)
      Uses a UNSW-NB15 ground-truth CSV for time-window label matching.
      Every extracted flow is matched against attack rows on endpoint pair +
      proto + time window (±TIME_TOLERANCE seconds).

  --dataset gotham
      Uses directory-structure labeling (no GT CSV, no timestamps).
      raw/benign/*.pcap          → all flows benign  (label=0)
      raw/malicious/<t>/*.pcap   → all flows attack   (label=1, attack_cat=t)
      See corpus/gotham_labels.py for full schema findings (Step 0).

Key invariants (shared across both datasets)
--------------------------------------------
Extraction invariant: run_pass_a + run_pass_b + build_flows are the functions
used here AND by extract_flows() at serve time.  They share one code path.

Orientation invariant:
  - Benign flows: prefer_src=None  → default rule (src = lower (ip, port))
  - Attack flows: prefer_src=<attacker_ip>  → attacker pinned as src
  Both cases go through canonicalize_orientation(), never around it.

Three-way labeling
------------------
Every extracted flow gets exactly one of three outcomes:

  ATTACK (label=1)
    UNSW:   matched an attack label row on endpoint pair + proto + time window.
    Gotham: flow is in a malicious PCAP (label determined at PCAP level).
    Assembled with prefer_src=<attacker_ip>.

  BENIGN (label=0)
    UNSW:   flow extracted cleanly but matched NO attack label row in the GT CSV.
            A no-match is NEVER a reason to drop.
    Gotham: flow is in a benign PCAP.
    RETAINED as a benign example.

  DROPPED (n_dropped_unprocessable)
    Flow could not be processed at all: tshark failed on the pcap (reason
    'extraction_fail'), the pcap header timestamp was unreadable (reason
    'no_timestamp'), or assembly raised an unexpected exception (reason 'other').
    Only genuine processing failures are in this bucket.

Label-row accounting
--------------------
UNSW:   label_rows_total = attack rows in GT CSV; label_rows_matched = distinct
        attack rows matched by ≥1 flow; label_rows_unmatched = rows matched
        by 0 flows.  WARNING when >20% unmatched.
Gotham: label_rows_total = number of attack PCAPs; label_rows_matched = attack
        PCAPs that produced ≥1 attack flow.

Class-balance gate
------------------
build_corpus() / build_corpus_gotham() call assert_sane_balance() BEFORE writing
the parquet.  Pass allow_skewed=True to override.

Output columns: IDENTITY_COLUMNS + FEATURE_COLUMNS + ['label', 'attack_cat']
The 'ts' column contains absolute epoch seconds (pcap_start + rel_start).
"""
from __future__ import annotations

import dataclasses
import logging
import os
import struct
import subprocess as _sp
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adns_flows import (
    FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    build_flows,
    extract_flows_nfstream,
    flow_to_row,
    orientation_key,
    run_pass_a,
    run_pass_b,
)
from adns_flows.schema import Flow as _Flow

log = logging.getLogger(__name__)

# Robustness margin against Bro/tshark connection-tracking timing differences.
TIME_TOLERANCE = 1.0  # seconds

_PROTO_NORM: dict[str, str] = {"tcp": "TCP", "udp": "UDP"}

OUTPUT_COLUMNS: list[str] = (
    list(IDENTITY_COLUMNS) + list(FEATURE_COLUMNS) + ["label", "attack_cat"]
)

# Drop reason keys — used in CorpusStats.dropped_reasons
REASON_NO_TIMESTAMP         = "no_timestamp"
REASON_EXTRACTION_FAIL      = "extraction_fail"
REASON_FLAGS_UNEXTRACTABLE  = "flags_unextractable"
REASON_OTHER                = "other"
# NFStream path — reachable reasons: 'extraction_fail', 'other'.
# 'no_timestamp' is unreachable (NFStream reads absolute timestamps from packets).
# 'flags_unextractable' is unreachable (NFStream is single-pass; no separate flag stage).

# Default per-source-IP cap on degenerate one-sided flood flows
# (src_pkts<=1, dst_pkts==0). Limits trivial-shape dominance in the corpus
# while preserving a representative sample of each flood source's traffic.
DEFAULT_FLOOD_CAP = 3_000

# Warning threshold: if this fraction of attack rows matched nothing, warn.
UNMATCHED_WARN_FRAC = 0.20

# Hard limit: if this fraction of total flows seen were silently dropped,
# HALT before writing parquet.  A rate above this almost always indicates an
# extraction bug (e.g. get_pcap_start_epoch returning None for an unrecognised
# pcap format) rather than normal cleaning.
MAX_DROP_FRAC = 0.10


# ── exceptions ─────────────────────────────────────────────────────────────

class CorpusBalanceError(ValueError):
    """Raised when the labeled corpus fails the class-balance sanity check.

    Caught by assert_sane_balance().  The message includes the diagnosis and
    recommended corrective action.
    """


class CorpusDropRateError(ValueError):
    """Raised when the fraction of silently-dropped flows exceeds MAX_DROP_FRAC.

    A high drop rate is an extraction bug signal, not normal cleaning.
    Fix the underlying cause; do NOT raise the threshold.
    """


# ── stats container ────────────────────────────────────────────────────────

@dataclasses.dataclass
class CorpusStats:
    """Counters for one build (or one pcap batch within a build).

    Three-way labeling counters
    ---------------------------
    n_attack                : flows labeled as attack (label=1)
    n_benign                : flows labeled as benign (label=0), including
                              all no-match flows that were RETAINED
    n_dropped_unprocessable : flows dropped for genuine processing reasons
                              (see dropped_reasons for the breakdown)
    dropped_reasons         : {'no_timestamp': N, 'extraction_fail': N, 'other': N}

    Label-row accounting
    --------------------
    label_rows_total     : attack rows (label=1) in the GT CSV
    label_rows_matched   : distinct attack rows matched by ≥1 flow
    (label_rows_unmatched is a computed property)
    """
    n_attack:                int = 0
    n_benign:                int = 0
    n_dropped_unprocessable: int = 0
    dropped_reasons:         dict[str, int] = dataclasses.field(default_factory=dict)
    label_rows_total:        int = 0
    label_rows_matched:      int = 0

    @property
    def label_rows_unmatched(self) -> int:
        return self.label_rows_total - self.label_rows_matched

    @property
    def total_kept(self) -> int:
        return self.n_attack + self.n_benign

    @property
    def benign_frac(self) -> float:
        return self.n_benign / max(self.total_kept, 1)

    @property
    def attack_frac(self) -> float:
        return self.n_attack / max(self.total_kept, 1)

    def merge(self, other: CorpusStats) -> None:
        """Accumulate another batch's counters into this stats object in place."""
        self.n_attack                += other.n_attack
        self.n_benign                += other.n_benign
        self.n_dropped_unprocessable += other.n_dropped_unprocessable
        for reason, count in other.dropped_reasons.items():
            self.dropped_reasons[reason] = self.dropped_reasons.get(reason, 0) + count


# ── pcap epoch reader ──────────────────────────────────────────────────────

def _pcapng_start_epoch(f) -> float | None:
    """Extract first-packet timestamp from an open pcapng file (f.seek(0) already done)."""
    try:
        raw = f.read(12)              # block_type(4) + btl(4) + bo_magic(4)
        if len(raw) < 12:
            return None
        bo_le = struct.unpack("<I", raw[8:12])[0]
        if bo_le == 0x1A2B3C4D:
            endian = "<"
        elif bo_le == 0x4D3C2B1A:
            endian = ">"
        else:
            return None
        shb_btl = struct.unpack(f"{endian}I", raw[4:8])[0]
        f.seek(shb_btl)               # skip SHB entirely

        ts_resol = 1_000_000          # default: microseconds

        while True:
            hdr = f.read(8)           # block_type + block_total_length
            if len(hdr) < 8:
                return None
            block_type, btl = struct.unpack(f"{endian}II", hdr)

            if block_type == 0x00000001:   # IDB — may carry if_tsresol (option 9)
                body = f.read(btl - 8)     # body + trailing btl
                opt_end = len(body) - 4    # exclude trailing btl from option scan
                opt_off = 8                # after link_type(2)+reserved(2)+snap_len(4)
                while opt_off + 4 <= opt_end:
                    opt_code, opt_len = struct.unpack(f"{endian}HH", body[opt_off:opt_off+4])
                    opt_off += 4
                    if opt_code == 0:
                        break
                    if opt_code == 9 and opt_len >= 1 and opt_off < opt_end:
                        rb = body[opt_off]
                        ts_resol = (2 if (rb & 0x80) else 10) ** (rb & 0x7F)
                    opt_off += opt_len
                    if opt_len % 4:
                        opt_off += 4 - (opt_len % 4)

            elif block_type == 0x00000006: # EPB — interface_id(4)+ts_high(4)+ts_low(4)
                preamble = f.read(12)
                if len(preamble) < 12:
                    return None
                _, ts_high, ts_low = struct.unpack(f"{endian}III", preamble)
                return ((ts_high << 32) | ts_low) / ts_resol

            elif block_type == 0x00000002: # OPB — iface_id(2)+drops(2)+ts_high(4)+ts_low(4)
                preamble = f.read(12)
                if len(preamble) < 12:
                    return None
                _, _, ts_high, ts_low = struct.unpack(f"{endian}HHII", preamble)
                return ((ts_high << 32) | ts_low) / ts_resol

            else:                          # SPB or unknown — no timestamp, skip
                f.seek(btl - 8, 1)
    except (OSError, struct.error):
        return None


def get_pcap_start_epoch(pcap_path: str | Path) -> float | None:
    """Return the first-packet timestamp from a pcap or pcapng file.

    Handles pcap (LE/BE, microsecond/nanosecond) and pcapng (SHB + EPB/OPB blocks).
    Returns None (not 0.0) on any read or parse error so callers can distinguish
    a failed parse from a pcap that genuinely started at epoch 0.
    """
    try:
        with open(pcap_path, "rb") as f:
            hdr = f.read(24)
            if len(hdr) < 24:
                return None
            magic = struct.unpack("<I", hdr[:4])[0]
            if magic == 0xA1B2C3D4:       # pcap LE microseconds
                endian, nano = "<", False
            elif magic == 0xD4C3B2A1:     # pcap BE microseconds
                endian, nano = ">", False
            elif magic == 0xA1B23C4D:     # pcap LE nanoseconds
                endian, nano = "<", True
            elif magic == 0x4D3CB2A1:     # pcap BE nanoseconds
                endian, nano = ">", True
            elif magic == 0x0A0D0D0A:     # pcapng
                f.seek(0)
                return _pcapng_start_epoch(f)
            else:
                return None
            pkt = f.read(16)
            if len(pkt) < 16:
                return None
            ts_sec, ts_sub = struct.unpack(f"{endian}II", pkt[:8])
            divisor = 1_000_000_000 if nano else 1_000_000
            return ts_sec + ts_sub / divisor
    except OSError:
        return None


# ── label CSV loader ───────────────────────────────────────────────────────

def _parse_port(value: Any) -> int:
    """Convert a port value (int, float, or hex string) to int."""
    try:
        s = str(value).strip()
        if s.startswith(("0x", "0X")):
            return int(s, 16)
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def load_label_index(
    csv_path: str | Path,
) -> tuple[dict[tuple, list[dict]], int]:
    """Load a UNSW-NB15 ground-truth CSV into an endpoint-pair index.

    Returns (index, total_attack_rows) where:
      index            : dict keyed by orientation_key → [label_row_dict, ...]
      total_attack_rows: count of rows with label=1 — used for accounting

    Each entry in the index carries a private '_row_idx' field (int) that
    identifies the original CSV row.  _apply_labels() adds matched '_row_idx'
    values to a set so build_corpus() can compute label_rows_matched.

    Accepts UNSW-NB15 canonical column names (srcip, sport, dstip, dsport,
    proto, stime, ltime, attack_cat, label) plus common variants.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Column normalisation — handles two GT file layouts:
    #   Layout A (UNSW-NB15_1..4.csv per-flow feature files):
    #       srcip, sport, dstip, dsport, proto, stime, ltime, label, attack_cat
    #   Layout B (NUSW-NB15_GT.csv per-event attack list, no 'label' column):
    #       source_ip, source_port, destination_ip, destination_port,
    #       protocol, start_time, last_time, attack_category
    _COL_ALIASES: dict[str, list[str]] = {
        "srcip":      ["src_ip", "ip_src", "source_ip"],
        "dstip":      ["dst_ip", "ip_dst", "destination_ip"],
        "sport":      ["src_port", "srcport", "source_port"],
        "dsport":     ["dst_port", "dstport", "destination_port"],
        "proto":      ["protocol"],
        "stime":      ["start_time"],
        "ltime":      ["last_time"],
        "attack_cat": ["attackcat", "attack_category"],
    }
    rename: dict[str, str] = {}
    for col in df.columns:
        for canonical, aliases in _COL_ALIASES.items():
            if col in aliases and col != canonical:
                rename[col] = canonical
                break
    if rename:
        df = df.rename(columns=rename)

    # Layout B has no 'label' column — every row is an attack event.
    all_attacks_implicit = "label" not in df.columns
    if all_attacks_implicit:
        df["label"] = 1

    index: dict[tuple, list[dict]] = {}
    total_attack_rows = 0
    skipped = 0

    for row_idx, row in enumerate(df.to_dict("records")):
        srcip = str(row.get("srcip", "")).strip()
        dstip = str(row.get("dstip", "")).strip()
        if not srcip or not dstip:
            skipped += 1
            continue
        sport  = _parse_port(row.get("sport",  0))
        dsport = _parse_port(row.get("dsport", 0))
        proto  = str(row.get("proto", "")).strip().lower()
        try:
            stime = float(row.get("stime", 0))
            ltime = float(row.get("ltime", 0))
            label = int(row.get("label", 0))
        except (ValueError, TypeError):
            skipped += 1
            continue
        attack_cat = str(row.get("attack_cat", "")).strip()

        if label == 1:
            total_attack_rows += 1

        key = orientation_key(srcip, sport, dstip, dsport)
        index.setdefault(key, []).append({
            "srcip":      srcip,
            "dstip":      dstip,
            "sport":      sport,
            "dsport":     dsport,
            "proto":      proto,
            "stime":      stime,
            "ltime":      ltime,
            "label":      label,
            "attack_cat": attack_cat,
            "_row_idx":   row_idx,   # private: for label-row accounting
        })

    if skipped:
        log.warning("Skipped %d label rows with missing/invalid fields", skipped)
    return index, total_attack_rows


# ── tshark extraction (shared with parity test) ────────────────────────────

def extract_pcap_flows(
    pcap_path: str | Path,
    tshark_bin: str,
) -> tuple[list[dict], dict]:
    """Run both tshark passes and return (conv_dicts, flag_counts) unassembled.

    The live path (adns_flows.extract_flows) calls run_pass_a + run_pass_b
    internally; this wrapper makes the raw data available so the parity test
    can verify that corpus extraction == live extraction.
    """
    pcap = str(pcap_path)
    return run_pass_a(tshark_bin, pcap=pcap), run_pass_b(tshark_bin, pcap=pcap)


# ── pass-B chunking helpers ────────────────────────────────────────────────

def _find_editcap(tshark_bin: str) -> str:
    """Return editcap path from the same Wireshark install as tshark."""
    tshark_dir = Path(tshark_bin).parent
    for name in ("editcap.exe", "editcap"):
        candidate = tshark_dir / name
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"editcap not found alongside tshark at {tshark_dir}"
    )


def _editcap_env(tshark_bin: str) -> dict:
    """Env dict for editcap subprocess — mirrors adns_flows extract._tshark_env."""
    env = os.environ.copy()
    tshark_dir = os.path.dirname(os.path.abspath(tshark_bin))
    env["PATH"] = tshark_dir + os.pathsep + env.get("PATH", "")
    env.setdefault("WIRESHARK_RUN_FROM_BUILD_DIRECTORY", "0")
    return env


def _merge_flag_counts(
    merged: dict[tuple, dict[str, int]],
    chunk: dict[tuple, dict[str, int]],
) -> None:
    """Add chunk flag counts into merged in-place (sum per flag per key)."""
    for key, counts in chunk.items():
        if key not in merged:
            merged[key] = dict(counts)
        else:
            for flag, n in counts.items():
                merged[key][flag] = merged[key].get(flag, 0) + n


def run_pass_b_chunked(
    tshark_bin: str,
    pcap_path: str | Path,
    *,
    chunk_size: int = 500_000,
    editcap_timeout: int = 600,
    chunk_timeout_window: int = 90,
) -> dict[tuple, dict[str, int]]:
    """Run pass B by splitting pcap into chunks of chunk_size packets.

    Uses editcap to split, then runs run_pass_b on each chunk and sums flag
    counts.  The result is mathematically identical to a single-pass run:
    each packet appears in exactly one chunk; orientation_key is symmetric so
    counts aggregate correctly across chunks.

    Parameters
    ----------
    tshark_bin          : absolute path to tshark
    pcap_path           : PCAP to process
    chunk_size          : max packets per chunk (default 500 000)
    editcap_timeout     : seconds to allow editcap to split the file
    chunk_timeout_window: window_sec passed to run_pass_b; timeout becomes
                          max(chunk_timeout_window+30, 90) seconds per chunk

    Raises
    ------
    RuntimeError   if editcap produces no chunk files
    subprocess.TimeoutExpired / OSError  propagated from run_pass_b
    """
    editcap = _find_editcap(tshark_bin)
    pcap_path = Path(pcap_path)
    env = _editcap_env(tshark_bin)
    popen_kw: dict = dict(env=env, cwd=str(Path(editcap).parent))
    if sys.platform == "win32":
        popen_kw["creationflags"] = _sp.CREATE_NO_WINDOW

    with tempfile.TemporaryDirectory() as tmp:
        chunk_stem = os.path.join(tmp, "chunk_.pcap")

        # editcap -c N in.pcap chunk_.pcap → chunk__00001.pcap, chunk__00002.pcap …
        _sp.run(
            [editcap, "-c", str(chunk_size), str(pcap_path), chunk_stem],
            capture_output=True,
            timeout=editcap_timeout,
            **popen_kw,
        )

        chunk_files = sorted(Path(tmp).glob("chunk_*.pcap"))
        if not chunk_files:
            raise RuntimeError(
                f"editcap produced no chunk files for {pcap_path.name}"
            )

        log.info(
            "  chunked pass B: %d chunks  (chunk_size=%d) for %s",
            len(chunk_files), chunk_size, pcap_path.name,
        )

        merged: dict[tuple, dict[str, int]] = {}
        for chunk in chunk_files:
            chunk_counts = run_pass_b(
                tshark_bin, pcap=str(chunk), window_sec=chunk_timeout_window,
            )
            _merge_flag_counts(merged, chunk_counts)

        return merged


# ── flood-cap sampling ─────────────────────────────────────────────────────

def apply_flood_cap(
    df: pd.DataFrame,
    cap: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, int]:
    """Cap degenerate one-sided flood flows per source IP.

    A "degenerate flood flow" satisfies ALL of:
      - label == 1 (attack)
      - src_pkts <= 1
      - dst_pkts == 0

    For each unique src_ip, at most `cap` such flows are retained; the rest
    are discarded (random sample with fixed `seed` for reproducibility).
    Non-degenerate flows and benign flows pass through unchanged.

    Returns
    -------
    (filtered_df, n_dropped)  where n_dropped is the number of discarded flows.
    """
    if cap <= 0:
        return df, 0

    flood_mask = (df["label"] == 1) & (df["src_pkts"] <= 1) & (df["dst_pkts"] == 0)
    floods  = df[flood_mask]
    others  = df[~flood_mask]

    if len(floods) == 0:
        return df, 0

    rng = np.random.default_rng(seed)
    keep_idx: list = []
    for _, group in floods.groupby("src_ip"):
        idx = group.index.to_numpy()
        if len(idx) <= cap:
            keep_idx.extend(idx.tolist())
        else:
            keep_idx.extend(rng.choice(idx, size=cap, replace=False).tolist())

    floods_capped = floods.loc[keep_idx]
    n_dropped = len(floods) - len(floods_capped)

    result = (
        pd.concat([others, floods_capped])
        .sort_values(["ts", "src_ip", "src_port", "dst_ip", "dst_port"])
        .reset_index(drop=True)
    )
    return result, n_dropped


# ── per-flow label matching ────────────────────────────────────────────────

def _match_label(
    conv: dict,
    abs_ts: float,
    key: tuple,
    label_index: dict[tuple, list[dict]],
) -> dict | None:
    """Return the first label row matching this conv on proto + time window, or None."""
    candidates = label_index.get(key)
    if not candidates:
        return None
    conv_proto = conv["proto"]  # "TCP" or "UDP"
    for row in candidates:
        if _PROTO_NORM.get(row["proto"], row["proto"].upper()) != conv_proto:
            continue
        if row["stime"] - TIME_TOLERANCE <= abs_ts <= row["ltime"] + TIME_TOLERANCE:
            return row
    return None


# ── three-way labeling core ────────────────────────────────────────────────

def _apply_labels(
    conv_dicts: list[dict],
    flag_counts: dict,
    pcap_start_epoch: float | None,
    label_index: dict[tuple, list[dict]],
    matched_attack_row_indices: set[int],
) -> tuple[list[dict], CorpusStats]:
    """Label a batch of conv_dicts against the label_index.

    Three-way outcomes per flow:

      ATTACK  — matched an attack label row → label=1, prefer_src=attacker_ip
      BENIGN  — no match (or matched a label=0 row) → label=0, prefer_src=None
                A no-match flow is ALWAYS retained.  This is the critical fix:
                "no label row" means benign traffic, not corrupt data.
      DROPPED — genuine processing failure (no_timestamp or assembly error)

    Parameters
    ----------
    conv_dicts               : raw conv dicts from run_pass_a
    flag_counts              : flag dict from run_pass_b
    pcap_start_epoch         : epoch seconds for the pcap's first packet,
                               or None if the header was unreadable
    label_index              : from load_label_index()
    matched_attack_row_indices: set mutated in place — _row_idx of every
                               attack label row matched by at least one flow

    Returns (output_rows, batch_stats).  output_rows are ready to append to the
    corpus; batch_stats holds per-batch counters to merge into the run total.
    """
    stats = CorpusStats()
    rows: list[dict] = []

    # If the pcap header timestamp is unreadable we cannot reconstruct abs_ts,
    # so we cannot match against the GT CSV time window.  Drop all convs.
    if pcap_start_epoch is None:
        n = len(conv_dicts)
        if n:
            log.warning(
                "Dropping %d flow(s): pcap header timestamp unreadable "
                "(reason: %s)", n, REASON_NO_TIMESTAMP,
            )
        stats.n_dropped_unprocessable = n
        stats.dropped_reasons[REASON_NO_TIMESTAMP] = n
        return rows, stats

    for conv in conv_dicts:
        abs_ts = pcap_start_epoch + conv["rel_start"]
        ep_a, ep_b = conv["ep_a"], conv["ep_b"]
        key = orientation_key(ep_a[0], ep_a[1], ep_b[0], ep_b[1])

        try:
            label_row = _match_label(conv, abs_ts, key, label_index)

            if label_row is not None and label_row.get("label") == 1:
                # ── ATTACK ────────────────────────────────────────────────
                prefer_src = label_row["srcip"]
                attack_cat = label_row["attack_cat"]
                label_val  = 1
                matched_attack_row_indices.add(label_row["_row_idx"])
            else:
                # ── BENIGN ────────────────────────────────────────────────
                # No match, OR matched a label=0 row in the GT CSV.
                # Either way: normal traffic, retain with label=0.
                # This is the path that was incorrectly "continue" before —
                # no-match flows are NOT dropped.
                prefer_src = None
                attack_cat = ""
                label_val  = 0

            flow = build_flows([conv], flag_counts, prefer_src=prefer_src)[0]
            row = flow_to_row(flow)
            row["ts"]         = abs_ts
            row["label"]      = label_val
            row["attack_cat"] = attack_cat
            rows.append(row)

            if label_val == 1:
                stats.n_attack += 1
            else:
                stats.n_benign += 1

        except Exception as exc:
            log.warning(
                "Error assembling conv %s<->%s: %s (reason: %s)",
                conv.get("ep_a"), conv.get("ep_b"), exc, REASON_OTHER,
            )
            stats.n_dropped_unprocessable += 1
            stats.dropped_reasons[REASON_OTHER] = (
                stats.dropped_reasons.get(REASON_OTHER, 0) + 1
            )

    return rows, stats


# ── CIC-IDS2017 label matching ────────────────────────────────────────────

def _match_cic_window(
    conv: dict,
    abs_ts: float,
    cic_windows: list[dict],
) -> dict | None:
    """Return the first CIC attack window matching this conv, or None.

    Brute-force attacks use random source ports, so orientation_key lookup is
    impossible.  Match on:
      1. proto == window["proto"]
      2. abs_ts in [window["stime"] - TOL, window["ltime"] + TOL]
      3. one endpoint IP == attacker_ip AND the other == victim_ip
      4. victim-side port == window["dst_port"]
    """
    ep_a_ip, ep_a_port = conv["ep_a"]
    ep_b_ip, ep_b_port = conv["ep_b"]
    proto = conv["proto"]

    for w in cic_windows:
        if proto != w["proto"]:
            continue
        if not (w["stime"] - TIME_TOLERANCE <= abs_ts <= w["ltime"] + TIME_TOLERANCE):
            continue
        # ep_a=attacker, ep_b=victim → victim port matches dst_port
        if (ep_a_ip == w["attacker_ip"] and ep_b_ip == w["victim_ip"]
                and ep_b_port == w["dst_port"]):
            return w
        # ep_b=attacker, ep_a=victim (canonical order swapped)
        if (ep_b_ip == w["attacker_ip"] and ep_a_ip == w["victim_ip"]
                and ep_a_port == w["dst_port"]):
            return w
    return None


def _apply_labels_cic(
    conv_dicts: list[dict],
    flag_counts: dict,
    pcap_start_epoch: float | None,
    cic_windows: list[dict],
    matched_window_cats: set[str],
) -> tuple[list[dict], CorpusStats]:
    """Label CIC-IDS2017 flows by IP+dst_port+time-window matching.

    Unlike UNSW (orientation_key lookup), brute-force traffic has random source
    ports, so we match per conv via _match_cic_window().  Flows that match no
    window are retained as benign (label=0) — never dropped.

    pcap_start_epoch=None → drop all flows as no_timestamp (same as _apply_labels).
    matched_window_cats : set mutated in place — attack_cat of each matched window.
    """
    stats = CorpusStats()
    rows: list[dict] = []

    if pcap_start_epoch is None:
        n = len(conv_dicts)
        if n:
            log.warning(
                "Dropping %d flow(s): pcap header timestamp unreadable "
                "(reason: %s)", n, REASON_NO_TIMESTAMP,
            )
        stats.n_dropped_unprocessable = n
        stats.dropped_reasons[REASON_NO_TIMESTAMP] = n
        return rows, stats

    for conv in conv_dicts:
        abs_ts = pcap_start_epoch + conv["rel_start"]

        try:
            window = _match_cic_window(conv, abs_ts, cic_windows)

            if window is not None:
                prefer_src = window["attacker_ip"]
                attack_cat = window["attack_cat"]
                label_val  = 1
                matched_window_cats.add(attack_cat)
            else:
                prefer_src = None
                attack_cat = ""
                label_val  = 0

            flow = build_flows([conv], flag_counts, prefer_src=prefer_src)[0]
            row  = flow_to_row(flow)
            row["ts"]         = abs_ts
            row["label"]      = label_val
            row["attack_cat"] = attack_cat
            rows.append(row)

            if label_val == 1:
                stats.n_attack += 1
            else:
                stats.n_benign += 1

        except Exception as exc:
            log.warning(
                "Error assembling conv %s<->%s: %s (reason: %s)",
                conv.get("ep_a"), conv.get("ep_b"), exc, REASON_OTHER,
            )
            stats.n_dropped_unprocessable += 1
            stats.dropped_reasons[REASON_OTHER] = (
                stats.dropped_reasons.get(REASON_OTHER, 0) + 1
            )

    return rows, stats


# ── NFStream: protocol number → label string ──────────────────────────────────
# Used by NFStream labeling functions to match GT CSV proto strings ("tcp"/"udp").
_NF_PROTO_STR: dict[int, str] = {6: "TCP", 17: "UDP", 1: "ICMP"}


def _reorient_flow(flow: _Flow, prefer_src: str | None) -> _Flow:
    """Return flow re-oriented so prefer_src is canonical src.

    If prefer_src matches flow.dst_ip, swap all directional fields and IPs/ports.
    If prefer_src is None, already matches flow.src_ip, or matches neither endpoint,
    return the flow unchanged.
    """
    if prefer_src is None or flow.src_ip == prefer_src:
        return flow
    if flow.dst_ip == prefer_src:
        return _Flow(
            ts=flow.ts,
            src_ip=flow.dst_ip,     dst_ip=flow.src_ip,
            src_port=flow.dst_port, dst_port=flow.src_port,
            proto=flow.proto,       duration=flow.duration,
            src_bytes=flow.dst_bytes, dst_bytes=flow.src_bytes,
            src_pkts=flow.dst_pkts,   dst_pkts=flow.src_pkts,
            syn_count=flow.syn_count, ack_count=flow.ack_count,
            rst_count=flow.rst_count, fin_count=flow.fin_count,
            psh_count=flow.psh_count, urg_count=flow.urg_count,
        )
    return flow   # prefer_src not in flow → keep default orientation


# ── NFStream three-way labeling: UNSW time-window ────────────────────────────

def _apply_labels_nf(
    flows: list,
    label_index: dict,
    matched_attack_row_indices: set[int],
) -> tuple[list[dict], CorpusStats]:
    """Label NFStream flows against the UNSW label index (time-window matching).

    flow.ts is absolute epoch seconds (from PCAP packets); no pcap_start_epoch needed.

    Drop reasons reachable on this path: 'other' (assembly exception).
    'no_timestamp' and 'flags_unextractable' are unreachable (NFStream is single-pass
    and reads absolute timestamps directly).

    Three-way outcomes:
      ATTACK  — flow.ts in GT [stime, ltime] ±TOL, endpoint+proto match → label=1,
                re-oriented so attacker is canonical src.
      BENIGN  — no match → label=0 (retained, never dropped)
      DROPPED — assembly exception → reason='other'
    """
    stats = CorpusStats()
    rows: list[dict] = []

    for flow in flows:
        key       = orientation_key(flow.src_ip, flow.src_port, flow.dst_ip, flow.dst_port)
        conv_like = {"proto": _NF_PROTO_STR.get(flow.proto, str(flow.proto))}

        try:
            label_row = _match_label(conv_like, flow.ts, key, label_index)

            if label_row is not None and label_row.get("label") == 1:
                prefer_src = label_row["srcip"]
                attack_cat = label_row["attack_cat"]
                label_val  = 1
                matched_attack_row_indices.add(label_row["_row_idx"])
                flow = _reorient_flow(flow, prefer_src)
            else:
                attack_cat = ""
                label_val  = 0

            row = flow_to_row(flow)
            row["ts"]         = flow.ts
            row["label"]      = label_val
            row["attack_cat"] = attack_cat
            rows.append(row)

            if label_val == 1:
                stats.n_attack += 1
            else:
                stats.n_benign += 1

        except Exception as exc:
            log.warning(
                "Error labeling NFStream flow %s:%s<->%s:%s: %s (reason: %s)",
                flow.src_ip, flow.src_port, flow.dst_ip, flow.dst_port,
                exc, REASON_OTHER,
            )
            stats.n_dropped_unprocessable += 1
            stats.dropped_reasons[REASON_OTHER] = (
                stats.dropped_reasons.get(REASON_OTHER, 0) + 1
            )

    return rows, stats


# ── NFStream three-way labeling: Gotham PCAP-level ───────────────────────────

def _apply_labels_gotham_nf(
    flows: list,
    is_attack: bool,
    attack_cat: str,
    attacker_ips: list[str],
) -> tuple[list[dict], CorpusStats]:
    """Label NFStream flows for Gotham (PCAP-level, no time-window matching).

    For each attack flow, the first matching attacker IP in attacker_ips is used
    to pin canonical src via _reorient_flow().  Falls back to default orientation
    when no attacker IP appears in the flow (mirrors tshark's per-flow IP lookup).

    Drop reasons reachable: 'other' only.
    """
    stats = CorpusStats()
    rows: list[dict] = []

    for flow in flows:
        try:
            if is_attack:
                matching_attacker = next(
                    (ip for ip in attacker_ips if ip in (flow.src_ip, flow.dst_ip)),
                    None,
                )
                flow      = _reorient_flow(flow, matching_attacker)
                label_val = 1
                cat       = attack_cat
            else:
                label_val = 0
                cat       = ""

            row = flow_to_row(flow)
            row["ts"]         = flow.ts
            row["label"]      = label_val
            row["attack_cat"] = cat
            rows.append(row)

            if label_val == 1:
                stats.n_attack += 1
            else:
                stats.n_benign += 1

        except Exception as exc:
            log.warning(
                "Error labeling NFStream Gotham flow: %s (reason: %s)",
                exc, REASON_OTHER,
            )
            stats.n_dropped_unprocessable += 1
            stats.dropped_reasons[REASON_OTHER] = (
                stats.dropped_reasons.get(REASON_OTHER, 0) + 1
            )

    return rows, stats


# ── NFStream three-way labeling: CIC IP+port+time window ─────────────────────

def _match_cic_window_nf(flow: _Flow, cic_windows: list[dict]) -> dict | None:
    """Match a NFStream Flow against CIC attack windows by IP+port+time.

    Checks both canonical orderings since canonical src/dst may not align with
    attacker/victim (brute-force attacks use random source ports).
    """
    proto_str = _NF_PROTO_STR.get(flow.proto, str(flow.proto))
    for w in cic_windows:
        if proto_str != w["proto"]:
            continue
        if not (w["stime"] - TIME_TOLERANCE <= flow.ts <= w["ltime"] + TIME_TOLERANCE):
            continue
        if (flow.src_ip == w["attacker_ip"] and flow.dst_ip == w["victim_ip"]
                and flow.dst_port == w["dst_port"]):
            return w
        if (flow.dst_ip == w["attacker_ip"] and flow.src_ip == w["victim_ip"]
                and flow.src_port == w["dst_port"]):
            return w
    return None


def _apply_labels_cic_nf(
    flows: list,
    cic_windows: list[dict],
    matched_window_cats: set[str],
) -> tuple[list[dict], CorpusStats]:
    """Label NFStream flows for CIC-IDS2017 by IP+port+time-window matching.

    Drop reasons reachable: 'other' only.
    """
    stats = CorpusStats()
    rows: list[dict] = []

    for flow in flows:
        try:
            window = _match_cic_window_nf(flow, cic_windows)

            if window is not None:
                flow      = _reorient_flow(flow, window["attacker_ip"])
                attack_cat = window["attack_cat"]
                label_val  = 1
                matched_window_cats.add(attack_cat)
            else:
                attack_cat = ""
                label_val  = 0

            row = flow_to_row(flow)
            row["ts"]         = flow.ts
            row["label"]      = label_val
            row["attack_cat"] = attack_cat
            rows.append(row)

            if label_val == 1:
                stats.n_attack += 1
            else:
                stats.n_benign += 1

        except Exception as exc:
            log.warning(
                "Error labeling NFStream CIC flow: %s (reason: %s)",
                exc, REASON_OTHER,
            )
            stats.n_dropped_unprocessable += 1
            stats.dropped_reasons[REASON_OTHER] = (
                stats.dropped_reasons.get(REASON_OTHER, 0) + 1
            )

    return rows, stats


# ── class-balance gate ─────────────────────────────────────────────────────

def assert_sane_balance(
    n_attack: int,
    n_benign: int,
    *,
    min_benign_frac: float = 0.50,
    min_attack_frac: float = 0.001,
) -> None:
    """Raise CorpusBalanceError if the corpus is implausibly skewed.

    Two failure modes it catches:

    benign_frac < min_benign_frac (default 0.50):
        The corpus is mostly attacks.  This almost always means the old
        "drop unmatched" bug — unmatched benign flows were deleted, leaving
        only the matched attacks.  A near-100%-attack corpus trains a
        classifier that trivially fires on everything.

    attack_frac < min_attack_frac (default 0.001):
        Almost no attacks in the corpus.  This almost always means that
        time-matching failed — attack rows exist in the GT CSV but none
        lined up temporally with extracted flows.  Check the epoch
        reconstruction with --probe-attack.

    Override both checks with allow_skewed=True in build_corpus() for
    genuinely unusual pcaps.
    """
    total = n_attack + n_benign
    if total == 0:
        raise CorpusBalanceError(
            "Corpus is empty — no flows were labeled. "
            "Verify the pcap directory and GT CSV are correct."
        )
    benign_frac = n_benign / total
    attack_frac = n_attack / total

    if benign_frac < min_benign_frac:
        raise CorpusBalanceError(
            f"Corpus is {100 * attack_frac:.1f}% attack ({n_attack:,} flows); "
            f"benign fraction {100 * benign_frac:.1f}% is below threshold "
            f"{100 * min_benign_frac:.0f}%. "
            "Diagnosis: unmatched flows were almost certainly dropped incorrectly — "
            "benign traffic with no GT label row was deleted instead of retained as "
            "label=0. "
            "Fix: ensure no-match flows are assembled with prefer_src=None and label=0. "
            "Override with allow_skewed=True only if this pcap is genuinely attack-heavy."
        )
    if attack_frac < min_attack_frac:
        raise CorpusBalanceError(
            f"Corpus is {100 * attack_frac:.4f}% attack ({n_attack:,} flows); "
            f"attack fraction is below threshold {100 * min_attack_frac:.3f}%. "
            "Diagnosis: time-matching almost certainly failed — attack rows exist in "
            "the GT CSV but none aligned temporally with the extracted flows. "
            "Check: (1) pcap epoch reconstruction accuracy with --probe-attack; "
            "(2) timezone or units mismatch in the GT CSV stime/ltime columns; "
            "(3) whether the correct pcap files correspond to the given label CSV."
        )


def assert_drop_rate(
    n_dropped: int,
    total_seen: int,
    dropped_reasons: dict[str, int],
    *,
    max_drop_frac: float = MAX_DROP_FRAC,
) -> None:
    """Raise CorpusDropRateError if n_dropped / total_seen > max_drop_frac.

    total_seen = n_attack + n_benign + n_dropped (every flow the pipeline saw).
    A rate above max_drop_frac almost always indicates an extraction bug
    (e.g. an unrecognised pcap magic in get_pcap_start_epoch silently returning
    None) rather than genuine cleaning.  Fix the root cause; never raise the
    threshold as a workaround.

    Safe on an empty pipeline run (total_seen == 0 → no check performed;
    the balance gate catches empty corpora separately).
    """
    if total_seen == 0:
        return
    drop_frac = n_dropped / total_seen
    if drop_frac > max_drop_frac:
        breakdown = "  ".join(
            f"{k}={v:,}" for k, v in sorted(dropped_reasons.items())
        )
        raise CorpusDropRateError(
            f"Drop rate {100 * drop_frac:.1f}% "
            f"({n_dropped:,} dropped / {total_seen:,} total flows seen) "
            f"exceeds {100 * max_drop_frac:.0f}% threshold. "
            f"Reason breakdown: [{breakdown}]. "
            "A high drop rate almost always indicates an extraction bug — "
            "fix the underlying cause rather than raising the threshold. "
            "Common causes: unrecognised pcap magic in get_pcap_start_epoch, "
            "tshark version mismatch, or wrong pcap directory."
        )


# ── main pipeline ──────────────────────────────────────────────────────────

def build_corpus(
    pcap_dir: str | Path,
    label_csv: str | Path,
    tshark_bin: str | None,
    out_parquet: str | Path,
    *,
    allow_skewed: bool = False,
    flood_cap: int = DEFAULT_FLOOD_CAP,
    extractor: str = "nfstream",
    n_meters: int = 1,
) -> tuple[pd.DataFrame, CorpusStats]:
    """Extract, label, and save a training corpus from UNSW-NB15 pcaps.

    Parameters
    ----------
    pcap_dir    : directory containing .pcap / .pcapng files
    label_csv   : UNSW-NB15 ground-truth CSV path
    tshark_bin  : absolute path to the tshark binary
    out_parquet : output path for the labeled parquet corpus
    allow_skewed: if True, skip the class-balance gate (prints WARNING, proceeds)
    flood_cap   : per-source-IP cap on degenerate one-sided flood flows
                  (src_pkts<=1, dst_pkts==0, label=1).  0 = disable.
                  Default DEFAULT_FLOOD_CAP.  Must match Gotham v2 value.

    Returns (labeled_DataFrame, CorpusStats).

    Raises CorpusDropRateError BEFORE writing the parquet if >MAX_DROP_FRAC of
    all flows seen were silently dropped (extraction bug signal).

    Raises CorpusBalanceError BEFORE writing the parquet if the corpus is
    implausibly skewed (nearly all attack, or nearly no attack), unless
    allow_skewed=True.

    Stats logged at INFO:
      n_attack / n_benign / n_dropped: three-way labeling counts
      label_rows_total / matched / unmatched: GT CSV accounting

    extractor: 'nfstream' (default) or 'tshark'.
        nfstream — single-pass via extract_flows_nfstream(); tshark not required.
        tshark   — two-pass (run_pass_a + run_pass_b + chunked fallback); requires tshark_bin.
    n_meters: NFStream worker count (default 1; corpus builds may use higher for speed).

    Construction parity with Gotham v2
    -----------------------------------
    NFStream: extract_flows_nfstream() (single-pass, canonical config)
    tshark:   run_pass_a (window_sec=570; timeout=600s)
              + run_pass_b → TimeoutExpired → run_pass_b_chunked → quarantine
    Flood cap: DEFAULT_FLOOD_CAP per source IP (src_pkts<=1, dst_pkts==0, label=1)
    Drop gate: assert_drop_rate before writing parquet
    Balance gate: assert_sane_balance before writing parquet
    """
    if extractor == "tshark" and not tshark_bin:
        raise ValueError("tshark_bin is required when extractor='tshark'")
    pcap_dir    = Path(pcap_dir)
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading label index from %s", label_csv)
    label_index, total_attack_rows = load_label_index(label_csv)
    log.info(
        "Label index: %d unique endpoint pairs, %d attack rows (label=1)",
        len(label_index), total_attack_rows,
    )

    pcap_files = sorted(
        p for p in pcap_dir.iterdir()
        if p.suffix.lower() in (".pcap", ".pcapng")
    )
    if not pcap_files:
        raise FileNotFoundError(f"No pcap/pcapng files found in {pcap_dir}")
    log.info("Found %d pcap file(s)", len(pcap_files))

    all_rows:   list[dict]   = []
    total_stats              = CorpusStats(label_rows_total=total_attack_rows)
    matched_attack_row_idx:  set[int] = set()

    for pcap_path in pcap_files:
        log.info("Processing %s", pcap_path.name)

        if extractor == "nfstream":
            # ── NFStream single-pass extraction ────────────────────────────
            try:
                flows = extract_flows_nfstream(str(pcap_path), n_meters=n_meters)
            except Exception as exc:
                log.warning(
                    "NFStream extraction failed on %s: %s — dropping PCAP",
                    pcap_path.name, exc,
                )
                total_stats.n_dropped_unprocessable += 1
                total_stats.dropped_reasons[REASON_EXTRACTION_FAIL] = (
                    total_stats.dropped_reasons.get(REASON_EXTRACTION_FAIL, 0) + 1
                )
                continue
            batch_rows, batch_stats = _apply_labels_nf(
                flows, label_index, matched_attack_row_idx,
            )

        else:
            # ── tshark two-pass extraction (kept until Phase 6) ────────────
            pcap_start = get_pcap_start_epoch(pcap_path)
            if pcap_start is None:
                log.warning(
                    "%s: header timestamp unreadable — flows will be dropped",
                    pcap_path.name,
                )

            try:
                conv_dicts = run_pass_a(
                    tshark_bin, pcap=str(pcap_path), window_sec=570,
                )
            except Exception as exc:
                log.warning(
                    "Pass A failed on %s: %s — dropping PCAP",
                    pcap_path.name, exc,
                )
                total_stats.n_dropped_unprocessable += 1
                total_stats.dropped_reasons[REASON_EXTRACTION_FAIL] = (
                    total_stats.dropped_reasons.get(REASON_EXTRACTION_FAIL, 0) + 1
                )
                continue

            # NEVER silently zero-fill.  Fabricated zeros corrupt flag features.
            flags_ok = True
            try:
                flag_counts = run_pass_b(tshark_bin, pcap=str(pcap_path))
            except _sp.TimeoutExpired:
                size_mb = pcap_path.stat().st_size / 1_048_576
                log.warning(
                    "%s: pass B timed out (%.0f MB) — trying chunked pass B",
                    pcap_path.name, size_mb,
                )
                try:
                    flag_counts = run_pass_b_chunked(tshark_bin, pcap_path)
                    log.info("  chunked pass B completed for %s", pcap_path.name)
                except Exception as chunk_exc:
                    log.warning(
                        "%s: chunked pass B also failed (%s) — quarantining "
                        "all flows as flags_unextractable",
                        pcap_path.name, chunk_exc,
                    )
                    flags_ok = False
            except Exception as exc:
                log.warning(
                    "%s: pass B failed (%s) — quarantining flows",
                    pcap_path.name, exc,
                )
                flags_ok = False

            if not flags_ok:
                n_q = len(conv_dicts)
                total_stats.n_dropped_unprocessable += n_q
                total_stats.dropped_reasons[REASON_FLAGS_UNEXTRACTABLE] = (
                    total_stats.dropped_reasons.get(REASON_FLAGS_UNEXTRACTABLE, 0) + n_q
                )
                continue

            batch_rows, batch_stats = _apply_labels(
                conv_dicts, flag_counts, pcap_start, label_index, matched_attack_row_idx,
            )

        all_rows.extend(batch_rows)
        total_stats.merge(batch_stats)

    total_stats.label_rows_matched = len(matched_attack_row_idx)

    # ── label-row accounting warning ───────────────────────────────────────
    if total_attack_rows > 0:
        unmatched_frac = total_stats.label_rows_unmatched / total_attack_rows
        if unmatched_frac > UNMATCHED_WARN_FRAC:
            log.warning(
                "HIGH UNMATCHED RATE: %d of %d attack rows (%.0f%%) matched no flow. "
                "Most likely cause: timezone/epoch offset in GT CSV timestamps or "
                "wrong pcap-to-labelfile pairing. "
                "Run: python -m corpus.build_corpus --probe-attack <ROW_IDX> <PCAP> <GT_CSV>",
                total_stats.label_rows_unmatched,
                total_attack_rows,
                100 * unmatched_frac,
            )

    log.info(
        "Corpus stats — n_attack=%d  n_benign=%d  n_dropped=%d  "
        "label_rows: total=%d  matched=%d  unmatched=%d  dropped_reasons=%s",
        total_stats.n_attack, total_stats.n_benign, total_stats.n_dropped_unprocessable,
        total_stats.label_rows_total, total_stats.label_rows_matched,
        total_stats.label_rows_unmatched, total_stats.dropped_reasons,
    )

    # ── drop-rate gate (before writing parquet) ────────────────────────────
    total_seen = total_stats.total_kept + total_stats.n_dropped_unprocessable
    assert_drop_rate(
        total_stats.n_dropped_unprocessable, total_seen, total_stats.dropped_reasons,
    )

    # ── class-balance gate (before writing parquet) ────────────────────────
    if not allow_skewed:
        assert_sane_balance(total_stats.n_attack, total_stats.n_benign)
    else:
        log.warning(
            "ALLOW_SKEWED: class-balance check bypassed. "
            "n_attack=%d (%.1f%%)  n_benign=%d (%.1f%%)  Proceed with caution.",
            total_stats.n_attack, 100 * total_stats.attack_frac,
            total_stats.n_benign, 100 * total_stats.benign_frac,
        )

    if not all_rows:
        log.warning("No flows produced — corpus is empty")
        df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        df = pd.DataFrame(all_rows)[OUTPUT_COLUMNS]
        df = df.sort_values(
            ["ts", "src_ip", "src_port", "dst_ip", "dst_port"]
        ).reset_index(drop=True)

    # ── Flood cap (same as Gotham v2: DEFAULT_FLOOD_CAP/src_ip) ───────────
    if flood_cap > 0 and len(df) > 0:
        n_before = len(df)
        n_att_before = int((df["label"] == 1).sum())
        df, n_flood_dropped = apply_flood_cap(df, cap=flood_cap)
        n_att_after = int((df["label"] == 1).sum())
        log.info(
            "Flood cap (N=%d/src_ip): dropped %d degenerate flows  "
            "attack before=%d after=%d  "
            "total before=%d after=%d",
            flood_cap, n_flood_dropped,
            n_att_before, n_att_after,
            n_before, len(df),
        )
        total_stats.n_attack = n_att_after
        total_stats.n_benign = int((df["label"] == 0).sum())

    df.to_parquet(out_parquet, index=False)
    log.info("Wrote %d rows to %s", len(df), out_parquet)
    return df, total_stats


# ── Gotham labeling (PCAP-level, no time-window matching) ─────────────────

def _apply_labels_gotham(
    conv_dicts: list[dict],
    flag_counts: dict,
    pcap_start_epoch: float | None,
    is_attack: bool,
    attack_cat: str,
    attacker_ips: list[str],
) -> tuple[list[dict], CorpusStats]:
    """Label Gotham flows by PCAP-level label (directory-based, no time matching).

    For benign PCAPs:  every flow → label=0, prefer_src=None.
    For attack PCAPs:  every flow → label=1, attack_cat=<cat>,
                       prefer_src=first attacker IP matching one endpoint.

    pcap_start_epoch=None drops all flows as no_timestamp — same behaviour
    as _apply_labels() for consistent DROPPED accounting.
    """
    stats = CorpusStats()
    rows: list[dict] = []

    if pcap_start_epoch is None:
        n = len(conv_dicts)
        if n:
            log.warning(
                "Dropping %d flow(s): pcap header timestamp unreadable "
                "(reason: %s)", n, REASON_NO_TIMESTAMP,
            )
        stats.n_dropped_unprocessable = n
        stats.dropped_reasons[REASON_NO_TIMESTAMP] = n
        return rows, stats

    for conv in conv_dicts:
        abs_ts = pcap_start_epoch + conv["rel_start"]

        try:
            if is_attack:
                ep_a_ip = conv["ep_a"][0]
                ep_b_ip = conv["ep_b"][0]
                prefer_src = next(
                    (ip for ip in attacker_ips if ip in (ep_a_ip, ep_b_ip)),
                    None,   # fall back to default orientation if no attacker IP matches
                )
                label_val  = 1
                cat        = attack_cat
            else:
                prefer_src = None
                label_val  = 0
                cat        = ""

            flow = build_flows([conv], flag_counts, prefer_src=prefer_src)[0]
            row  = flow_to_row(flow)
            row["ts"]         = abs_ts
            row["label"]      = label_val
            row["attack_cat"] = cat
            rows.append(row)

            if label_val == 1:
                stats.n_attack += 1
            else:
                stats.n_benign += 1

        except Exception as exc:
            log.warning(
                "Error assembling conv %s<->%s: %s (reason: %s)",
                conv.get("ep_a"), conv.get("ep_b"), exc, REASON_OTHER,
            )
            stats.n_dropped_unprocessable += 1
            stats.dropped_reasons[REASON_OTHER] = (
                stats.dropped_reasons.get(REASON_OTHER, 0) + 1
            )

    return rows, stats


def build_corpus_gotham(
    gotham_root: str | Path,
    tshark_bin: str | None,
    out_parquet: str | Path,
    *,
    allow_skewed: bool = False,
    flood_cap: int = DEFAULT_FLOOD_CAP,
    extractor: str = "nfstream",
    n_meters: int = 1,
) -> tuple[pd.DataFrame, CorpusStats]:
    """Extract, label, and save a Gotham training corpus.

    Parameters
    ----------
    gotham_root : Gotham dataset root (must contain raw/benign/ and raw/malicious/)
    tshark_bin  : absolute path to the tshark binary
    out_parquet : output path for the labeled parquet corpus
    allow_skewed: if True, skip the class-balance gate (useful for single-pcap
                  sanity checks where the PCAP is expectedly single-class)
    flood_cap   : per-source-IP cap on degenerate one-sided flood flows
                  (src_pkts<=1, dst_pkts==0, label=1).  Set to 0 to disable.
                  Default DEFAULT_FLOOD_CAP.

    Label-row accounting in Gotham mode
    ------------------------------------
    label_rows_total   = number of attack PCAPs discovered
    label_rows_matched = attack PCAPs that produced ≥1 attack flow
    label_rows_unmatched = attack PCAPs producing 0 flows (likely tshark failure)

    The class-balance gate applies to the FULL corpus, not per-PCAP.
    Individual benign PCAPs have no attacks (legitimately all-benign);
    individual attack PCAPs have no benign flows (legitimately all-attack).
    Use allow_skewed=True when running a single-pcap sanity check.

    extractor: 'nfstream' (default) or 'tshark'.
    n_meters: NFStream worker count (default 1).
    """
    if extractor == "tshark" and not tshark_bin:
        raise ValueError("tshark_bin is required when extractor='tshark'")
    from corpus.gotham_labels import load_gotham_corpus_spec

    gotham_root = Path(gotham_root)
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    specs = load_gotham_corpus_spec(gotham_root)
    attack_specs = [(i, s) for i, s in enumerate(specs) if s.is_attack]

    log.info(
        "Gotham corpus spec: %d PCAPs total (%d attack, %d benign)",
        len(specs), len(attack_specs), len(specs) - len(attack_specs),
    )

    all_rows:   list[dict] = []
    total_stats = CorpusStats(label_rows_total=len(attack_specs))
    attack_pcaps_with_flows: set[int] = set()   # spec indices of attack PCAPs with ≥1 flow

    for spec_idx, spec in enumerate(specs):
        log.info(
            "Processing %s  [%s]",
            spec.pcap_path.name,
            spec.attack_cat if spec.is_attack else "benign",
        )

        if extractor == "nfstream":
            # ── NFStream single-pass extraction ────────────────────────────
            try:
                flows = extract_flows_nfstream(str(spec.pcap_path), n_meters=n_meters)
            except Exception as exc:
                log.warning(
                    "NFStream extraction failed on %s: %s — dropping PCAP",
                    spec.pcap_path.name, exc,
                )
                total_stats.n_dropped_unprocessable += 1
                total_stats.dropped_reasons[REASON_EXTRACTION_FAIL] = (
                    total_stats.dropped_reasons.get(REASON_EXTRACTION_FAIL, 0) + 1
                )
                continue
            batch_rows, batch_stats = _apply_labels_gotham_nf(
                flows, spec.is_attack, spec.attack_cat, spec.attacker_ips,
            )

        else:
            # ── tshark two-pass extraction (kept until Phase 6) ────────────
            pcap_start = get_pcap_start_epoch(spec.pcap_path)
            if pcap_start is None:
                log.warning(
                    "%s: header timestamp unreadable — flows will be dropped",
                    spec.pcap_path.name,
                )

            try:
                conv_dicts = run_pass_a(
                    tshark_bin, pcap=str(spec.pcap_path), window_sec=570,
                )
            except Exception as exc:
                log.warning(
                    "Pass A failed on %s: %s — dropping PCAP",
                    spec.pcap_path.name, exc,
                )
                total_stats.n_dropped_unprocessable += 1
                total_stats.dropped_reasons[REASON_EXTRACTION_FAIL] = (
                    total_stats.dropped_reasons.get(REASON_EXTRACTION_FAIL, 0) + 1
                )
                continue

            # NEVER silently zero-fill.  Fabricated zeros corrupt flag features
            # and create a capture-size-correlated leak (large DDoS pcaps would
            # have all flags=0, making them trivially distinguishable from flows
            # where flags were actually observed).  Any PCAP whose TCP flag counts
            # cannot be extracted after chunking is quarantined entirely.
            flags_ok = True
            try:
                flag_counts = run_pass_b(tshark_bin, pcap=str(spec.pcap_path))
            except _sp.TimeoutExpired:
                size_mb = spec.pcap_path.stat().st_size / 1_048_576
                log.warning(
                    "%s: pass B timed out (%.0f MB) — trying chunked pass B",
                    spec.pcap_path.name, size_mb,
                )
                try:
                    flag_counts = run_pass_b_chunked(tshark_bin, spec.pcap_path)
                    log.info("  chunked pass B completed for %s", spec.pcap_path.name)
                except Exception as chunk_exc:
                    log.warning(
                        "%s: chunked pass B also failed (%s) — quarantining "
                        "all flows as flags_unextractable",
                        spec.pcap_path.name, chunk_exc,
                    )
                    flags_ok = False
            except Exception as exc:
                log.warning(
                    "%s: pass B failed (%s) — quarantining flows",
                    spec.pcap_path.name, exc,
                )
                flags_ok = False

            if not flags_ok:
                n_q = len(conv_dicts)
                total_stats.n_dropped_unprocessable += n_q
                total_stats.dropped_reasons[REASON_FLAGS_UNEXTRACTABLE] = (
                    total_stats.dropped_reasons.get(REASON_FLAGS_UNEXTRACTABLE, 0) + n_q
                )
                continue

            batch_rows, batch_stats = _apply_labels_gotham(
                conv_dicts, flag_counts, pcap_start,
                spec.is_attack, spec.attack_cat, spec.attacker_ips,
            )

        all_rows.extend(batch_rows)
        total_stats.merge(batch_stats)

        if spec.is_attack and batch_stats.n_attack > 0:
            attack_pcaps_with_flows.add(spec_idx)

    total_stats.label_rows_matched = len(attack_pcaps_with_flows)

    if len(attack_specs) > 0:
        unmatched_frac = total_stats.label_rows_unmatched / len(attack_specs)
        if unmatched_frac > UNMATCHED_WARN_FRAC:
            log.warning(
                "HIGH UNMATCHED RATE: %d of %d attack PCAPs produced 0 attack flows. "
                "Most likely cause: tshark extraction failure on those files.",
                total_stats.label_rows_unmatched, len(attack_specs),
            )

    log.info(
        "Gotham corpus stats — n_attack=%d  n_benign=%d  n_dropped=%d  "
        "attack_pcaps: total=%d  matched=%d  unmatched=%d  dropped_reasons=%s",
        total_stats.n_attack, total_stats.n_benign, total_stats.n_dropped_unprocessable,
        len(attack_specs), total_stats.label_rows_matched,
        total_stats.label_rows_unmatched, total_stats.dropped_reasons,
    )

    # ── drop-rate gate (before writing parquet) ────────────────────────────
    total_seen = total_stats.total_kept + total_stats.n_dropped_unprocessable
    assert_drop_rate(
        total_stats.n_dropped_unprocessable, total_seen, total_stats.dropped_reasons,
    )

    if not allow_skewed:
        assert_sane_balance(total_stats.n_attack, total_stats.n_benign)
    else:
        log.warning(
            "ALLOW_SKEWED: class-balance check bypassed. "
            "n_attack=%d (%.1f%%)  n_benign=%d (%.1f%%)  Proceed with caution.",
            total_stats.n_attack, 100 * total_stats.attack_frac,
            total_stats.n_benign, 100 * total_stats.benign_frac,
        )

    if not all_rows:
        log.warning("No flows produced — corpus is empty")
        df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        df = pd.DataFrame(all_rows)[OUTPUT_COLUMNS]
        df = df.sort_values(
            ["ts", "src_ip", "src_port", "dst_ip", "dst_port"]
        ).reset_index(drop=True)

    # ── Flood cap ────────────────────────────────────────────────────────────
    if flood_cap > 0 and len(df) > 0:
        n_before = len(df)
        n_att_before = int((df["label"] == 1).sum())
        df, n_flood_dropped = apply_flood_cap(df, cap=flood_cap)
        n_att_after = int((df["label"] == 1).sum())
        log.info(
            "Flood cap (N=%d/src_ip): dropped %d degenerate flows  "
            "attack before=%d after=%d  "
            "total before=%d after=%d",
            flood_cap, n_flood_dropped,
            n_att_before, n_att_after,
            n_before, len(df),
        )
        total_stats.n_attack = n_att_after
        total_stats.n_benign = int((df["label"] == 0).sum())

    df.to_parquet(out_parquet, index=False)
    log.info("Wrote %d rows to %s", len(df), out_parquet)
    return df, total_stats


# ── CIC-IDS2017 pipeline ──────────────────────────────────────────────────

def build_corpus_cic(
    pcap_path: str | Path,
    cic_windows: list[dict],
    tshark_bin: str | None,
    out_parquet: str | Path,
    *,
    allow_skewed: bool = False,
    flood_cap: int = DEFAULT_FLOOD_CAP,
    extractor: str = "nfstream",
    n_meters: int = 1,
) -> tuple[pd.DataFrame, CorpusStats]:
    """Extract, label, and save a CIC-IDS2017 Tuesday corpus.

    Parameters
    ----------
    pcap_path   : path to Tuesday-WorkingHours.pcap (single file, ~11 GB)
    cic_windows : attack window list from corpus.cic_labels.CIC_ATTACK_WINDOWS
    tshark_bin  : absolute path to the tshark binary
    out_parquet : output path for the labeled parquet corpus
    allow_skewed: if True, skip the class-balance gate
    flood_cap   : per-source-IP cap on degenerate one-sided flood flows

    Attack labeling
    ---------------
    CIC brute-force attacks use random source ports, so UNSW-style
    orientation_key lookup is inapplicable.  _apply_labels_cic() matches by:
      proto==TCP AND abs_ts in window AND attacker_ip/victim_ip/dst_port match.
    All other flows → label=0 (benign).

    Construction parity with UNSW v2 and Gotham v2
    -----------------------------------------------
    NFStream: extract_flows_nfstream() (single-pass, canonical config)
    tshark:   run_pass_a(window_sec=570) + run_pass_b → chunked → quarantine
    Flood cap: DEFAULT_FLOOD_CAP per source IP (src_pkts<=1, dst_pkts==0, label=1)
    Drop gate: assert_drop_rate before writing parquet
    Balance gate: assert_sane_balance before writing parquet

    extractor: 'nfstream' (default) or 'tshark'.
    n_meters: NFStream worker count (default 1).
    """
    if extractor == "tshark" and not tshark_bin:
        raise ValueError("tshark_bin is required when extractor='tshark'")

    pcap_path   = Path(pcap_path)
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    if not pcap_path.exists():
        raise FileNotFoundError(f"CIC PCAP not found: {pcap_path}")

    log.info(
        "CIC build: %s  extractor=%s  (%d attack windows)",
        pcap_path.name, extractor, len(cic_windows),
    )

    total_stats = CorpusStats(label_rows_total=len(cic_windows))
    all_rows: list[dict] = []

    if extractor == "nfstream":
        # ── NFStream single-pass ───────────────────────────────────────────
        try:
            flows = extract_flows_nfstream(str(pcap_path), n_meters=n_meters)
        except Exception as exc:
            raise RuntimeError(
                f"NFStream extraction failed on {pcap_path.name}: {exc}"
            ) from exc
        matched_window_cats: set[str] = set()
        batch_rows, batch_stats = _apply_labels_cic_nf(
            flows, cic_windows, matched_window_cats,
        )
        all_rows.extend(batch_rows)
        total_stats.merge(batch_stats)
        total_stats.label_rows_matched = len(matched_window_cats)

    else:
        # ── tshark two-pass (kept until Phase 6) ──────────────────────────
        pcap_start = get_pcap_start_epoch(pcap_path)
        if pcap_start is None:
            log.warning(
                "%s: header timestamp unreadable — all flows will be dropped",
                pcap_path.name,
            )

        try:
            conv_dicts = run_pass_a(tshark_bin, pcap=str(pcap_path), window_sec=570)
        except Exception as exc:
            raise RuntimeError(
                f"Pass A failed on {pcap_path.name}: {exc}"
            ) from exc

        flags_ok = True
        try:
            flag_counts = run_pass_b(tshark_bin, pcap=str(pcap_path))
        except _sp.TimeoutExpired:
            size_mb = pcap_path.stat().st_size / 1_048_576
            log.warning(
                "%s: pass B timed out (%.0f MB) — trying chunked pass B",
                pcap_path.name, size_mb,
            )
            try:
                flag_counts = run_pass_b_chunked(tshark_bin, pcap_path)
                log.info("  chunked pass B completed for %s", pcap_path.name)
            except Exception as chunk_exc:
                log.warning(
                    "%s: chunked pass B also failed (%s) — quarantining flows",
                    pcap_path.name, chunk_exc,
                )
                flags_ok = False
        except Exception as exc:
            log.warning(
                "%s: pass B failed (%s) — quarantining flows",
                pcap_path.name, exc,
            )
            flags_ok = False

        if not flags_ok:
            n_q = len(conv_dicts)
            total_stats.n_dropped_unprocessable += n_q
            total_stats.dropped_reasons[REASON_FLAGS_UNEXTRACTABLE] = n_q
        else:
            matched_window_cats_tsh: set[str] = set()
            batch_rows, batch_stats = _apply_labels_cic(
                conv_dicts, flag_counts, pcap_start, cic_windows, matched_window_cats_tsh,
            )
            all_rows.extend(batch_rows)
            total_stats.merge(batch_stats)
            total_stats.label_rows_matched = len(matched_window_cats_tsh)

    log.info(
        "CIC corpus stats — n_attack=%d  n_benign=%d  n_dropped=%d  "
        "windows: total=%d  matched=%d  dropped_reasons=%s",
        total_stats.n_attack, total_stats.n_benign,
        total_stats.n_dropped_unprocessable,
        len(cic_windows), total_stats.label_rows_matched,
        total_stats.dropped_reasons,
    )

    # ── gates ─────────────────────────────────────────────────────────────────
    total_seen = total_stats.total_kept + total_stats.n_dropped_unprocessable
    assert_drop_rate(
        total_stats.n_dropped_unprocessable, total_seen, total_stats.dropped_reasons,
    )

    if not allow_skewed:
        assert_sane_balance(total_stats.n_attack, total_stats.n_benign)
    else:
        log.warning(
            "ALLOW_SKEWED: class-balance check bypassed. "
            "n_attack=%d (%.1f%%)  n_benign=%d (%.1f%%)  Proceed with caution.",
            total_stats.n_attack, 100 * total_stats.attack_frac,
            total_stats.n_benign, 100 * total_stats.benign_frac,
        )

    if not all_rows:
        log.warning("No flows produced — corpus is empty")
        df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        df = pd.DataFrame(all_rows)[OUTPUT_COLUMNS]
        df = df.sort_values(
            ["ts", "src_ip", "src_port", "dst_ip", "dst_port"]
        ).reset_index(drop=True)

    if flood_cap > 0 and len(df) > 0:
        n_before     = len(df)
        n_att_before = int((df["label"] == 1).sum())
        df, n_flood_dropped = apply_flood_cap(df, cap=flood_cap)
        n_att_after  = int((df["label"] == 1).sum())
        log.info(
            "Flood cap (N=%d/src_ip): dropped %d degenerate flows  "
            "attack before=%d after=%d  total before=%d after=%d",
            flood_cap, n_flood_dropped,
            n_att_before, n_att_after,
            n_before, len(df),
        )
        total_stats.n_attack = n_att_after
        total_stats.n_benign = int((df["label"] == 0).sum())

    df.to_parquet(out_parquet, index=False)
    log.info("Wrote %d rows to %s", len(df), out_parquet)
    return df, total_stats


# ── CLI subcommands ────────────────────────────────────────────────────────

def _cmd_sanity_check(pcap_path: str, label_csv: str, tshark_bin: str) -> None:
    """Run the full pipeline on a single pcap, print stats, PASS/FAIL verdict.

    Does NOT write a parquet.  Run this before committing to a full multi-pcap
    build to verify epoch reconstruction and class balance on one representative
    pcap.
    """
    import tempfile, os

    pcap = Path(pcap_path)
    if not pcap.exists():
        print(f"ERROR: pcap not found: {pcap}")
        sys.exit(1)

    # Wrap the single pcap in a temp dir so build_corpus's pcap_dir logic works.
    with tempfile.TemporaryDirectory() as tmpdir:
        link = Path(tmpdir) / pcap.name
        try:
            os.symlink(pcap.resolve(), link)
        except OSError:
            import shutil
            shutil.copy2(pcap, link)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
            tmp_parquet = tf.name

        try:
            df, stats = build_corpus(
                pcap_dir=tmpdir,
                label_csv=label_csv,
                tshark_bin=tshark_bin,
                out_parquet=tmp_parquet,
                allow_skewed=True,  # balance check is printed separately below
            )
        except Exception as exc:
            print(f"\nERROR during extraction: {exc}")
            os.unlink(tmp_parquet)
            sys.exit(1)
        finally:
            if os.path.exists(tmp_parquet):
                os.unlink(tmp_parquet)

    total = stats.n_attack + stats.n_benign
    benign_pct = 100 * stats.n_benign  / max(total, 1)
    attack_pct = 100 * stats.n_attack  / max(total, 1)

    print("\n=== Sanity-check results ===")
    print(f"  Pcap              : {pcap.name}")
    print(f"  Flows kept        : {total:,}  (attack={stats.n_attack:,}  benign={stats.n_benign:,})")
    print(f"  Attack fraction   : {attack_pct:.2f}%   Benign: {benign_pct:.2f}%")
    print(f"  Dropped           : {stats.n_dropped_unprocessable}  reasons={stats.dropped_reasons}")
    print(f"\n  Label-row accounting:")
    print(f"    GT attack rows  : {stats.label_rows_total:,}")
    print(f"    Matched         : {stats.label_rows_matched:,}")
    print(f"    Unmatched       : {stats.label_rows_unmatched:,}  "
          f"({100*stats.label_rows_unmatched/max(stats.label_rows_total,1):.1f}%)")

    try:
        assert_sane_balance(stats.n_attack, stats.n_benign)
        print("\n  Balance check     : PASS")
    except CorpusBalanceError as e:
        print(f"\n  Balance check     : FAIL\n  {e}")
        sys.exit(1)


def _cmd_sanity_check_gotham(
    pcap_path: str,
    gotham_root: str,
    tshark_bin: str,
) -> None:
    """Run single-pcap sanity check for a Gotham PCAP (no parquet written).

    Detects attack type from the PCAP's parent directory name.  If the parent
    is not a known attack category or 'benign', reports an error.

    Gotham per-device PCAPs are expectedly single-class (all-benign OR
    all-attack), so --allow-skewed is set automatically for the extraction step.
    The balance verdict is printed based on the pcap's actual content.
    """
    import tempfile, os
    from corpus.gotham_labels import ATTACK_CAT_MAP, ATTACKER_IPS, GothamPcapSpec

    pcap = Path(pcap_path)
    if not pcap.exists():
        print(f"ERROR: pcap not found: {pcap}")
        sys.exit(1)

    parent_name = pcap.parent.name
    if parent_name == "benign":
        spec = GothamPcapSpec(
            pcap_path=pcap, is_attack=False, attack_cat="", attacker_ips=[],
        )
    elif parent_name in ATTACK_CAT_MAP:
        spec = GothamPcapSpec(
            pcap_path=pcap,
            is_attack=True,
            attack_cat=ATTACK_CAT_MAP[parent_name],
            attacker_ips=ATTACKER_IPS.get(parent_name, []),
        )
    else:
        print(
            f"ERROR: cannot determine label from parent dir '{parent_name}'. "
            f"Expected 'benign' or one of: {sorted(ATTACK_CAT_MAP)}"
        )
        sys.exit(1)

    print(f"\n=== Gotham sanity-check: {pcap.name} ===")
    print(f"  Detected label : {'attack / ' + spec.attack_cat if spec.is_attack else 'benign'}")
    print(f"  Attacker IPs   : {spec.attacker_ips or 'N/A (benign)'}")

    pcap_start = get_pcap_start_epoch(pcap)
    if pcap_start is None:
        print("WARNING: pcap header timestamp unreadable — all flows will be dropped")
    else:
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(pcap_start, tz=timezone.utc)
        print(f"  Pcap epoch     : {pcap_start:.3f}  ({dt.isoformat()})")

    try:
        conv_dicts, flag_counts = extract_pcap_flows(pcap, tshark_bin)
    except Exception as exc:
        print(f"\nERROR: tshark extraction failed: {exc}")
        sys.exit(1)

    batch_rows, stats = _apply_labels_gotham(
        conv_dicts, flag_counts, pcap_start,
        spec.is_attack, spec.attack_cat, spec.attacker_ips,
    )

    total = stats.n_attack + stats.n_benign
    benign_pct = 100 * stats.n_benign / max(total, 1)
    attack_pct = 100 * stats.n_attack / max(total, 1)

    print(f"\n=== Sanity-check results ===")
    print(f"  Pcap              : {pcap.name}")
    print(f"  Flows kept        : {total:,}  (attack={stats.n_attack:,}  benign={stats.n_benign:,})")
    print(f"  Attack fraction   : {attack_pct:.2f}%   Benign: {benign_pct:.2f}%")
    print(f"  Dropped           : {stats.n_dropped_unprocessable}  reasons={stats.dropped_reasons}")

    if spec.is_attack:
        print(f"\n  Label source      : PCAP-level (all flows -> attack/{spec.attack_cat})")
        print(f"  Attacker IPs hit  : ", end="")
        if batch_rows:
            srcs = {r["src_ip"] for r in batch_rows}
            print(", ".join(sorted(srcs & set(spec.attacker_ips))) or "(none -- default orientation used)")
        else:
            print("N/A (no flows)")
    else:
        print(f"\n  Label source      : PCAP-level (all flows -> benign)")

    print(f"\n  Note: single Gotham PCAPs are expectedly single-class.")
    note = "PASS (attack pcap is legitimately 100% attack)" if spec.is_attack else "PASS (benign pcap is legitimately 100% benign)"
    if total == 0:
        print("  Balance verdict   : WARN — no flows extracted")
    else:
        print(f"  Balance verdict   : {note}")


def _cmd_probe_cic(pcap_path: str, tshark_bin: str) -> None:
    """Print CIC-IDS2017 timezone probe: PCAP epoch vs expected attack windows.

    Reads the PCAP header and compares the start epoch to the hardcoded value in
    cic_labels.py.  Prints each attack window in both UTC and ADT so the caller
    can verify alignment with the published schedule without re-running io,stat.
    """
    from corpus.cic_labels import CIC_ATTACK_WINDOWS, PCAP_START_EPOCH
    from datetime import datetime, timezone, timedelta

    pcap_start = get_pcap_start_epoch(pcap_path)
    if pcap_start is None:
        print(f"ERROR: cannot read pcap header timestamp from {pcap_path}")
        sys.exit(1)

    ADT = timezone(timedelta(hours=-3))
    utc = timezone.utc

    print(f"\n=== CIC-IDS2017 Tuesday timezone probe ===")
    print(f"  PCAP         : {pcap_path}")
    print(f"  Start epoch  : {pcap_start:.3f}")
    print(
        f"  Start (UTC)  : "
        f"{datetime.fromtimestamp(pcap_start, tz=utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    print(
        f"  Start (ADT)  : "
        f"{datetime.fromtimestamp(pcap_start, tz=ADT).strftime('%Y-%m-%d %H:%M:%S ADT')}"
    )
    print(f"  Hardcoded    : {PCAP_START_EPOCH}")

    delta = pcap_start - PCAP_START_EPOCH
    if abs(delta) > 1.0:
        print(
            f"  WARNING: epoch differs from hardcoded by {delta:+.3f}s — "
            "update PCAP_START_EPOCH in cic_labels.py"
        )
    else:
        print(f"  Epoch match  : OK (delta={delta:+.3f}s)")

    print(f"\n  Attack windows (derived from PCAP io,stat analysis):")
    for w in CIC_ATTACK_WINDOWS:
        ws = datetime.fromtimestamp(w["stime"], tz=utc)
        we = datetime.fromtimestamp(w["ltime"],   tz=utc)
        as_ = datetime.fromtimestamp(w["stime"], tz=ADT)
        ae  = datetime.fromtimestamp(w["ltime"],   tz=ADT)
        print(f"    {w['attack_cat']}")
        print(f"      UTC  : {ws.strftime('%H:%M')}–{we.strftime('%H:%M')}")
        print(f"      ADT  : {as_.strftime('%H:%M')}–{ae.strftime('%H:%M')}")
        print(f"      {w['attacker_ip']} → {w['victim_ip']}:{w['dst_port']}/{w['proto']}")

    print(
        f"\n  Verdict: PASS — timezone confirmed ADT (UTC-3) from io,stat burst analysis"
    )


def _cmd_sanity_check_cic(pcap_path: str, tshark_bin: str) -> None:
    """Run the CIC build pipeline on a single PCAP (no parquet written).

    Suitable for passing a pre-sliced attack-window file (e.g. cic_ssh_window.pcap)
    rather than the full 11 GB PCAP.  Prints n_attack / n_benign / balance verdict.
    """
    import os
    import tempfile
    from corpus.cic_labels import CIC_ATTACK_WINDOWS

    pcap = Path(pcap_path)
    if not pcap.exists():
        print(f"ERROR: pcap not found: {pcap}")
        sys.exit(1)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
        tmp_parquet = tf.name

    try:
        df, stats = build_corpus_cic(
            pcap_path=pcap,
            cic_windows=CIC_ATTACK_WINDOWS,
            tshark_bin=tshark_bin,
            out_parquet=tmp_parquet,
            allow_skewed=True,
        )
    except Exception as exc:
        print(f"\nERROR during extraction: {exc}")
        if os.path.exists(tmp_parquet):
            os.unlink(tmp_parquet)
        sys.exit(1)
    finally:
        if os.path.exists(tmp_parquet):
            os.unlink(tmp_parquet)

    total = stats.n_attack + stats.n_benign
    print("\n=== CIC sanity-check results ===")
    print(f"  Pcap              : {pcap.name}")
    print(
        f"  Flows kept        : {total:,}  "
        f"(attack={stats.n_attack:,}  benign={stats.n_benign:,})"
    )
    print(
        f"  Attack fraction   : {100*stats.attack_frac:.3f}%  "
        f"Benign: {100*stats.benign_frac:.2f}%"
    )
    print(f"  Dropped           : {stats.n_dropped_unprocessable}  "
          f"reasons={stats.dropped_reasons}")
    print(
        f"  Windows matched   : {stats.label_rows_matched}/{stats.label_rows_total}"
    )

    try:
        assert_sane_balance(stats.n_attack, stats.n_benign)
        print("\n  Balance check     : PASS")
    except CorpusBalanceError as e:
        print(f"\n  Balance check     : FAIL\n  {e}")
        sys.exit(1)


def _cmd_probe_attack(
    row_idx: int,
    pcap_path: str,
    label_csv: str,
    tshark_bin: str,
) -> None:
    """Print per-candidate abs_ts vs GT [stime, ltime] for one attack row.

    Takes one attack row from the GT CSV (by zero-based index), extracts all
    flows from the pcap, and for every flow whose endpoint pair + proto matches
    the attack row (ignoring time), prints:

        ep_a  ep_b  proto  abs_ts  stime  ltime  delta_start  delta_end

    delta_start = abs_ts - stime  (positive: flow after window start)
    delta_end   = abs_ts - ltime  (negative: flow before window end)

    If deltas are a near-constant offset across candidates, that offset is the
    epoch/timezone bug to correct in the GT CSV or the pcap-header reader.
    """
    label_index, _ = load_label_index(label_csv)

    # Fetch the specific row
    df_raw = pd.read_csv(label_csv, low_memory=False)
    if row_idx >= len(df_raw):
        print(f"ERROR: row_idx={row_idx} is out of range (CSV has {len(df_raw)} rows)")
        sys.exit(1)
    # Normalise column names for display (same aliases as load_label_index)
    df_raw.columns = [c.strip().lower().replace(" ", "_") for c in df_raw.columns]
    _DISP_ALIASES = {
        "source_ip": "srcip", "src_ip": "srcip",
        "destination_ip": "dstip", "dst_ip": "dstip",
        "source_port": "sport", "src_port": "sport",
        "destination_port": "dsport", "dst_port": "dsport",
        "protocol": "proto",
        "start_time": "stime",
        "last_time": "ltime",
        "attack_category": "attack_cat",
    }
    df_raw = df_raw.rename(columns={k: v for k, v in _DISP_ALIASES.items() if k in df_raw.columns})
    raw = df_raw.iloc[row_idx]
    print(f"\n=== Probing GT row {row_idx} ===")
    print(f"  srcip={raw.get('srcip', '?')}  "
          f"sport={raw.get('sport', '?')}  "
          f"dstip={raw.get('dstip', '?')}  "
          f"dsport={raw.get('dsport', '?')}  "
          f"proto={raw.get('proto', '?')}  "
          f"stime={raw.get('stime', '?')}  "
          f"ltime={raw.get('ltime', '?')}")

    # Find all index entries for this row (by _row_idx)
    target_entries = []
    for entries in label_index.values():
        for e in entries:
            if e.get("_row_idx") == row_idx:
                target_entries.append(e)
    if not target_entries:
        print("  (row not found in label index — may have been skipped for bad data)")
        return
    entry = target_entries[0]
    tgt_key = orientation_key(entry["srcip"], entry["sport"], entry["dstip"], entry["dsport"])
    tgt_proto = _PROTO_NORM.get(entry["proto"], entry["proto"].upper())
    stime = entry["stime"]
    ltime = entry["ltime"]

    pcap_start = get_pcap_start_epoch(pcap_path)
    if pcap_start is None:
        print("ERROR: cannot read pcap header timestamp")
        sys.exit(1)
    conv_dicts = run_pass_a(tshark_bin, pcap=pcap_path)

    candidates = [
        c for c in conv_dicts
        if (orientation_key(c["ep_a"][0], c["ep_a"][1], c["ep_b"][0], c["ep_b"][1]) == tgt_key
            and c["proto"] == tgt_proto)
    ]

    if not candidates:
        print(f"\n  No flow candidates found for this endpoint pair + proto ({tgt_proto})")
        print("  (The pcap may not contain traffic between these endpoints)")
        return

    print(f"\n  Found {len(candidates)} candidate flow(s) — ignoring time window:\n")
    print(f"  {'ep_a':<22} {'ep_b':<22} {'abs_ts':>16} {'delta_start':>12} {'delta_end':>10}")
    print("  " + "-" * 86)
    in_window = 0
    for c in candidates:
        abs_ts      = pcap_start + c["rel_start"]
        delta_start = abs_ts - stime
        delta_end   = abs_ts - ltime
        marker      = " OK" if stime - TIME_TOLERANCE <= abs_ts <= ltime + TIME_TOLERANCE else ""
        if marker:
            in_window += 1
        ep_a_str = f"{c['ep_a'][0]}:{c['ep_a'][1]}"
        ep_b_str = f"{c['ep_b'][0]}:{c['ep_b'][1]}"
        print(f"  {ep_a_str:<22} {ep_b_str:<22} {abs_ts:>16.3f} "
              f"{delta_start:>+12.3f} {delta_end:>+10.3f}{marker}")
    print(f"\n  GT window : [{stime:.3f}, {ltime:.3f}]  (±{TIME_TOLERANCE}s tolerance)")
    print(f"  In-window : {in_window} / {len(candidates)} candidate(s)")
    if in_window == 0 and candidates:
        deltas = [pcap_start + c["rel_start"] - stime for c in candidates]
        avg_delta = sum(deltas) / len(deltas)
        print(f"  Avg offset: {avg_delta:+.3f}s  "
              f"(positive = pcap timestamps AFTER GT stime; "
              f"negative = BEFORE)")


# ── NFStream probe: re-prove labels land on NFStream output ──────────────────

def _cmd_probe_attack_nf(
    row_idx: int,
    pcap_path: str,
    label_csv: str,
    n_meters: int = 1,
) -> None:
    """Like --probe-attack but using NFStream flows (absolute timestamps).

    For each flow whose orientation_key + proto matches the target GT row
    (ignoring time), print flow.ts vs the GT [stime, ltime] window with delta.
    This re-proves that NFStream's flow grain still places attack flows inside
    the label window — mandatory before trusting an NFStream corpus build.
    """
    label_index, _ = load_label_index(label_csv)

    df_raw = pd.read_csv(label_csv, low_memory=False)
    if row_idx >= len(df_raw):
        print(f"ERROR: row_idx={row_idx} out of range (CSV has {len(df_raw)} rows)")
        import sys; sys.exit(1)
    df_raw.columns = [c.strip().lower().replace(" ", "_") for c in df_raw.columns]
    _DISP_ALIASES = {
        "source_ip": "srcip", "src_ip": "srcip",
        "destination_ip": "dstip", "dst_ip": "dstip",
        "source_port": "sport", "src_port": "sport",
        "destination_port": "dsport", "dst_port": "dsport",
        "protocol": "proto", "start_time": "stime",
        "last_time": "ltime", "attack_category": "attack_cat",
    }
    df_raw = df_raw.rename(columns={k: v for k, v in _DISP_ALIASES.items() if k in df_raw.columns})
    raw = df_raw.iloc[row_idx]

    print(f"\n=== NFStream probe — GT row {row_idx} ===")
    print(f"  srcip={raw.get('srcip','?')}  sport={raw.get('sport','?')}  "
          f"dstip={raw.get('dstip','?')}  dsport={raw.get('dsport','?')}  "
          f"proto={raw.get('proto','?')}  stime={raw.get('stime','?')}  "
          f"ltime={raw.get('ltime','?')}")

    target_entries = [
        e for entries in label_index.values() for e in entries
        if e.get("_row_idx") == row_idx
    ]
    if not target_entries:
        print("  (row not found in label index)")
        return
    entry    = target_entries[0]
    tgt_key  = orientation_key(entry["srcip"], entry["sport"], entry["dstip"], entry["dsport"])
    tgt_proto_num = {"tcp": 6, "udp": 17}.get(entry["proto"].lower(), -1)
    stime    = entry["stime"]
    ltime    = entry["ltime"]

    print(f"\n  Extracting flows via NFStream (n_meters={n_meters})…")
    flows = extract_flows_nfstream(pcap_path, n_meters=n_meters)

    candidates = [
        f for f in flows
        if (orientation_key(f.src_ip, f.src_port, f.dst_ip, f.dst_port) == tgt_key
            and f.proto == tgt_proto_num)
    ]

    if not candidates:
        print(f"\n  No flow candidates for this endpoint pair + proto (numeric={tgt_proto_num})")
        return

    print(f"\n  Found {len(candidates)} candidate flow(s) — ignoring time window:\n")
    print(f"  {'src':<22} {'dst':<22} {'flow.ts':>18} {'delta_start':>12} {'delta_end':>10}")
    print("  " + "-" * 88)
    in_window = 0
    for f in candidates:
        delta_start = f.ts - stime
        delta_end   = f.ts - ltime
        marker      = " OK" if stime - TIME_TOLERANCE <= f.ts <= ltime + TIME_TOLERANCE else ""
        if marker:
            in_window += 1
        src_str = f"{f.src_ip}:{f.src_port}"
        dst_str = f"{f.dst_ip}:{f.dst_port}"
        print(f"  {src_str:<22} {dst_str:<22} {f.ts:>18.3f} "
              f"{delta_start:>+12.3f} {delta_end:>+10.3f}{marker}")

    print(f"\n  GT window : [{stime:.3f}, {ltime:.3f}]  (±{TIME_TOLERANCE}s tolerance)")
    print(f"  In-window : {in_window} / {len(candidates)} candidate(s)")
    if in_window == 0 and candidates:
        avg_delta = sum(f.ts - stime for f in candidates) / len(candidates)
        print(f"  Avg offset: {avg_delta:+.3f}s  "
              f"(positive = NFStream ts AFTER GT stime; negative = BEFORE)")
        print("  VERDICT: FAIL — labels do not land on NFStream output for this row.")
        print("  Investigate: grain change (NFStream idle/active timeout), or timezone issue.")
    else:
        print("  VERDICT: PASS — at least one NFStream flow falls inside the GT window.")


def _cmd_sanity_check_gotham_nf(
    pcap_path: str,
    gotham_root: str,
    n_meters: int = 1,
) -> None:
    """Single-PCAP Gotham sanity check using NFStream (no parquet written)."""
    import sys
    from corpus.gotham_labels import ATTACK_CAT_MAP, ATTACKER_IPS, GothamPcapSpec

    pcap = Path(pcap_path)
    if not pcap.exists():
        print(f"ERROR: pcap not found: {pcap}"); sys.exit(1)

    parent_name = pcap.parent.name
    if parent_name == "benign":
        spec = GothamPcapSpec(pcap_path=pcap, is_attack=False, attack_cat="", attacker_ips=[])
    elif parent_name in ATTACK_CAT_MAP:
        spec = GothamPcapSpec(
            pcap_path=pcap, is_attack=True,
            attack_cat=ATTACK_CAT_MAP[parent_name],
            attacker_ips=ATTACKER_IPS.get(parent_name, []),
        )
    else:
        print(f"ERROR: cannot determine label from parent dir '{parent_name}'.")
        sys.exit(1)

    print(f"\n=== Gotham NFStream sanity-check: {pcap.name} ===")
    print(f"  Label    : {'attack / ' + spec.attack_cat if spec.is_attack else 'benign'}")
    print(f"  Attackers: {spec.attacker_ips or 'N/A (benign)'}")
    print(f"  Extracting via NFStream (n_meters={n_meters})…")

    try:
        flows = extract_flows_nfstream(str(pcap), n_meters=n_meters)
    except Exception as exc:
        print(f"\nERROR: NFStream extraction failed: {exc}"); sys.exit(1)

    batch_rows, stats = _apply_labels_gotham_nf(
        flows, spec.is_attack, spec.attack_cat, spec.attacker_ips,
    )
    total = stats.n_attack + stats.n_benign
    print(f"\n=== Sanity-check results ===")
    print(f"  Flows kept        : {total:,}  (attack={stats.n_attack:,}  benign={stats.n_benign:,})")
    print(f"  Dropped           : {stats.n_dropped_unprocessable}  reasons={stats.dropped_reasons}")
    note = ("PASS (attack pcap is legitimately 100% attack)"
            if spec.is_attack else "PASS (benign pcap is legitimately 100% benign)")
    print(f"  Balance verdict   : {note if total > 0 else 'WARN — no flows'}")


# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from adns_flows.extract import find_tshark

    ap = argparse.ArgumentParser(
        description="ADNS corpus builder and diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset selection
-----------------
  --dataset unsw    (default) UNSW-NB15: time-window label matching via GT CSV
  --dataset gotham  Gotham 2025: directory-based labeling, no GT CSV needed

Subcommands
-----------
  --sanity-check PCAP GT_CSV         (UNSW)
      Run the full pipeline on ONE pcap (no parquet written).
      Prints n_attack / n_benign / benign_fraction, label-row matched/unmatched
      counts, dropped-reason breakdown, and a PASS/FAIL balance verdict.
      Run this before a full multi-pcap build.

  --sanity-check PCAP                (Gotham, requires --dataset gotham)
      Run single-pcap sanity check using directory-based labeling.
      Attack type is detected from the parent directory name.

  --probe-attack ROW_IDX PCAP GT_CSV (UNSW only)
      For one attack row in the GT CSV (by zero-based index), find all candidate
      flows in PCAP by endpoint pair + proto (ignoring time), and print each
      candidate's reconstructed abs_ts vs the GT [stime, ltime] window with
      delta in seconds.  Near-constant offsets indicate an epoch/timezone bug.

  (no subcommand)
      UNSW full build : --pcap-dir DIR --label-csv CSV --out PARQUET
      Gotham full build: --dataset gotham --gotham-root PATH --out PARQUET
""",
    )
    ap.add_argument("--tshark", default=None,
                    help="Path to tshark binary (required only for --extractor tshark)")
    ap.add_argument("--extractor", choices=("nfstream", "tshark"), default="nfstream",
                    help="Extraction engine: 'nfstream' (default) or 'tshark'")
    ap.add_argument("--n-meters", type=int, default=1, metavar="N",
                    help="NFStream worker count (default 1; corpus builds may use higher)")
    ap.add_argument("--allow-skewed", action="store_true",
                    help="Skip class-balance gate (print warning and proceed)")
    ap.add_argument("--flood-cap", type=int, default=DEFAULT_FLOOD_CAP, metavar="N",
                    help=f"Per-source-IP cap on one-sided flood flows "
                         f"(src_pkts<=1, dst_pkts==0). 0=disable. "
                         f"Default {DEFAULT_FLOOD_CAP}")
    ap.add_argument("--dataset", choices=("unsw", "gotham", "cic"), default="unsw",
                    help="Dataset type: 'unsw' (default), 'gotham', or 'cic'")
    ap.add_argument("--gotham-root", metavar="PATH",
                    help="Gotham dataset root directory (required for --dataset gotham)")
    ap.add_argument("--pcap", metavar="PATH",
                    help="(CIC) Path to single PCAP file (full build or sanity-check)")

    sub = ap.add_mutually_exclusive_group()
    sub.add_argument("--sanity-check", nargs="+", metavar="ARG",
                     help="UNSW: PCAP GT_CSV  |  Gotham: PCAP  |  CIC: PCAP")
    sub.add_argument("--probe-attack", nargs=3, metavar=("ROW_IDX", "PCAP", "GT_CSV"),
                     help="(UNSW only) Print time-match diagnostics for one GT attack row")
    sub.add_argument("--probe-cic", metavar="PCAP",
                     help="(CIC only) Print PCAP epoch vs expected attack windows")

    ap.add_argument("--pcap-dir",   help="(UNSW) Directory of pcap files (full build)")
    ap.add_argument("--label-csv",  help="(UNSW) Ground-truth CSV path (full build)")
    ap.add_argument("--out",        help="Output parquet path (full build)")

    args = ap.parse_args()

    tshark = args.tshark or find_tshark()
    if tshark is None and args.extractor == "tshark":
        print("ERROR: tshark binary not found. "
              "Set TSHARK_BIN or pass --tshark.", file=sys.stderr)
        sys.exit(1)
    if tshark is None:
        log.info("tshark not found; using NFStream extractor (tshark not required)")

    if args.probe_cic:
        if args.dataset not in ("cic", "unsw"):  # allow omitting --dataset
            ap.error("--probe-cic is only applicable to CIC PCAPs")
        _cmd_probe_cic(args.probe_cic, tshark)
    elif args.sanity_check:
        if args.dataset == "gotham":
            if len(args.sanity_check) != 1:
                ap.error("Gotham --sanity-check expects exactly one argument: PCAP")
            gotham_root = args.gotham_root or ""
            if args.extractor == "nfstream":
                _cmd_sanity_check_gotham_nf(
                    args.sanity_check[0], gotham_root, n_meters=args.n_meters,
                )
            else:
                _cmd_sanity_check_gotham(args.sanity_check[0], gotham_root, tshark)
        elif args.dataset == "cic":
            if len(args.sanity_check) != 1:
                ap.error("CIC --sanity-check expects exactly one argument: PCAP")
            _cmd_sanity_check_cic(args.sanity_check[0], tshark)
        else:
            if len(args.sanity_check) != 2:
                ap.error("UNSW --sanity-check expects exactly two arguments: PCAP GT_CSV")
            pcap_arg, csv_arg = args.sanity_check
            _cmd_sanity_check(pcap_arg, csv_arg, tshark)
    elif args.probe_attack:
        if args.dataset == "gotham":
            ap.error("--probe-attack is only available for --dataset unsw")
        idx_arg, pcap_arg, csv_arg = args.probe_attack
        if args.extractor == "nfstream":
            _cmd_probe_attack_nf(
                int(idx_arg), pcap_arg, csv_arg, n_meters=args.n_meters,
            )
        else:
            _cmd_probe_attack(int(idx_arg), pcap_arg, csv_arg, tshark)
    elif args.dataset == "gotham":
        if not (args.gotham_root and args.out):
            ap.error("Gotham full build requires --gotham-root PATH and --out PARQUET")
        df, stats = build_corpus_gotham(
            gotham_root=args.gotham_root,
            tshark_bin=tshark,
            out_parquet=args.out,
            allow_skewed=args.allow_skewed,
            flood_cap=args.flood_cap,
            extractor=args.extractor,
            n_meters=args.n_meters,
        )
        print(f"Done. {len(df):,} rows -> {args.out}")
        print(f"  extractor={args.extractor}  n_attack={stats.n_attack}  "
              f"n_benign={stats.n_benign}  n_dropped={stats.n_dropped_unprocessable}")
    elif args.dataset == "cic":
        pcap_arg = args.pcap or (args.sanity_check[0] if args.sanity_check else None)
        if not (pcap_arg and args.out):
            ap.error("CIC full build requires --pcap PATH and --out PARQUET")
        from corpus.cic_labels import CIC_ATTACK_WINDOWS
        df, stats = build_corpus_cic(
            pcap_path=pcap_arg,
            cic_windows=CIC_ATTACK_WINDOWS,
            tshark_bin=tshark,
            out_parquet=args.out,
            allow_skewed=args.allow_skewed,
            flood_cap=args.flood_cap,
            extractor=args.extractor,
            n_meters=args.n_meters,
        )
        print(f"Done. {len(df):,} rows -> {args.out}")
        print(f"  extractor={args.extractor}  n_attack={stats.n_attack}  "
              f"n_benign={stats.n_benign}  n_dropped={stats.n_dropped_unprocessable}")
    else:
        if not (args.pcap_dir and args.label_csv and args.out):
            ap.error("UNSW full build requires --pcap-dir, --label-csv, and --out")
        df, stats = build_corpus(
            pcap_dir=args.pcap_dir,
            label_csv=args.label_csv,
            tshark_bin=tshark,
            out_parquet=args.out,
            allow_skewed=args.allow_skewed,
            flood_cap=args.flood_cap,
            extractor=args.extractor,
            n_meters=args.n_meters,
        )
        print(f"Done. {len(df):,} rows -> {args.out}")
        print(f"  extractor={args.extractor}  n_attack={stats.n_attack}  "
              f"n_benign={stats.n_benign}  n_dropped={stats.n_dropped_unprocessable}")
