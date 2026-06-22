"""Canonical NFStream configuration — single source of truth for feature-affecting parameters.

GOVERNING PRINCIPLE (from CLAUDE.md §EXTRACTION):
The config that builds the corpus and the config that runs live MUST be identical
in every parameter that affects a contract feature.  This module is the SSoT for
that invariant, analogous to schema.py for the feature list.

Feature-affecting parameters (NFSTREAM_FEATURE_PARAMS)
-------------------------------------------------------
  statistical_analysis=True
      REQUIRED.  Enables TCP flag count fields: bidirectional_syn_packets,
      bidirectional_ack_packets, bidirectional_rst_packets,
      bidirectional_fin_packets, bidirectional_psh_packets,
      bidirectional_urg_packets.  Setting False produces zero for all of them.
      That would silently zero 6 of 21 contract features (syn/ack/rst/fin/psh/urg_count).

  n_dissections=0
      DPI off.  No current contract feature uses NFStream's libndpi labels.
      The TCP flag counts come from statistical_analysis, NOT from DPI, so
      disabling DPI does not affect them.

  idle_timeout=120 (seconds)
      A flow with 2 min of silence is considered ended.  Wide enough to capture
      slow-scan attacks (typical inter-packet interval < 60 s) without holding
      state indefinitely.

  active_timeout=1800 (seconds)
      Long flows (tunnels, persistent connections) are force-closed at 30 min.
      This matches NFStream's own default and typical IDS conventions.

Flow-grain consequence
----------------------
Changing idle_timeout or active_timeout changes how long multi-packet flows
appear and thus changes duration, bytes_per_sec, and pkts_per_sec.  The corpus
must be built with the same grain as inference, or the model will see a shifted
feature distribution.  These values are therefore in NFSTREAM_FEATURE_PARAMS
even though they don't affect per-packet counts.

Feature-neutral parameters
--------------------------
  n_meters (default 1 for serving)
      Pure parallelism: how many NFStream worker processes to spawn.
      n_meters=1 caps the frozen-exe process tree at root+1 (Phase 0 constraint).
      For corpus building on multi-core machines n_meters can be increased — output
      is identical because NFStream workers independently process disjoint packet
      slices and merge deterministically.
      Do NOT change n_meters in corpus builds without verifying determinism.

Byte-accounting note
--------------------
NFStream accounting_mode=0 (the default) counts Ethernet frame bytes (L2).
A minimal TCP segment = 14 (Eth) + 20 (IP) + 20 (TCP) = 54 bytes.
tshark -z conv,tcp counts at the same layer (Ethernet frame bytes), so both
extractors report identical byte counts for the same pcap.  Verified empirically
in test_nfstream_parity.py::test_flag_counts_match_tshark.
"""
from __future__ import annotations

# ── Feature-affecting parameters ──────────────────────────────────────────────
# Identical in corpus-build and live-serve paths.  Changing ANY of these changes
# what the model sees at inference time and requires a full corpus rebuild.
NFSTREAM_FEATURE_PARAMS: dict = {
    "statistical_analysis": True,   # Required for TCP flag counts (6/21 features)
    "n_dissections": 0,             # DPI off; no contract feature needs app-layer labels
    "idle_timeout": 120,            # 2-min silence → flow closed
    "active_timeout": 1800,         # 30-min hard limit
}

# ── Feature-neutral parameters ─────────────────────────────────────────────────
_DEFAULT_N_METERS = 1   # 1 worker child; frozen-exe serving safety (Phase 0 constraint)


def make_nfstream_kwargs(*, n_meters: int = _DEFAULT_N_METERS) -> dict:
    """Return the complete kwargs dict for NFStreamer.

    Merges NFSTREAM_FEATURE_PARAMS with the requested parallelism level.
    Both corpus builders and the live serving path should call this function
    instead of constructing kwargs inline, so any future config change propagates
    automatically to all callers.

    n_meters: number of NFStream worker processes.  Default 1 for serving safety.
    """
    return {**NFSTREAM_FEATURE_PARAMS, "n_meters": n_meters}
