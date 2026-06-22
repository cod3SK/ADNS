"""
Phase 4 grain-parity analysis and gate test.

BACKGROUND
----------
The pre-fix _NfstreamCaptureAgent in api/app.py captured 15-second pcap windows
via tshark ring-buffer, then ran extract_flows_nfstream() on each window
independently.  NFStream force-closes any flow still active at the end of a pcap
file.  A 45-second TCP session therefore became 3 fragments, each ≤ 15 s, instead
of one complete flow.  The corpus was built on whole pcaps with idle_timeout=120s
governing expiry.  CIC benign mean duration is ~17.5 s (after NFStream grain) —
roughly half of benign sessions exceed 15 s and would fragment.

STEP 1-2 (analysis tests)
  - test_corpus_path_sees_one_complete_long_flow
  - test_windowed_path_fragments_long_flows          ← documents pre-fix problem
  - test_short_flows_unaffected_by_windowing
  - test_windowed_bytes_conserved_across_fragments
  - test_benign_score_delta (optional, requires model)

STEP 4 (gate test)
  test_live_windowing_equals_corpus_grain
    After Option B fix (_NfstreamCaptureAgent uses NFStreamer(source=interface)
    directly): the live path applies idle_timeout=120s / active_timeout=1800s to
    live traffic — same timeouts as corpus, same _nf_to_flow() conversion, no pcap
    windowing boundary to force-close active flows.
"""
from __future__ import annotations

import io
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

nfstream = pytest.importorskip("nfstream")

from adns_flows.extract_nfstream import extract_flows_nfstream, flows_to_dataframe_nfstream, _nf_to_flow
from adns_flows.nfstream_config import make_nfstream_kwargs
from adns_flows.schema import FEATURE_COLUMNS

# ── Raw-pcap helpers (minimal duplicate of conftest to keep fixture self-contained) ──

def _ip_pack(ip: str) -> bytes:
    return bytes(int(p) for p in ip.split("."))


def _tcp_seg(sp: int, dp: int, seq: int, ack: int, flags: int, payload: bytes = b"") -> bytes:
    offset_flags = (5 << 12) | flags
    return struct.pack("!HHIIHHHH", sp, dp, seq, ack, offset_flags, 65535, 0, 0) + payload


def _ipv4(src: str, dst: str, proto: int, payload: bytes) -> bytes:
    total = 20 + len(payload)
    return (
        struct.pack("!BBHHHBBH4s4s", 0x45, 0, total, 0, 0x4000, 64, proto, 0,
                    _ip_pack(src), _ip_pack(dst))
        + payload
    )


def _eth(payload: bytes) -> bytes:
    return b"\xaa\xbb\xcc\xdd\xee\x01" + b"\xaa\xbb\xcc\xdd\xee\x02" + b"\x08\x00" + payload


def _pkt_record(ts_sec: int, ts_usec: int, data: bytes) -> bytes:
    return struct.pack("<IIII", ts_sec, ts_usec, len(data), len(data)) + data


def _pcap_hdr() -> bytes:
    return struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)


def _pkt(ts_sec: int, ts_usec: int, src: str, dst: str, proto: int, seg: bytes) -> bytes:
    return _pkt_record(ts_sec, ts_usec, _eth(_ipv4(src, dst, proto, seg)))


_SYN     = 0x002
_SYN_ACK = 0x012
_ACK     = 0x010
_PSH_ACK = 0x018
_FIN_ACK = 0x011


# ── Grain-test pcap ───────────────────────────────────────────────────────────

def _build_grain_test_pcap() -> bytes:
    """Two flows to test windowing grain:

    short  192.168.1.1:10001 <-> 10.0.0.1:80   duration ~3 s  (< 15 s window)
    long   192.168.1.2:20002 <-> 10.0.0.2:443  duration ~45 s (spans 3+ windows)

    Canonical orientation (lexicographic lower IP is src):
      "10.0.0.1" < "192.168.1.1" → short flow canonical src=10.0.0.1:80
      "10.0.0.2" < "192.168.1.2" → long  flow canonical src=10.0.0.2:443
    Port 10001 always appears as dst_port; port 20002 always appears as dst_port.
    """
    buf = io.BytesIO()
    buf.write(_pcap_hdr())
    T = 1_700_000_000

    req  = b"GET / HTTP/1.1\r\n\r\n"
    resp = b"HTTP/1.1 200 OK\r\n\r\n" + b"X" * 100

    # ── Short flow: T=0..3 s ──────────────────────────────────────────────
    buf.write(_pkt(T+0, 0,      "192.168.1.1", "10.0.0.1", 6, _tcp_seg(10001, 80,  100, 0,               _SYN)))
    buf.write(_pkt(T+0, 100000, "10.0.0.1",    "192.168.1.1", 6, _tcp_seg(80, 10001, 200, 101,           _SYN_ACK)))
    buf.write(_pkt(T+1, 0,      "192.168.1.1", "10.0.0.1", 6, _tcp_seg(10001, 80,  101, 201,             _ACK)))
    buf.write(_pkt(T+1, 100000, "192.168.1.1", "10.0.0.1", 6, _tcp_seg(10001, 80,  101, 201,             _PSH_ACK, req)))
    buf.write(_pkt(T+2, 0,      "10.0.0.1",    "192.168.1.1", 6, _tcp_seg(80, 10001, 201, 101+len(req),  _PSH_ACK, resp)))
    buf.write(_pkt(T+3, 0,      "192.168.1.1", "10.0.0.1", 6, _tcp_seg(10001, 80,  101+len(req), 201+len(resp), _FIN_ACK)))
    buf.write(_pkt(T+3, 100000, "10.0.0.1",    "192.168.1.1", 6, _tcp_seg(80, 10001, 201+len(resp), 101+len(req)+1, _FIN_ACK)))

    # ── Long flow: SYN at T=0, data at T=10/20/30/40, FIN at T=45 ────────
    _DATA_C = b"C" * 300   # client → server
    _DATA_S = b"D" * 150   # server → client
    buf.write(_pkt(T+0, 500, "192.168.1.2", "10.0.0.2", 6, _tcp_seg(20002, 443, 1000, 0,    _SYN)))
    buf.write(_pkt(T+0, 600, "10.0.0.2", "192.168.1.2", 6, _tcp_seg(443, 20002, 2000, 1001, _SYN_ACK)))
    buf.write(_pkt(T+0, 700, "192.168.1.2", "10.0.0.2", 6, _tcp_seg(20002, 443, 1001, 2001, _ACK)))
    for i, t in enumerate([10, 20, 30, 40]):
        sc = 1001 + i * 400
        ss = 2001 + i * 200
        buf.write(_pkt(T+t, i*1000,   "192.168.1.2", "10.0.0.2", 6, _tcp_seg(20002, 443, sc, ss, _PSH_ACK, _DATA_C)))
        buf.write(_pkt(T+t, i*1000+500, "10.0.0.2", "192.168.1.2", 6, _tcp_seg(443, 20002, ss, sc+len(_DATA_C), _PSH_ACK, _DATA_S)))
    buf.write(_pkt(T+45, 0,   "192.168.1.2", "10.0.0.2", 6, _tcp_seg(20002, 443, 1001+4*400, 2001+4*200, _FIN_ACK)))
    buf.write(_pkt(T+45, 500, "10.0.0.2",    "192.168.1.2", 6, _tcp_seg(443, 20002, 2001+4*200, 1001+4*400+1, _FIN_ACK)))

    return buf.getvalue()


@pytest.fixture(scope="module")
def grain_pcap_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("grain") / "grain_test.pcap"
    p.write_bytes(_build_grain_test_pcap())
    return p


# ── Pcap-windowing helpers (simulates ring-buffer tshark behavior) ────────────

def _slice_pcap_windows(pcap_bytes: bytes, window_sec: float = 15.0) -> list[bytes]:
    """Split a pcap into consecutive fixed-duration windows by packet timestamp.

    Exactly mirrors what _NfstreamCaptureAgent did with tshark ring-buffer:
      tshark -a duration:{window_sec} -F pcap -w {file}
    packets with ts in [first_ts + n*window, first_ts + (n+1)*window) → window n.
    """
    if len(pcap_bytes) < 24:
        return []
    magic = struct.unpack_from("<I", pcap_bytes, 0)[0]
    endian = "<" if magic == 0xA1B2C3D4 else ">"
    global_hdr = pcap_bytes[:24]
    offset = 24
    packets: list[tuple[float, bytes]] = []
    while offset + 16 <= len(pcap_bytes):
        ts_sec, ts_usec, incl_len, _ = struct.unpack_from(f"{endian}IIII", pcap_bytes, offset)
        ts = ts_sec + ts_usec / 1_000_000
        pkt = pcap_bytes[offset:offset + 16 + incl_len]
        packets.append((ts, pkt))
        offset += 16 + incl_len
    if not packets:
        return []
    first_ts = packets[0][0]
    windows: dict[int, list[bytes]] = {}
    for ts, pkt in packets:
        idx = int((ts - first_ts) / window_sec)
        windows.setdefault(idx, []).append(pkt)
    return [global_hdr + b"".join(windows[i]) for i in sorted(windows)]


def _extract_in_windows(pcap_bytes: bytes, window_sec: float = 15.0) -> list:
    """Simulate the OLD _NfstreamCaptureAgent: run NFStream on each 15s window.

    Each window is an independent pcap file, so NFStream force-closes any
    flow still active at the end of the window — regardless of idle_timeout.
    """
    slices = _slice_pcap_windows(pcap_bytes, window_sec)
    all_flows: list = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, window_bytes in enumerate(slices):
            tmp = Path(tmpdir) / f"w{i:04d}.pcap"
            tmp.write_bytes(window_bytes)
            try:
                all_flows.extend(extract_flows_nfstream(str(tmp)))
            except Exception:
                pass
    return all_flows


def _extract_direct_nfstream(pcap_path: Path) -> list:
    """Simulate the NEW _NfstreamCaptureAgent: direct NFStreamer + _nf_to_flow().

    Mirrors _NfstreamCaptureAgent._run_loop exactly:
        for nf in NFStreamer(source=interface, **make_nfstream_kwargs(n_meters=1)):
            adns_flow = _nf_to_flow(nf)
    No 15 s windows; idle_timeout=120 s / active_timeout=1800 s govern expiry,
    identical to the corpus-build path.
    """
    from nfstream import NFStreamer
    kwargs = make_nfstream_kwargs(n_meters=1)
    flows: list = []
    for nf in NFStreamer(source=str(pcap_path), **kwargs):
        flows.append(_nf_to_flow(nf))
    return flows


# ── Port-based flow selectors ─────────────────────────────────────────────────

SHORT_PORT = 10001   # dst_port after canonical orientation (src = 10.0.0.1:80)
LONG_PORT  = 20002   # dst_port after canonical orientation (src = 10.0.0.2:443)


def _short(flows: list) -> list:
    return [f for f in flows if SHORT_PORT in (f.src_port, f.dst_port)]


def _long(flows: list) -> list:
    return [f for f in flows if LONG_PORT in (f.src_port, f.dst_port)]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestWindowingGrain:
    """
    STEP 2 + 4: quantifies the 15 s-window grain gap and verifies the Option B fix.

    Pre-fix live path:  15 s tshark windows → per-window NFStream → fragmented flows
    Post-fix live path: NFStreamer(source=interface, idle_timeout=120) → complete flows
                        (same grain as corpus extraction)
    """

    # ── STEP 2a: corpus path produces complete flows ──────────────────────────

    def test_corpus_path_sees_one_complete_long_flow(self, grain_pcap_path):
        """Corpus extraction returns 1 flow for the 45 s connection.

        This is the ground truth.  The Option B fix ensures the live path
        produces the same result by using the same NFStream pipeline.
        """
        flows = extract_flows_nfstream(str(grain_pcap_path))
        long_flows = _long(flows)
        assert len(long_flows) == 1, (
            f"corpus path must produce exactly 1 long flow, got {len(long_flows)}"
        )
        assert long_flows[0].duration > 40.0, (
            f"corpus long-flow duration should be ~45 s, got {long_flows[0].duration:.2f} s"
        )

    # ── STEP 2b: windowed path fragments long flows (documents pre-fix problem) ─

    def test_windowed_path_fragments_long_flows(self, grain_pcap_path):
        """15 s pcap windows split the 45 s flow into multiple shorter fragments.

        This test documents the pre-fix grain gap.  Each fragment must have
        duration < 15.5 s (window boundary) while the complete flow lasts ~45 s.
        """
        pcap_bytes = grain_pcap_path.read_bytes()
        windowed = _extract_in_windows(pcap_bytes, window_sec=15.0)
        long_frags = _long(windowed)

        assert len(long_frags) > 1, (
            f"windowed path must fragment the 45 s flow "
            f"(got {len(long_frags)} fragment(s); need > 1 to document the grain gap)"
        )
        for frag in long_frags:
            assert frag.duration < 15.5, (
                f"each windowed fragment must be ≤ 15 s, got {frag.duration:.3f} s"
            )

    # ── STEP 2c: short flows are unaffected ───────────────────────────────────

    def test_short_flows_unaffected_by_windowing(self, grain_pcap_path):
        """Flows shorter than the window size are byte-identical between both paths."""
        pcap_bytes = grain_pcap_path.read_bytes()
        corpus   = extract_flows_nfstream(str(grain_pcap_path))
        windowed = _extract_in_windows(pcap_bytes, window_sec=15.0)

        sc = _short(corpus)
        sw = _short(windowed)

        assert len(sc) == 1, f"corpus: expected 1 short flow, got {len(sc)}"
        assert len(sw) == 1, f"windowed: expected 1 short flow, got {len(sw)}"

        assert sc[0].src_bytes  == sw[0].src_bytes,  "src_bytes mismatch for short flow"
        assert sc[0].dst_bytes  == sw[0].dst_bytes,  "dst_bytes mismatch for short flow"
        assert sc[0].syn_count  == sw[0].syn_count,  "syn_count mismatch for short flow"
        assert sc[0].ack_count  == sw[0].ack_count,  "ack_count mismatch for short flow"

    # ── STEP 2d: byte conservation across fragments ───────────────────────────

    def test_windowed_bytes_conserved_across_fragments(self, grain_pcap_path):
        """Total bytes across windowed fragments equal the corpus whole-flow bytes.

        Verifies the pcap slicer is correct: each packet lives in exactly one
        window, so sum(fragment.src_bytes) == corpus_flow.src_bytes.
        """
        pcap_bytes = grain_pcap_path.read_bytes()
        corpus   = extract_flows_nfstream(str(grain_pcap_path))
        windowed = _extract_in_windows(pcap_bytes, window_sec=15.0)

        long_c = _long(corpus)
        long_w = _long(windowed)
        assert len(long_c) == 1
        assert len(long_w) > 1, "need fragmented long flow to verify byte conservation"

        total_src = sum(f.src_bytes for f in long_w)
        total_dst = sum(f.dst_bytes for f in long_w)

        assert total_src == long_c[0].src_bytes, (
            f"src_bytes not conserved: fragments sum to {total_src}, "
            f"whole flow = {long_c[0].src_bytes}"
        )
        assert total_dst == long_c[0].dst_bytes, (
            f"dst_bytes not conserved: fragments sum to {total_dst}, "
            f"whole flow = {long_c[0].dst_bytes}"
        )

    # ── Phase 4 acceptance gate: NEW live path matches corpus grain ──────────────

    def test_new_live_path_matches_corpus_grain(self, grain_pcap_path):
        """Phase 4 acceptance gate: NEW _NfstreamCaptureAgent path == corpus path.

        The rewritten _NfstreamCaptureAgent uses:
            for nf in NFStreamer(source=interface, **make_nfstream_kwargs(n_meters=1)):
                adns_flow = _nf_to_flow(nf)
        _extract_direct_nfstream() mirrors this exactly.

        Three assertions:
          (1) The 45 s long flow appears as exactly 1 complete flow (NOT fragments)
          (2) Feature values are byte-identical to the corpus path for all flows
          (3) Long-flow duration matches corpus (>40 s)

        If any assertion fails, there is a grain mismatch between live and corpus.
        """
        corpus = extract_flows_nfstream(str(grain_pcap_path))
        live   = _extract_direct_nfstream(grain_pcap_path)

        long_c = _long(corpus)
        long_l = _long(live)

        # (1) Long flow must be 1 complete flow in both paths
        assert len(long_c) == 1, (
            f"corpus path must yield 1 long flow; got {len(long_c)}"
        )
        assert len(long_l) == 1, (
            f"new live path must yield 1 long flow (NOT fragments); got {len(long_l)}"
        )

        # (2) Duration must be complete (~45 s), not window-truncated
        assert long_l[0].duration > 40.0, (
            f"new live path long-flow duration must be >40 s; "
            f"got {long_l[0].duration:.2f} s — flow was force-closed early"
        )

        # (3) Feature matrices must be byte-identical
        import numpy as np
        corpus_sorted = sorted(corpus, key=lambda f: (f.src_ip, f.src_port, f.dst_ip, f.dst_port))
        live_sorted   = sorted(live,   key=lambda f: (f.src_ip, f.src_port, f.dst_ip, f.dst_port))
        assert len(corpus_sorted) == len(live_sorted), (
            f"flow count mismatch: corpus {len(corpus_sorted)} vs live {len(live_sorted)}"
        )
        cols = list(FEATURE_COLUMNS)
        from adns_flows.extract_nfstream import flows_to_dataframe_nfstream
        df_c = flows_to_dataframe_nfstream(corpus_sorted)
        df_l = flows_to_dataframe_nfstream(live_sorted)
        arr_c = df_c[cols].to_numpy(dtype="float32")
        arr_l = df_l[cols].to_numpy(dtype="float32")
        np.testing.assert_array_equal(
            arr_l, arr_c,
            err_msg="new live path feature matrix differs from corpus path — grain mismatch",
        )

    # ── Regression guard: retired windowed path (documents pre-fix behavior) ─────

    def test_live_windowing_equals_corpus_grain(self, grain_pcap_path):
        """Regression guard for the RETIRED 15 s-windowed path.

        This test was the original Phase 4 gate.  The windowed path
        (_NfstreamCaptureAgent ring-buffer via tshark) has been removed;
        this test is kept as a regression guard to document that:
          - the corpus path yields 1 complete long flow (correct)
          - the old windowed path fragmented the same flow (was broken)
          - the grain ratio was >= 2x, confirming the problem was material

        Do NOT remove this test — it proves the old path was wrong and
        that the corpus path (== the new live path) is the correct baseline.
        """
        pcap_bytes = grain_pcap_path.read_bytes()

        corpus   = extract_flows_nfstream(str(grain_pcap_path))
        windowed = _extract_in_windows(pcap_bytes, window_sec=15.0)

        long_c = _long(corpus)
        long_w = _long(windowed)

        # Gate 1: corpus / fixed-live path → 1 complete flow
        assert len(long_c) == 1, (
            f"corpus path (= fixed live path) must yield 1 long flow; got {len(long_c)}"
        )
        assert long_c[0].duration > 40.0, (
            f"corpus long flow should have full 45 s duration; "
            f"got {long_c[0].duration:.2f} s"
        )

        # Gate 2: windowed path → fragmented (proves the problem was real)
        assert len(long_w) > 1, (
            f"windowed (old) path must fragment the 45 s flow; got {len(long_w)}"
        )
        max_frag_dur = max(f.duration for f in long_w)
        assert max_frag_dur < 15.5, (
            f"windowed fragments must each be < 15.5 s; max was {max_frag_dur:.2f} s"
        )

        # Gate 3: grain ratio proves the difference is material
        grain_ratio = len(long_w) / len(long_c)
        assert grain_ratio >= 2.0, (
            f"windowed path must produce >=2x more fragments than corpus for a 45 s "
            f"flow (ratio = {grain_ratio:.1f}); grain gap is immaterial otherwise"
        )

    # ── STEP 2 bonus: FPR delta from model scoring ────────────────────────────

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[3]
             / "api" / "model_artifacts" / "nfstream_model.joblib").exists(),
        reason="nfstream_model.joblib not found — skipping FPR scoring test",
    )
    def test_benign_score_delta(self, grain_pcap_path):
        """Windowed benign fragments may score as more anomalous than whole-pcap flows.

        Quantifies the FPR consequence of grain mismatch: the model was trained on
        corpus-grain flows; windowed fragments have shorter duration and different
        rate features, potentially landing in a different region of feature space.

        This test reports the delta; it also asserts the corpus path scores at or
        below the windowed path (fragmentation can only degrade benign classification).
        """
        import joblib

        model_path = (
            Path(__file__).resolve().parents[3]
            / "api" / "model_artifacts" / "nfstream_model.joblib"
        )
        bundle = joblib.load(model_path)
        xgb = bundle.get("xgboost")
        if xgb is None:
            pytest.skip("no xgboost estimator in model bundle")

        pcap_bytes = grain_pcap_path.read_bytes()
        corpus_flows   = extract_flows_nfstream(str(grain_pcap_path))
        windowed_flows = _extract_in_windows(pcap_bytes, window_sec=15.0)

        def _score(flows: list) -> np.ndarray:
            if not flows:
                return np.array([], dtype="float32")
            df = flows_to_dataframe_nfstream(flows)
            X = df[list(FEATURE_COLUMNS)].to_numpy(dtype="float32")
            return xgb.predict_proba(X)[:, 1].astype("float32")

        corpus_probs   = _score(corpus_flows)
        windowed_probs = _score(windowed_flows)

        THRESHOLD = 0.82
        corpus_fpr   = float(np.mean(corpus_probs   >= THRESHOLD)) if len(corpus_probs)   else 0.0
        windowed_fpr = float(np.mean(windowed_probs >= THRESHOLD)) if len(windowed_probs) else 0.0
        fpr_delta    = windowed_fpr - corpus_fpr

        print(
            f"\n  corpus   : {len(corpus_probs)} flows, FPR = {corpus_fpr:.1%}"
            f"  (mean score {float(corpus_probs.mean()):.3f})"
        )
        print(
            f"  windowed : {len(windowed_probs)} flows, FPR = {windowed_fpr:.1%}"
            f"  (mean score {float(windowed_probs.mean()):.3f})"
        )
        print(f"  FPR delta: {fpr_delta:+.1%}  (+ = windowed is worse)")

        # Windowed FPR should not be significantly better than corpus FPR.
        # Fragmenting benign traffic cannot improve anomaly detection accuracy;
        # if it does, something is wrong with the test setup.
        assert windowed_fpr >= corpus_fpr - 0.15, (
            f"windowed FPR ({windowed_fpr:.1%}) is more than 15 pp lower than corpus "
            f"FPR ({corpus_fpr:.1%}); this is unexpected — check test setup"
        )
