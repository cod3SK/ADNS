"""NFStream frozen-exe Phase-0 packaging test.

Modes
-----
  default / --mode pcap            read a synthetic two-packet pcap (built in-memory)
  --mode pcap  --pcap FILE         read a real pcap file
  --mode live                      capture on the default Wi-Fi NPF device for 60 s
  --mode live  --interface NAME    capture on a different NPF device
  --mode live  --duration SEC      override capture window (default 60)
  --list-interfaces                print the default NPF GUID and exit

Frozen-exe multiprocessing notes
---------------------------------
* multiprocessing.freeze_support() is called as the VERY FIRST thing inside
  if __name__ == '__main__': so spawned worker processes are intercepted
  before any heavy work runs.  Without this call each worker re-runs main()
  and the process tree grows exponentially (the fork-bomb we already saw).

* n_meters=1 is set on every NFStreamer call.  NFStream uses
  get_context('spawn') on Windows and spawns n_meters worker processes.
  Fixing the count to 1 keeps the tree bounded at root + 1 child.

* n_dissections=0 disables libndpi DPI, keeping the worker lightweight.

Serving config (decided here)
------------------------------
  n_meters=1, n_dissections=0
  statistical_analysis=True  for corpus/pcap work
  statistical_analysis=False for live detection (lower latency)
"""

import sys
import os
import struct
import io
import tempfile
import argparse
import time
import threading

# ── NFStreamer kwargs ──────────────────────────────────────────────────────────

# Constrained config for frozen/serving use.
# n_meters=1 → exactly 1 worker child; n_dissections=0 → no DPI overhead.
_PCAP_KWARGS = dict(
    n_meters=1,
    n_dissections=0,
    statistical_analysis=True,
)
_LIVE_KWARGS = dict(
    n_meters=1,
    n_dissections=0,
    statistical_analysis=False,
    idle_timeout=5,
    active_timeout=30,
)

# NPF device for Intel Wi-Fi 6E AX211 on this machine.
# Verify with: Get-NetAdapter | Where Status -eq Up
_DEFAULT_IFACE = r"\Device\NPF_{E466F43A-35D6-409B-AC2B-A026C362E238}"


# ── synthetic pcap builder ────────────────────────────────────────────────────

def build_mini_pcap() -> bytes:
    """Build a minimal libpcap file with one TCP SYN + SYN-ACK flow."""
    def _ip_pack(ip):
        return bytes(int(p) for p in ip.split('.'))

    def _tcp(sp, dp, seq, ack, flags):
        off = (5 << 12) | flags
        return struct.pack('!HHIIHHHH', sp, dp, seq, ack, off, 65535, 0, 0)

    def _ip4(src, dst, proto, payload):
        t = 20 + len(payload)
        return struct.pack('!BBHHHBBH4s4s',
                           0x45, 0, t, 0, 0x4000, 64, proto, 0,
                           _ip_pack(src), _ip_pack(dst)) + payload

    def _eth(p):
        return b'\xaa\xbb\xcc\xdd\xee\x02' + b'\xaa\xbb\xcc\xdd\xee\x01' + b'\x08\x00' + p

    def _pkt(ts, us, src, dst, proto, seg):
        d = _eth(_ip4(src, dst, proto, seg))
        return struct.pack('<IIII', ts, us, len(d), len(d)) + d

    buf = io.BytesIO()
    buf.write(struct.pack('<IHHiIII', 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
    T = 1_700_000_000
    buf.write(_pkt(T,   0, '192.168.1.1', '10.0.0.1', 6,
                   _tcp(12345, 80, 1000, 0,    0x002)))  # SYN
    buf.write(_pkt(T, 100, '10.0.0.1', '192.168.1.1', 6,
                   _tcp(80, 12345, 2000, 1001, 0x012)))  # SYN-ACK
    return buf.getvalue()


# ── test modes ────────────────────────────────────────────────────────────────

def run_pcap_test(pcap_path=None) -> int:
    """Read from a pcap file and return flow count (>0 is PASS)."""
    from nfstream import NFStreamer   # deferred so child workers don't import
    print("NFStream import: OK")

    cleanup = False
    if pcap_path is None:
        data = build_mini_pcap()
        with tempfile.NamedTemporaryFile(suffix='.pcap', delete=False) as f:
            f.write(data)
            pcap_path = f.name
        cleanup = True
        print(f"Synthetic pcap written: {pcap_path}")
    else:
        print(f"Reading pcap: {pcap_path}")

    try:
        flows = list(NFStreamer(source=pcap_path, **_PCAP_KWARGS))
        print(f"Flows produced: {len(flows)}")
        for fl in flows:
            print(f"  {fl.src_ip}:{fl.src_port} -> {fl.dst_ip}:{fl.dst_port}"
                  f"  proto={fl.protocol}"
                  f"  bytes={fl.bidirectional_bytes}"
                  f"  syn={fl.bidirectional_syn_packets}")
        return len(flows)
    finally:
        if cleanup:
            try:
                os.unlink(pcap_path)
            except OSError:
                pass


def run_live_test(interface: str, duration_sec: int) -> int:
    """Capture live traffic for duration_sec and return flow count.

    A count of 0 is a WARN (idle interface), not a hard FAIL.
    The child-count-over-time is measured externally by run_frozen_guarded.ps1.
    """
    from nfstream import NFStreamer   # deferred

    print(f"NFStream import: OK")
    print(f"Live capture:  interface={interface!r}  duration={duration_sec}s")
    print(f"Start: {time.strftime('%H:%M:%S')}")
    sys.stdout.flush()

    stop = threading.Event()
    timer = threading.Timer(duration_sec, stop.set)
    timer.daemon = True
    timer.start()

    streamer = NFStreamer(source=interface, **_LIVE_KWARGS)
    n = 0
    t0 = time.time()
    try:
        for flow in streamer:
            n += 1
            if n % 100 == 0 or (time.time() - t0) > (duration_sec - 2):
                print(f"  [{time.time()-t0:.0f}s] flows so far: {n}")
                sys.stdout.flush()
            if stop.is_set():
                break
    finally:
        timer.cancel()

    elapsed = time.time() - t0
    print(f"Stop: {time.strftime('%H:%M:%S')} — {n} flows in {elapsed:.1f}s")
    sys.stdout.flush()
    return n


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="nfstream_pkg_test",
        description="NFStream Phase-0 frozen-exe packaging test",
    )
    parser.add_argument("--mode", choices=["pcap", "live"], default="pcap",
                        help="pcap=read file  live=capture interface  (default: pcap)")
    parser.add_argument("--pcap", metavar="FILE",
                        help="Pcap file path for mode=pcap (default: synthetic)")
    parser.add_argument("--interface", default=_DEFAULT_IFACE,
                        help="NPF device name for mode=live")
    parser.add_argument("--duration", type=int, default=60,
                        help="Live capture seconds (default: 60)")
    parser.add_argument("--list-interfaces", action="store_true",
                        help="Print default Wi-Fi NPF GUID and exit")
    args = parser.parse_args()

    print(f"Python {sys.version}")
    print(f"Frozen : {getattr(sys, 'frozen', False)}")
    print(f"MEIPASS: {getattr(sys, '_MEIPASS', 'N/A')}")
    sys.stdout.flush()

    if args.list_interfaces:
        print(f"Default Wi-Fi NPF device: {_DEFAULT_IFACE}")
        sys.exit(0)

    try:
        if args.mode == "pcap":
            n = run_pcap_test(args.pcap)
            if n > 0:
                print("PCAP TEST: PASS")
                sys.exit(0)
            else:
                print("PCAP TEST: FAIL — 0 flows produced")
                sys.exit(1)
        else:
            n = run_live_test(args.interface, args.duration)
            if n > 0:
                print(f"LIVE TEST: PASS  (flows={n})")
            else:
                print(f"LIVE TEST: WARN  (flows=0, interface may be idle)")
            sys.exit(0)
    except Exception as exc:
        import traceback
        print(f"TEST FAIL — {exc}")
        traceback.print_exc()
        sys.exit(1)


# ── freeze support ────────────────────────────────────────────────────────────
# MUST be the first call inside __main__ so spawned workers are intercepted
# before any heavy work (NFStream import, pcap open) runs.

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
