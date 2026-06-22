"""
Shared fixtures for adns_flows tests.

build_fixture_pcap() constructs a minimal pcap in pure Python (no scapy) using
raw struct packing. The pcap contains three flows:
  1. Normal TCP — 192.168.1.1:12345 <-> 10.0.0.1:80  (full handshake + data)
  2. SYN-heavy  — 192.168.1.2:54321 <-> 10.0.0.2:22  (5 SYNs, 1 RST-ACK)
  3. UDP DNS    — 192.168.1.1:54322 <-> 8.8.8.8:53    (1 query, 1 response)
"""
from __future__ import annotations

import io
import struct

import pytest


# ── raw pcap / Ethernet / IP / TCP / UDP constructors ─────────────────────

def _ip_pack(ip: str) -> bytes:
    return bytes(int(p) for p in ip.split("."))


def _tcp_seg(
    sp: int, dp: int, seq: int, ack: int, flags: int, payload: bytes = b""
) -> bytes:
    offset_flags = (5 << 12) | flags
    return (
        struct.pack("!HHIIHHHH", sp, dp, seq, ack, offset_flags, 65535, 0, 0)
        + payload
    )


def _udp_seg(sp: int, dp: int, payload: bytes = b"") -> bytes:
    return struct.pack("!HHHH", sp, dp, 8 + len(payload), 0) + payload


def _ipv4(src: str, dst: str, proto: int, payload: bytes) -> bytes:
    total = 20 + len(payload)
    return (
        struct.pack(
            "!BBHHHBBH4s4s",
            0x45, 0, total, 0, 0x4000, 64, proto, 0,
            _ip_pack(src), _ip_pack(dst),
        )
        + payload
    )


def _eth(payload: bytes) -> bytes:
    mac_a = b"\xaa\xbb\xcc\xdd\xee\x01"
    mac_b = b"\xaa\xbb\xcc\xdd\xee\x02"
    return mac_b + mac_a + b"\x08\x00" + payload


def _pkt_record(ts_sec: int, ts_usec: int, data: bytes) -> bytes:
    return struct.pack("<IIII", ts_sec, ts_usec, len(data), len(data)) + data


def _pcap_header() -> bytes:
    return struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)


def _pkt(ts_sec: int, ts_usec: int, src: str, dst: str, proto: int, seg: bytes) -> bytes:
    return _pkt_record(ts_sec, ts_usec, _eth(_ipv4(src, dst, proto, seg)))


_SYN     = 0x002
_SYN_ACK = 0x012
_ACK     = 0x010
_PSH_ACK = 0x018
_FIN_ACK = 0x011
_RST_ACK = 0x014


def build_fixture_pcap() -> bytes:
    buf = io.BytesIO()
    buf.write(_pcap_header())
    T = 1_700_000_000

    # ── flow 1: normal TCP handshake + data ───────────────────────────────
    req  = b"GET / HTTP/1.1\r\n\r\n"
    resp = b"HTTP/1.1 200 OK\r\n\r\nHello"

    buf.write(_pkt(T, 0,   "192.168.1.1", "10.0.0.1", 6, _tcp_seg(12345, 80, 1000, 0,                       _SYN)))
    buf.write(_pkt(T, 100, "10.0.0.1", "192.168.1.1", 6, _tcp_seg(80, 12345, 2000, 1001,                    _SYN_ACK)))
    buf.write(_pkt(T, 200, "192.168.1.1", "10.0.0.1", 6, _tcp_seg(12345, 80, 1001, 2001,                    _ACK)))
    buf.write(_pkt(T, 300, "192.168.1.1", "10.0.0.1", 6, _tcp_seg(12345, 80, 1001, 2001,                    _PSH_ACK, req)))
    buf.write(_pkt(T, 500, "10.0.0.1", "192.168.1.1", 6, _tcp_seg(80, 12345, 2001, 1001 + len(req),         _PSH_ACK, resp)))
    buf.write(_pkt(T, 600, "192.168.1.1", "10.0.0.1", 6, _tcp_seg(12345, 80, 1001 + len(req), 2001 + len(resp),  _FIN_ACK)))
    buf.write(_pkt(T, 700, "10.0.0.1", "192.168.1.1", 6, _tcp_seg(80, 12345, 2001 + len(resp), 1001 + len(req) + 1, _FIN_ACK)))
    buf.write(_pkt(T, 800, "192.168.1.1", "10.0.0.1", 6, _tcp_seg(12345, 80, 1001 + len(req) + 1, 2001 + len(resp) + 1, _ACK)))

    # ── flow 2: SYN-heavy (port scan retries) — 5 SYNs, 1 RST-ACK ────────
    # Same 4-tuple → single conv; syn_count(5) > ack_count(1) for the RST-ACK.
    for i in range(5):
        buf.write(_pkt(T + 1, i * 200_000, "192.168.1.2", "10.0.0.2", 6,
                       _tcp_seg(54321, 22, 3000, 0, _SYN)))
    buf.write(_pkt(T + 1, 5 * 200_000, "10.0.0.2", "192.168.1.2", 6,
                   _tcp_seg(22, 54321, 4000, 3001, _RST_ACK)))

    # ── flow 3: UDP DNS (1 query, 1 response) ─────────────────────────────
    dns_q = b"\x00\x01" + b"\x00" * 10
    dns_r = b"\x00\x01" + b"\x00" * 20
    buf.write(_pkt(T + 2, 0,   "192.168.1.1", "8.8.8.8",     17, _udp_seg(54322, 53, dns_q)))
    buf.write(_pkt(T + 2, 100, "8.8.8.8",     "192.168.1.1", 17, _udp_seg(53, 54322, dns_r)))

    return buf.getvalue()


@pytest.fixture(scope="session")
def fixture_pcap_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("pcap") / "fixture.pcap"
    p.write_bytes(build_fixture_pcap())
    return p


# ── Initiator-order pcap fixtures for orientation tests ───────────────────
#
# Both pcaps carry one TCP flow: 10.0.0.5:12345 → 10.0.0.9:80.
# fwd: SYN from 10.0.0.5 appears first → tshark likely lists 10.0.0.5 as A.
# rev: the SYN-ACK-like packet from 10.0.0.9 appears first → tshark may list
#      10.0.0.9 as A. After canonicalization, both must have the same src.

def _initiator_fwd() -> bytes:
    """10.0.0.5 sends SYN first; 10.0.0.9 replies."""
    buf = io.BytesIO()
    buf.write(_pcap_header())
    T = 1_700_001_000
    data = b"HELLO"
    buf.write(_pkt(T, 0,   "10.0.0.5", "10.0.0.9", 6, _tcp_seg(12345, 80, 1000, 0,    _SYN)))
    buf.write(_pkt(T, 100, "10.0.0.9", "10.0.0.5", 6, _tcp_seg(80, 12345, 2000, 1001, _SYN_ACK)))
    buf.write(_pkt(T, 200, "10.0.0.5", "10.0.0.9", 6, _tcp_seg(12345, 80, 1001, 2001, _ACK)))
    buf.write(_pkt(T, 300, "10.0.0.5", "10.0.0.9", 6, _tcp_seg(12345, 80, 1001, 2001, _PSH_ACK, data)))
    buf.write(_pkt(T, 400, "10.0.0.9", "10.0.0.5", 6, _tcp_seg(80, 12345, 2001, 1001+len(data), _ACK)))
    return buf.getvalue()


def _initiator_rev() -> bytes:
    """Same flow but 10.0.0.9's packets appear first in the capture file,
    so tshark may list 10.0.0.9 as the A-side in conv output."""
    buf = io.BytesIO()
    buf.write(_pcap_header())
    T = 1_700_001_000
    data = b"HELLO"
    # Write 10.0.0.9 packets first (same logical flow, reordered in file)
    buf.write(_pkt(T, 0,   "10.0.0.9", "10.0.0.5", 6, _tcp_seg(80, 12345, 2000, 1001, _SYN_ACK)))
    buf.write(_pkt(T, 100, "10.0.0.9", "10.0.0.5", 6, _tcp_seg(80, 12345, 2001, 1001+len(data), _ACK)))
    # Then 10.0.0.5 packets
    buf.write(_pkt(T, 200, "10.0.0.5", "10.0.0.9", 6, _tcp_seg(12345, 80, 1000, 0,    _SYN)))
    buf.write(_pkt(T, 300, "10.0.0.5", "10.0.0.9", 6, _tcp_seg(12345, 80, 1001, 2001, _ACK)))
    buf.write(_pkt(T, 400, "10.0.0.5", "10.0.0.9", 6, _tcp_seg(12345, 80, 1001, 2001, _PSH_ACK, data)))
    return buf.getvalue()


@pytest.fixture(scope="session")
def initiator_pcap_fwd_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("orient") / "initiator_fwd.pcap"
    p.write_bytes(_initiator_fwd())
    return p


@pytest.fixture(scope="session")
def initiator_pcap_rev_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("orient") / "initiator_rev.pcap"
    p.write_bytes(_initiator_rev())
    return p
