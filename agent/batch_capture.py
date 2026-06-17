#!/usr/bin/env python3
"""
Batch capture agent: ring-buffer pcap + two-pass tshark processing.

Runs tshark in ring-buffer mode writing a new pcap every BATCH_WINDOW_SECONDS.
For each completed pcap:
  Pass 1: tshark -z conv,tcp + conv,udp  →  real duration, directional bytes, packet counts
  Pass 2: tshark -T fields               →  app-layer data (DNS, HTTP, SSL)

Merges both passes on (src_ip, src_port, dst_ip, dst_port) and POSTs enriched
flow records to /ingest_batch. Flows scored server-side with real feature values
instead of the per-packet hardcoded defaults.

Environment variables:
  INTERFACE              Network interface (default: eth0)
  TSHARK_BIN             Path to tshark binary
  BATCH_WINDOW_SECONDS   Ring-buffer rotation interval in seconds (default: 15)
  BATCH_DIR              Directory for temporary pcap files (default: OS temp)
  BATCH_API_URL          POST target (default: http://127.0.0.1:5000/ingest_batch)
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = os.environ.get("BATCH_API_URL", "http://127.0.0.1:5000/ingest_batch")
INTERFACE = os.environ.get("INTERFACE", "eth0")
BATCH_WINDOW_SECONDS = int(os.environ.get("BATCH_WINDOW_SECONDS", "15"))
BATCH_DIR = os.environ.get("BATCH_DIR", "")
MAX_PCAP_FILES = 4  # ring buffer depth; tshark deletes oldest automatically

_DEFAULT_TSHARK = (
    r"C:\Program Files\Wireshark\tshark.exe"
    if sys.platform == "win32"
    else "/usr/bin/tshark"
)
TSHARK_BIN = os.environ.get("TSHARK_BIN", _DEFAULT_TSHARK)

# ---------------------------------------------------------------------------
# tshark field list for pass 2 (per-packet app-layer dissection)
# ---------------------------------------------------------------------------

PASS2_FIELDS = [
    "frame.time_epoch",
    "ip.src",
    "ip.dst",
    "ip.proto",
    "tcp.srcport",
    "tcp.dstport",
    "udp.srcport",
    "udp.dstport",
    "dns.qry.name",
    "dns.qry.type",
    "dns.qry.class",
    "dns.flags.rcode",
    "dns.flags.authoritative",
    "dns.flags.recdesired",
    "dns.flags.recavail",
    "http.request.method",
    "http.request.full_uri",
    "http.user_agent",
    "http.response.code",
    "http.content_length",
    "http.referer",
    "http.request.version",
    "http.content_type",
    "ssl.handshake.version",
    "ssl.handshake.ciphersuite",
]

# ---------------------------------------------------------------------------
# Static lookup tables
# ---------------------------------------------------------------------------

PROTO_MAP = {
    "1": "ICMP", "6": "TCP", "17": "UDP",
    "47": "GRE", "50": "ESP", "51": "AH", "58": "ICMPV6", "132": "SCTP",
}

SERVICE_PORTS = {
    20: "ftp", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
    53: "dns", 67: "dhcp", 68: "dhcp", 80: "http", 110: "pop3",
    123: "ntp", 135: "rpc", 143: "imap", 161: "snmp", 389: "ldap",
    443: "https", 445: "smb", 465: "smtps", 993: "imaps", 995: "pop3s",
    1433: "mssql", 1521: "oracle", 3306: "mysql", 3389: "rdp", 5060: "sip",
}

# Matches one data row from tshark -z conv,tcp or -z conv,udp output.
# Column order: A_IP:A_PORT <-> B_IP:B_PORT | <-(B→A) frames bytes | | ->(A→B) frames bytes | | total | rel_start | duration
CONV_RE = re.compile(
    r"(\S+):(\d+)\s+<->\s+(\S+):(\d+)\s*"
    r"\|\s*(\d+)\s+(\d+)\s*\|\s*"
    r"\|\s*(\d+)\s+(\d+)\s*\|\s*"
    r"\|\s*\d+\s+\d+\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_proto(val: str) -> str:
    v = (val or "").strip()
    if not v:
        return "OTHER"
    return PROTO_MAP.get(v, v.upper()) if v.isdigit() else v.upper()


def _safe_int(v) -> Optional[int]:
    if v is None or v == "":
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(s, 16 if s.lower().startswith("0x") else 10)
    except ValueError:
        digits = "".join(c for c in s if c.isdigit())
        return int(digits) if digits else None


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v) if v not in (None, "") else default
    except (ValueError, TypeError):
        return default


def _infer_service(proto: str, src_port: Optional[int], dst_port: Optional[int],
                   dns_query: Optional[str], http_method: Optional[str],
                   ssl_version: Optional[int]) -> str:
    if http_method:
        return "http"
    if ssl_version is not None or dst_port in {443, 8443}:
        return "https"
    if dns_query or dst_port == 53:
        return "dns"
    port = dst_port or src_port
    return SERVICE_PORTS.get(port, proto.lower()) if port else proto.lower()


# ---------------------------------------------------------------------------
# pcap directory management
# ---------------------------------------------------------------------------

def _get_batch_dir() -> Path:
    d = Path(BATCH_DIR) if BATCH_DIR else Path(tempfile.gettempdir()) / "adns_batch"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _find_completed_pcaps(batch_dir: Path) -> list[Path]:
    """Return all pcap files except the one currently being written (newest mtime)."""
    pcaps = sorted(batch_dir.glob("cap_*.pcap"), key=lambda p: p.stat().st_mtime)
    if len(pcaps) < 2:
        return []
    return pcaps[:-1]


# ---------------------------------------------------------------------------
# Pass 1: conversation-level flow metrics via -z conv
# ---------------------------------------------------------------------------

def _run_conv_stats(tshark_bin: str, pcap: Path) -> list[dict]:
    try:
        result = subprocess.run(
            [tshark_bin, "-r", str(pcap), "-q", "-z", "conv,tcp", "-z", "conv,udp"],
            capture_output=True, text=True, timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except subprocess.TimeoutExpired:
        logger.warning("conv stats timed out for %s", pcap.name)
        return []
    except Exception as exc:
        logger.warning("conv stats failed for %s: %s", pcap.name, exc)
        return []

    flows: list[dict] = []
    current_proto = "TCP"
    for line in result.stdout.splitlines():
        if "TCP Conversations" in line:
            current_proto = "TCP"
        elif "UDP Conversations" in line:
            current_proto = "UDP"
        m = CONV_RE.search(line)
        if not m:
            continue
        (a_ip, a_port, b_ip, b_port,
         frames_ba, bytes_ba,   # <- column: B→A (responder→initiator)
         frames_ab, bytes_ab,   # -> column: A→B (initiator→responder)
         rel_start, duration) = m.groups()
        flows.append({
            "proto": current_proto,
            "src_ip": a_ip,
            "src_port": int(a_port),
            "dst_ip": b_ip,
            "dst_port": int(b_port),
            "src_bytes": int(bytes_ab),   # initiator sent
            "dst_bytes": int(bytes_ba),   # responder replied
            "src_pkts": int(frames_ab),
            "dst_pkts": int(frames_ba),
            "duration": float(duration),
            "rel_start": float(rel_start),
        })
    return flows


# ---------------------------------------------------------------------------
# Pass 2: per-packet app-layer field extraction
# ---------------------------------------------------------------------------

def _run_field_pass(tshark_bin: str, pcap: Path) -> list[dict]:
    cmd = [tshark_bin, "-r", str(pcap), "-T", "fields", "-Y", "ip"]
    for field in PASS2_FIELDS:
        cmd.extend(["-e", field])
    cmd.extend(["-E", "separator=\t", "-E", "header=n"])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except subprocess.TimeoutExpired:
        logger.warning("field pass timed out for %s", pcap.name)
        return []
    except Exception as exc:
        logger.warning("field pass failed for %s: %s", pcap.name, exc)
        return []

    n = len(PASS2_FIELDS)
    packets: list[dict] = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < n:
            parts += [""] * (n - len(parts))
        elif len(parts) > n:
            parts = parts[:n]

        src_ip = parts[1].strip()
        dst_ip = parts[2].strip()
        if not src_ip or not dst_ip:
            continue

        proto = _norm_proto(parts[3])
        src_port = _safe_int(parts[4]) or _safe_int(parts[6]) or 0
        dst_port = _safe_int(parts[5]) or _safe_int(parts[7]) or 0

        pkt: dict = {
            "ts": _safe_float(parts[0], time.time()),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "proto": proto,
            "src_port": src_port,
            "dst_port": dst_port,
        }

        # Map field positions to record keys
        field_extractions = [
            ("dns_query",    parts[8].strip()),
            ("dns_qtype",    _safe_int(parts[9])),
            ("dns_qclass",   _safe_int(parts[10])),
            ("dns_rcode",    _safe_int(parts[11])),
            ("dns_AA",       _safe_int(parts[12])),
            ("dns_RD",       _safe_int(parts[13])),
            ("dns_RA",       _safe_int(parts[14])),
            ("http_method",  parts[15].strip()),
            ("http_uri",     parts[16].strip()),
            ("http_user_agent", parts[17].strip()),
            ("http_status_code", _safe_int(parts[18])),
            ("http_content_length", _safe_int(parts[19])),
            ("http_referrer", parts[20].strip()),
            ("http_version", parts[21].strip()),
            ("http_content_type", parts[22].strip()),
            ("ssl_version",  _safe_int(parts[23])),
            ("ssl_cipher",   parts[24].strip()),
        ]
        for key, val in field_extractions:
            if val is not None and val != "":
                pkt[key] = val

        packets.append(pkt)

    return packets


# ---------------------------------------------------------------------------
# Merge pass 1 + pass 2 into enriched flow records
# ---------------------------------------------------------------------------

def _build_app_index(packets: list[dict]) -> dict:
    """Index first-seen app-layer field values by both directions of each connection."""
    index: dict = {}
    skip = {"ts", "src_ip", "dst_ip", "proto", "src_port", "dst_port"}
    for pkt in packets:
        s = (pkt.get("src_ip", ""), pkt.get("src_port") or 0)
        d = (pkt.get("dst_ip", ""), pkt.get("dst_port") or 0)
        for key in ((s[0], s[1], d[0], d[1]), (d[0], d[1], s[0], s[1])):
            if key not in index:
                index[key] = {}
            for k, v in pkt.items():
                if k in skip or v is None or v == "":
                    continue
                index[key].setdefault(k, v)
    return index


def _merge_flows(conv_flows: list[dict], app_index: dict, pcap_mtime: float) -> list[dict]:
    result: list[dict] = []
    for flow in conv_flows:
        key = (flow["src_ip"], flow["src_port"], flow["dst_ip"], flow["dst_port"])
        app = app_index.get(key, {})

        total_bytes = flow["src_bytes"] + flow["dst_bytes"]
        # Approximate flow start timestamp from pcap file end time
        flow_ts = pcap_mtime - BATCH_WINDOW_SECONDS + flow["rel_start"]

        rec: dict = {
            "ts": flow_ts,
            "src_ip": flow["src_ip"],
            "dst_ip": flow["dst_ip"],
            "proto": flow["proto"],
            "bytes": total_bytes,
            "src_bytes": flow["src_bytes"],
            "dst_bytes": flow["dst_bytes"],
            "src_pkts": flow["src_pkts"],
            "dst_pkts": flow["dst_pkts"],
            "duration": flow["duration"],
            "src_port": flow["src_port"],
            "dst_port": flow["dst_port"],
        }

        # Merge app-layer fields (don't overwrite flow-level fields)
        for k, v in app.items():
            rec.setdefault(k, v)

        # Derive service label
        rec["service"] = _infer_service(
            flow["proto"],
            flow["src_port"],
            flow["dst_port"],
            app.get("dns_query"),
            app.get("http_method"),
            app.get("ssl_version"),
        )

        # Map http_content_type → request or response mime field
        content_type = app.get("http_content_type", "")
        if content_type:
            if app.get("http_method"):
                rec.setdefault("http_orig_mime_types", content_type)
            elif app.get("http_status_code"):
                rec.setdefault("http_resp_mime_types", content_type)

        # Derive dns_rejected from rcode
        dns_rcode = app.get("dns_rcode")
        if dns_rcode is not None:
            rec.setdefault("dns_rejected", 1 if dns_rcode != 0 else 0)

        result.append(rec)
    return result


# ---------------------------------------------------------------------------
# Process one completed pcap file
# ---------------------------------------------------------------------------

def _process_pcap(tshark_bin: str, pcap: Path, session: requests.Session) -> None:
    logger.info("processing %s", pcap.name)
    pcap_mtime = pcap.stat().st_mtime

    conv_flows = _run_conv_stats(tshark_bin, pcap)
    if not conv_flows:
        logger.debug("no flows in %s", pcap.name)
        try:
            pcap.unlink()
        except OSError:
            pass
        return

    packets = _run_field_pass(tshark_bin, pcap)
    app_index = _build_app_index(packets)
    flows = _merge_flows(conv_flows, app_index, pcap_mtime)

    if flows:
        try:
            resp = session.post(API_URL, json=flows, timeout=15)
            if resp.ok:
                logger.info("posted %d batch flows from %s", len(flows), pcap.name)
            else:
                logger.warning("ingest_batch returned %s for %s", resp.status_code, pcap.name)
        except requests.RequestException as exc:
            logger.warning("failed to post batch flows: %s", exc)

    try:
        pcap.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [batch] %(message)s",
    )

    if not os.path.isfile(TSHARK_BIN):
        logger.error("tshark not found at %s; set TSHARK_BIN env var", TSHARK_BIN)
        return

    batch_dir = _get_batch_dir()
    pcap_base = str(batch_dir / "cap")
    logger.info(
        "starting on interface=%s  window=%ds  dir=%s",
        INTERFACE, BATCH_WINDOW_SECONDS, batch_dir,
    )

    writer_cmd = [
        TSHARK_BIN,
        "-i", INTERFACE,
        "-b", f"duration:{BATCH_WINDOW_SECONDS}",
        "-b", f"files:{MAX_PCAP_FILES}",
        "-w", pcap_base,
        "-q",
    ]
    try:
        writer = subprocess.Popen(
            writer_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except Exception as exc:
        logger.error("failed to start tshark writer: %s", exc)
        return

    logger.info("tshark writer pid=%d", writer.pid)
    session = requests.Session()

    try:
        while writer.poll() is None:
            for pcap in _find_completed_pcaps(batch_dir):
                _process_pcap(TSHARK_BIN, pcap, session)
            time.sleep(2)
    except KeyboardInterrupt:
        logger.info("interrupted, shutting down")
    finally:
        if writer.poll() is None:
            writer.terminate()
            try:
                writer.wait(timeout=5)
            except subprocess.TimeoutExpired:
                writer.kill()
        for pcap in batch_dir.glob("cap_*.pcap"):
            try:
                pcap.unlink()
            except OSError:
                pass
        logger.info("batch capture stopped")


if __name__ == "__main__":
    run()
