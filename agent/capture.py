#!/usr/bin/env python3
"""Capture network flows via tshark on eth0 and forward batches to /ingest."""

import logging
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

import requests


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


API_URL = _env_str("API_URL", "http://127.0.0.1:5000/ingest")
_DEFAULT_TSHARK = (
    r"C:\Program Files\Wireshark\tshark.exe"
    if sys.platform == "win32"
    else "/usr/bin/tshark"
)
TSHARK_BIN = _env_str("TSHARK_BIN", _DEFAULT_TSHARK)
INTERFACE = _env_str("INTERFACE", "eth0")
BATCH_SIZE = _env_int("BATCH_SIZE", 50)
POST_INTERVAL = _env_float("POST_INTERVAL", 2.0)  # seconds
RETRY_DELAY = _env_float("RETRY_DELAY", 3.0)  # seconds
DEFAULT_DURATION = 0.01

TSHARK_FIELDS = [
    "frame.time_epoch",
    "ip.src",
    "ip.dst",
    "ip.proto",
    "frame.len",
    "tcp.srcport",
    "tcp.dstport",
    "udp.srcport",
    "udp.dstport",
    "dns.qry.name",
    "dns.qry.type",
    "dns.qry.class",
    "dns.flags.rcode",
    "http.request.method",
    "http.request.full_uri",
    "http.user_agent",
    "http.response.code",
    "http.content_length",
    "ssl.handshake.version",
    "ssl.handshake.ciphersuite",
]

PROTOCOL_MAP = {
    "1": "ICMP",
    "6": "TCP",
    "17": "UDP",
    "47": "GRE",
    "50": "ESP",
    "51": "AH",
    "58": "ICMPV6",
    "132": "SCTP",
}

SERVICE_PORT_MAP = {
    20: "ftp",
    21: "ftp",
    22: "ssh",
    23: "telnet",
    25: "smtp",
    53: "dns",
    67: "dhcp",
    68: "dhcp",
    80: "http",
    110: "pop3",
    123: "ntp",
    135: "rpc",
    143: "imap",
    161: "snmp",
    389: "ldap",
    443: "https",
    445: "smb",
    465: "smtps",
    993: "imaps",
    995: "pop3s",
    1433: "mssql",
    1521: "oracle",
    3306: "mysql",
    3389: "rdp",
    5060: "sip",
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [agent] %(message)s",
    )


def build_tshark_cmd() -> List[str]:
    cmd = [TSHARK_BIN, "-i", INTERFACE, "-T", "fields"]
    for field in TSHARK_FIELDS:
        cmd.extend(["-e", field])
    cmd.extend(
        [
            "-Y",
            "ip",
            "-l",
            "-E",
            "separator=\t",
            "-E",
            "header=n",
        ]
    )
    return cmd


def _safe_float(value: str, fallback: float) -> float:
    if not value:
        return fallback
    try:
        return float(value)
    except ValueError:
        return fallback


def _safe_int(value: str) -> Optional[int]:
    if not value:
        return None
    try:
        if value.lower().startswith("0x"):
            return int(value, 16)
        return int(value)
    except ValueError:
        digits = "".join(ch for ch in value if ch.isdigit())
        return int(digits) if digits else None


def _normalize_proto(value: str) -> str:
    if not value:
        return "OTHER"
    value = value.strip()
    if not value:
        return "OTHER"
    if value.isdigit():
        return PROTOCOL_MAP.get(value, f"PROTO_{value}")
    return value.upper()


def _infer_service(proto: str, src_port: Optional[int], dst_port: Optional[int], dns_query: Optional[str], http_method: Optional[str], ssl_version: Optional[int]) -> Optional[str]:
    if http_method:
        return "http"
    if ssl_version is not None or (dst_port in {443, 8443}):
        return "https"
    if dns_query or (dst_port == 53):
        return "dns"
    port = dst_port or src_port
    if port:
        return SERVICE_PORT_MAP.get(port, proto.lower())
    return proto.lower()


def _record_from_parts(parts: List[str]) -> Optional[Dict]:
    if len(parts) < len(TSHARK_FIELDS):
        parts = parts + [""] * (len(TSHARK_FIELDS) - len(parts))
    elif len(parts) > len(TSHARK_FIELDS):
        parts = parts[: len(TSHARK_FIELDS)]

    ts = _safe_float(parts[0], time.time())
    src = parts[1] or ""
    dst = parts[2] or ""
    proto = _normalize_proto(parts[3])
    length = max(0, _safe_int(parts[4]) or 0)

    tcp_src = _safe_int(parts[5])
    tcp_dst = _safe_int(parts[6])
    udp_src = _safe_int(parts[7])
    udp_dst = _safe_int(parts[8])

    src_port = tcp_src or udp_src or 0
    dst_port = tcp_dst or udp_dst or 0

    dns_query = parts[9].strip() or ""
    dns_qtype = _safe_int(parts[10])
    dns_qclass = _safe_int(parts[11])
    dns_rcode = _safe_int(parts[12])

    http_method = parts[13].strip() or ""
    http_uri = parts[14].strip() or ""
    http_user_agent = parts[15].strip() or ""
    http_status = _safe_int(parts[16])
    http_content_len = _safe_int(parts[17])

    ssl_version = _safe_int(parts[18])
    ssl_cipher = parts[19].strip() or ""

    service = _infer_service(proto, src_port, dst_port, dns_query or None, http_method or None, ssl_version)

    rec: Dict = {
        "ts": ts,
        "src_ip": src,
        "dst_ip": dst,
        "proto": proto,
        "bytes": length,
        "score": 0.0,
        "duration": DEFAULT_DURATION,
        "src_bytes": length,
        "dst_bytes": 0,
        "src_pkts": 1,
        "dst_pkts": 0,
    }

    if src_port:
        rec["src_port"] = src_port
    if dst_port:
        rec["dst_port"] = dst_port
    if service:
        rec["service"] = service
    if dns_query:
        rec["dns_query"] = dns_query
    if dns_qtype is not None:
        rec["dns_qtype"] = dns_qtype
    if dns_qclass is not None:
        rec["dns_qclass"] = dns_qclass
    if dns_rcode is not None:
        rec["dns_rcode"] = dns_rcode

    if http_method:
        rec["http_method"] = http_method
    if http_uri:
        rec["http_uri"] = http_uri
    if http_user_agent:
        rec["http_user_agent"] = http_user_agent
    if http_status is not None:
        rec["http_status_code"] = http_status
    if http_content_len is not None:
        if http_method:
            rec["http_request_body_len"] = http_content_len
        elif http_status is not None:
            rec["http_response_body_len"] = http_content_len

    if ssl_version is not None:
        rec["ssl_version"] = ssl_version
    if ssl_cipher:
        rec["ssl_cipher"] = ssl_cipher

    return rec


def run() -> None:
    configure_logging()

    if not os.path.isfile(TSHARK_BIN):
        logging.error("tshark binary not found at %s; install tshark and retry.", TSHARK_BIN)
        return

    cmd = build_tshark_cmd()
    logging.info("starting tshark on %s using %s", INTERFACE, TSHARK_BIN)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        logging.info("spawned tshark process pid=%s", proc.pid)
    except FileNotFoundError:
        logging.error("tshark executable is missing; exiting.")
        return

    buf: List[dict] = []
    last_post = time.time()
    session = requests.Session()

    try:
        assert proc.stdout is not None  # for typing
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            rec = _record_from_parts(parts)
            if not rec:
                continue
            buf.append(rec)

            if len(buf) == 1:
                logging.debug(
                    "buffering new flow window starting with %s -> %s (%s %s)",
                    rec.get("src_ip", "?"),
                    rec.get("dst_ip", "?"),
                    rec.get("proto", "?"),
                    rec.get("bytes", 0),
                )

            now = time.time()
            if len(buf) >= BATCH_SIZE or (now - last_post) >= POST_INTERVAL:
                batch = buf
                buf = []
                if post_batch(batch, session):
                    last_post = now
                else:
                    buf = batch + buf
                    time.sleep(RETRY_DELAY)

    except KeyboardInterrupt:
        logging.info("interrupted, shutting down.")
    finally:
        if proc.poll() is None:
            proc.terminate()


def post_batch(batch: List[dict], session: requests.Session) -> bool:
    if not batch:
        return True

    try:
        resp = session.post(API_URL, json=batch, timeout=5)
        if resp.ok:
            logging.info("posted %d flows (status=%s)", len(batch), resp.status_code)
            return True

        logging.warning(
            "ingest endpoint responded with %s; will retry.", resp.status_code
        )
    except requests.RequestException as exc:
        logging.warning("error posting batch: %s; will retry.", exc)

    return False


if __name__ == "__main__":
    run()
