#!/usr/bin/env python3
"""Standalone CLI for generating synthetic attack flows.

Posts flow batches directly to the ADNS /ingest endpoint so the full
detection pipeline scores them — without touching the UI.

Usage examples
--------------
# One-shot batch of 50 DDoS flows
python core/attack_generator.py --type ddos --count 50

# Stream injection flows for 2 minutes, one batch per second
python core/attack_generator.py --type injection --duration 120 --interval 1

# Point at a non-default API
python core/attack_generator.py --type scanning --api-url http://10.0.0.5:5000

Supported attack types: attack, scanning, dos, ddos, injection
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime, timedelta, timezone

try:
    import urllib.request
    import urllib.error
except ImportError:
    pass

DEFAULT_COUNTS = {
    "attack": 40,
    "scanning": 80,
    "dos": 60,
    "ddos": 120,
    "injection": 40,
}

VALID_TYPES = set(DEFAULT_COUNTS)


# ---------------------------------------------------------------------------
# Flow generation (mirrors generate_attack_flows in api/app.py)
# ---------------------------------------------------------------------------

def _pattern_ip(pattern: str, rng: random.Random) -> str:
    parts = pattern.split(".")
    octets = []
    for part in parts:
        if part in {"x", "*"}:
            octets.append(str(rng.randint(1, 254)))
        elif part == "y":
            octets.append(str(rng.randint(0, 99)))
        else:
            octets.append(part)
    while len(octets) < 4:
        octets.append(str(rng.randint(1, 254)))
    return ".".join(octets[:4])


def _make_extra(proto: str, src_port: int, dst_port: int, byte_count: int,
                service_hint: str | None = None, rng: random.Random | None = None) -> dict:
    rng = rng or random.Random()
    total = max(0, int(byte_count))
    reply = int(total * rng.uniform(0.05, 0.3))
    return {
        "src_port": src_port,
        "dst_port": dst_port,
        "service": service_hint or proto.lower(),
        "duration": rng.uniform(2.0, 15.0),
        "src_bytes": total,
        "dst_bytes": reply,
        "src_pkts": max(3, total // 600),
        "dst_pkts": max(3, reply // 600),
    }


def generate_flows(kind: str, count: int) -> list[dict]:
    """Return a list of flow dicts ready for /ingest."""
    if kind not in VALID_TYPES:
        raise ValueError(f"unknown attack type '{kind}'. Valid: {', '.join(sorted(VALID_TYPES))}")

    rng = random.Random()
    now = datetime.now(timezone.utc)
    flows: list[dict] = []

    for i in range(count):
        if kind == "ddos":
            dst = rng.choice(["198.51.100.42", "198.51.100.47", "203.0.113.10"])
            src = _pattern_ip("10.x.y.x", rng)
            bytes_val = rng.randint(180_000, 520_000)
            offset = rng.uniform(0, 90)
            src_port = rng.randint(1024, 65000)
            extra = _make_extra("tcp", src_port, rng.choice([80, 443]), bytes_val, "http", rng)
            proto = "TCP"
        elif kind == "dos":
            src = rng.choice(["10.0.5.33", "10.0.5.34"])
            dst = rng.choice(["203.0.113.55", "203.0.113.56"])
            bytes_val = rng.randint(90_000, 180_000)
            offset = rng.uniform(0, 60)
            src_port = rng.randint(10_000, 60000)
            extra = _make_extra("tcp", src_port, 443, bytes_val, "https", rng)
            proto = "TCP"
        elif kind == "scanning":
            src = rng.choice(["172.16.8.4", "172.16.8.5"])
            dst = f"192.168.{rng.randint(1, 10)}.{(i % 200) + 1}"
            proto = rng.choice(["UDP", "TCP"])
            bytes_val = rng.randint(800, 5000)
            offset = rng.uniform(0, 180)
            dst_port = rng.randint(1, 1024)
            src_port = rng.randint(2000, 9000)
            extra = _make_extra(proto.lower(), src_port, dst_port, bytes_val, "scan", rng)
        elif kind == "injection":
            src = rng.choice(["10.12.11.7", "10.12.11.8"])
            dst = _pattern_ip("203.0.113.x", rng)
            bytes_val = rng.randint(4_000, 18_000)
            offset = rng.uniform(0, 45)
            dst_port = rng.choice([1433, 3306, 5432, 9200])
            src_port = rng.randint(30000, 65000)
            extra = _make_extra("tcp", src_port, dst_port, bytes_val, "sql", rng)
            extra["http_method"] = "POST"
            extra["http_uri"] = "/login"
            proto = "TCP"
        else:  # "attack"
            src = rng.choice(["10.0.5.33", "10.0.5.34"])
            dst = _pattern_ip("203.0.113.x", rng)
            bytes_val = rng.randint(160_000, 360_000)
            offset = rng.uniform(0, 120)
            src_port = rng.randint(20000, 60000)
            extra = _make_extra("tcp", src_port, 443, bytes_val, "https", rng)
            proto = "TCP"

        ts = (now - timedelta(seconds=offset)).isoformat()
        flows.append({
            "timestamp": ts,
            "src_ip": src,
            "dst_ip": dst,
            "proto": proto,
            "bytes": max(0, int(bytes_val)),
            **extra,
        })

    return flows


# ---------------------------------------------------------------------------
# HTTP POST helper (stdlib only — no requests dep)
# ---------------------------------------------------------------------------

def post_flows(api_url: str, flows: list[dict]) -> dict:
    url = api_url.rstrip("/") + "/ingest"
    body = json.dumps(flows).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {raw}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection failed: {exc.reason}") from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic attack flows and POST them to ADNS /ingest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--type", "-t",
        dest="attack_type",
        default="ddos",
        choices=sorted(VALID_TYPES),
        help="Attack scenario to simulate (default: ddos)",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=None,
        help="Flows per batch (default: per-type default)",
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Stream batches for this many seconds (0 = one-shot, default: 0)",
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Seconds between batches in streaming mode (default: 1.0)",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="ADNS API base URL (default: $API_URL or http://127.0.0.1:5000)",
    )
    return parser


def main() -> None:
    import os
    parser = build_parser()
    args = parser.parse_args()

    api_url = args.api_url or os.environ.get("API_URL", "http://127.0.0.1:5000")
    count = args.count if args.count is not None else DEFAULT_COUNTS[args.attack_type]
    count = max(1, min(count, 500))
    duration = max(0, args.duration)
    interval = max(0.1, args.interval)

    print(f"[attack_generator] type={args.attack_type}  count={count}  "
          f"duration={duration}s  api={api_url}")

    if duration == 0:
        flows = generate_flows(args.attack_type, count)
        try:
            result = post_flows(api_url, flows)
        except RuntimeError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            sys.exit(1)
        ingested = result.get("ingested", "?")
        print(f"[ok] ingested={ingested}")
        return

    # Streaming mode
    deadline = time.time() + duration
    total = 0
    batch_num = 0
    try:
        while time.time() < deadline:
            batch_num += 1
            flows = generate_flows(args.attack_type, count)
            try:
                result = post_flows(api_url, flows)
                ingested = result.get("ingested", len(flows))
                total += ingested
                print(f"[batch {batch_num}] ingested={ingested}  total={total}")
            except RuntimeError as exc:
                print(f"[batch {batch_num}] error: {exc}", file=sys.stderr)
            remaining = deadline - time.time()
            if remaining > 0:
                time.sleep(min(interval, remaining))
    except KeyboardInterrupt:
        print(f"\n[stopped] total ingested ≈ {total}")
        return

    print(f"[done] total ingested ≈ {total}")


if __name__ == "__main__":
    main()
