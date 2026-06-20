"""CLI entry point: python -m adns_flows --pcap FILE --out CSV"""
from __future__ import annotations

import argparse
import sys

from .assemble import extract_flows, flows_to_dataframe
from .extract import find_tshark


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m adns_flows",
        description="Extract bidirectional flow features from a pcap or live capture.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pcap", metavar="FILE", help="read from pcap file")
    source.add_argument("--iface", metavar="NAME", help="capture live from interface")
    parser.add_argument("--out", metavar="CSV", required=True, help="output CSV path")
    parser.add_argument(
        "--window", type=int, default=60, metavar="SEC",
        help="capture window in seconds for live mode (default: 60)",
    )
    parser.add_argument(
        "--tshark", metavar="PATH",
        help="explicit path to tshark binary (overrides auto-detect)",
    )
    args = parser.parse_args()

    tshark_bin = args.tshark or find_tshark()
    if not tshark_bin:
        sys.exit(
            "tshark not found. Install Wireshark or set the TSHARK_BIN env var."
        )

    flows = extract_flows(
        tshark_bin,
        pcap=args.pcap,
        iface=args.iface,
        window_sec=args.window,
    )
    df = flows_to_dataframe(flows)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} flows → {args.out}")


if __name__ == "__main__":
    main()
