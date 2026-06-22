"""CLI entry point: python -m adns_flows --pcap FILE --out CSV"""
from __future__ import annotations

import argparse
import sys

from .extract_nfstream import extract_flows_nfstream, flows_to_dataframe_nfstream


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m adns_flows",
        description="Extract bidirectional flow features from a pcap file.",
    )
    parser.add_argument("--pcap", metavar="FILE", required=True, help="read from pcap file")
    parser.add_argument("--out", metavar="CSV", required=True, help="output CSV path")
    parser.add_argument(
        "--n-meters", type=int, default=1, metavar="N",
        help="NFStream worker count (default: 1)",
    )
    args = parser.parse_args()

    flows = extract_flows_nfstream(args.pcap, n_meters=args.n_meters)
    df = flows_to_dataframe_nfstream(flows)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} flows → {args.out}")


if __name__ == "__main__":
    main()
