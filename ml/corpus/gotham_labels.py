"""
Gotham Dataset 2025 label adapter for the ADNS corpus pipeline.

Step 0 schema findings
-----------------------
Gotham metadata is NOT a per-event start/end/src/dst/port/type log.
No timestamps exist anywhere in the metadata.

  metadata-benign.json   : dict keyed by device family
      columns: device_ip (list), server_ip (list), iot_application, bidirectional, label
  metadata-<attack>.json : list of IP-pair rules
      columns: source_ip, destination_ip, [protocol], [source_port],
               [destination_port], label
      wildcards: "192.168.x.x" = any IP in the subnet

Ground truth is DIRECTORY-ENCODED:
  raw/benign/*.pcap            → all flows benign  (label=0)
  raw/malicious/<type>/*.pcap  → all flows attack   (label=1)

Consequence: time-window matching (UNSW approach) is impossible. Labeling is
PCAP-level. Every flow in a malicious PCAP is treated as an attack flow with
attack_cat derived from the directory name.

Attack-cat normalization
------------------------
Gotham directory      raw sub-labels                          ADNS attack_cat
──────────────────    ──────────────────────────────────────  ──────────────────
network-scanning      TCP Scan                                scanning
coap-amplificator     UDP Scan, CoAP Amplification            coap_amplification
merlin                Merlin {TCP,UDP,ICMP} Flooding,          merlin_dos
                      Merlin C&C Communication
mirai-dos             Mirai {TCP,UDP,GRE} Flooding,            mirai_dos
                      Mirai C&C Communication
mirai-infection       TCP Scan, Telnet Brute Force, Reporting, mirai_infection
                      Ingress Tool Transfer, File Download,
                      C&C Communication

UNSW ↔ Gotham vocabulary mapping
---------------------------------
  UNSW attack_cat    Gotham equivalent
  ─────────────────  ──────────────────────────────────────────────────
  Reconnaissance     scanning         (network-scanning masscan scans)
  DoS                mirai_dos        (Mirai TCP/UDP/GRE floods)
  Backdoor           mirai_infection  (partial: infection + C2 channel)
  (no equivalent)    coap_amplification  — Gotham-specific
  (no equivalent)    merlin_dos          — Gotham-specific (Merlin C2+DDoS)

Gotham-specific categories with no UNSW counterpart:
  coap_amplification — CoAP reflective amplification; exploits CoAP "observe"
                       mechanism; requires app-layer inspection (deferred v2)
  merlin_dos         — Merlin post-exploitation framework C2 + custom floods;
                       C2 channel may be TLS-wrapped (deferred v2)
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

# ── attack-cat normalization ───────────────────────────────────────────────

ATTACK_CAT_MAP: dict[str, str] = {
    "network-scanning":  "scanning",
    "coap-amplificator": "coap_amplification",
    "merlin":            "merlin_dos",
    "mirai-dos":         "mirai_dos",
    "mirai-infection":   "mirai_infection",
}

# Primary attacker IP(s) per attack scenario (for prefer_src orientation).
# Sourced from metadata JSON files in the reference labeller; only non-wildcard
# IPs are listed — wildcard rules ("192.168.x.x") cannot pin a specific endpoint.
# For each PCAP, _apply_labels_gotham() uses the first listed IP that matches
# one of the flow's endpoints; if none match, default orientation applies.
ATTACKER_IPS: dict[str, list[str]] = {
    # Masscan TCP scanner
    "network-scanning":  ["192.168.35.10"],
    # UDP scanner + CoAP amplification initiator
    "coap-amplificator": ["192.168.35.10", "192.168.0.200"],
    # Merlin C2 controller + flood bots
    "merlin":            [
        "192.168.34.10",   # Merlin C2
        "192.168.20.10",   # flood bot
        "192.168.17.15",   # flood bot
        "192.168.17.10",   # flood bot
    ],
    # Mirai C&C + DDoS bot
    "mirai-dos":         [
        "192.168.33.10",   # Mirai C&C
        "192.168.20.10",   # DDoS bot
        "192.168.17.10",   # DDoS bot
    ],
    # Mirai infection chain: scanner → bruteforcer → C2 → reporter → dropper
    "mirai-infection":   [
        "192.168.0.100",   # scanner/bruteforcer
        "192.168.33.10",   # C&C
        "192.168.33.11",   # reporter
        "192.168.33.12",   # ingress tool transfer
        "192.168.33.13",   # file download
    ],
}


@dataclasses.dataclass
class GothamPcapSpec:
    """Metadata for one Gotham PCAP file derived from its directory position."""
    pcap_path:    Path
    is_attack:    bool
    attack_cat:   str        # "" for benign; normalised cat for attack
    attacker_ips: list[str]  # empty for benign; used for prefer_src orientation


def load_gotham_corpus_spec(gotham_root: Path) -> list[GothamPcapSpec]:
    """Walk gotham_root and return one GothamPcapSpec per PCAP.

    Expected on-disk layout (preserved as-shipped, not flattened):
      <gotham_root>/raw/benign/*.pcap
      <gotham_root>/raw/malicious/<attack_type>/*.pcap

    PCAPs are discovered by glob (*.pcap); .pcapng is not used in this dataset.
    Results are sorted for deterministic ordering.

    Raises
    ------
    FileNotFoundError
        If <gotham_root>/raw does not exist.
    """
    raw_dir = Path(gotham_root) / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Gotham raw directory not found: {raw_dir}"
        )

    specs: list[GothamPcapSpec] = []

    # ── benign PCAPs ──────────────────────────────────────────────────────
    benign_dir = raw_dir / "benign"
    if benign_dir.is_dir():
        for pcap in sorted(benign_dir.glob("*.pcap")):
            specs.append(GothamPcapSpec(
                pcap_path=pcap,
                is_attack=False,
                attack_cat="",
                attacker_ips=[],
            ))

    # ── malicious PCAPs ───────────────────────────────────────────────────
    malicious_dir = raw_dir / "malicious"
    if malicious_dir.is_dir():
        for attack_dir in sorted(d for d in malicious_dir.iterdir() if d.is_dir()):
            dir_name   = attack_dir.name
            attack_cat = ATTACK_CAT_MAP.get(dir_name, dir_name)
            attacker_ips = ATTACKER_IPS.get(dir_name, [])
            for pcap in sorted(attack_dir.glob("*.pcap")):
                specs.append(GothamPcapSpec(
                    pcap_path=pcap,
                    is_attack=True,
                    attack_cat=attack_cat,
                    attacker_ips=attacker_ips,
                ))

    return specs
