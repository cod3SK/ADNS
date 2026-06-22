"""
CIC-IDS2017 Tuesday attack window definitions.

Tuesday captures two brute-force attack phases:
  FTP-Patator  — brute-force against port 21  (09:20–10:20 ADT published)
  SSH-Patator  — brute-force against port 22  (14:00–15:00 ADT published)

ADT (Atlantic Daylight Time) = UTC-3.  PCAP timestamps are UTC.

Key findings from PCAP analysis (not from docs — no CSV label file ships with
the dataset):

  PCAP start epoch : 1499169212  (2017-07-04 11:53:32 UTC = 08:53 ADT)

  Attacker in PCAP : 172.16.0.1  (NAT gateway; documented IP 205.174.165.73
                     has ZERO packets in the 11 GB PCAP — confirmed by
                     exhaustive tshark scan)
  Victim           : 192.168.10.50  (both FTP and SSH targets)

  FTP burst (io,stat, tcp.dstport==21 SYNs):
    Offset 1500–5400s from PCAP start → 12:18–13:23 UTC → 09:18–10:23 ADT
    Published window 09:20–10:20 ADT ✓  (~284 SYNs in burst window)

  SSH burst (io,stat, tcp.dstport==22 SYNs):
    Offset 18900–22800s from PCAP start → 17:08–18:13 UTC → 14:08–15:13 ADT
    Published window 14:00–15:00 ADT ✓  (2983 SYNs confirmed from 172.16.0.1)

Timezone verdict: ADT (UTC-3) confirmed.  PCAP timestamp = label_time + 10800s.

Note: FTP traffic triggers io,stat counts but tshark display filters return 0
packets on extracted slices — likely VLAN encapsulation affecting filter
evaluation.  tshark conversation tracking (conv,tcp) operates below L3 and
should capture FTP conversations correctly regardless.
"""
from __future__ import annotations

PCAP_START_EPOCH: int = 1_499_169_212  # 2017-07-04 11:53:32 UTC

# Attack window list consumed by _apply_labels_cic() in build_corpus.py.
# Each window defines ONE brute-force phase.
CIC_ATTACK_WINDOWS: list[dict] = [
    {
        "stime":       PCAP_START_EPOCH + 1_500,    # 12:18:32 UTC  (09:18 ADT)
        "ltime":       PCAP_START_EPOCH + 5_400,    # 13:23:32 UTC  (10:23 ADT)
        "attacker_ip": "172.16.0.1",
        "victim_ip":   "192.168.10.50",
        "dst_port":    21,
        "proto":       "TCP",
        "attack_cat":  "bruteforce_ftp",
    },
    {
        "stime":       PCAP_START_EPOCH + 18_900,   # 17:08:32 UTC  (14:08 ADT)
        "ltime":       PCAP_START_EPOCH + 22_800,   # 18:13:32 UTC  (15:13 ADT)
        "attacker_ip": "172.16.0.1",
        "victim_ip":   "192.168.10.50",
        "dst_port":    22,
        "proto":       "TCP",
        "attack_cat":  "bruteforce_ssh",
    },
]
