# ADNS — Claude context file

## 1. PROJECT

ADNS is a network anomaly detector that runs as a Windows desktop exe (Flask API +
React dashboard, PyInstaller single-exe bundle). The goal is a deployable detector
trained on features that are actually observable at serve time. The original v0
model (`api/model_runner.py:MetaFeatureBuilder`) was trained on a ~46-column
TON_IoT-derived schema that included `conn_state`, `service`, `dns_*`, `ssl_*`,
`http_*`, and `weird_*` — none of which live tshark output can supply. At serve
time `_build_row` invented these: `service` was hashed via `_stable_hash(proto)`,
`conn_state` was hashed from a direction-tag string, app-layer fields were zeroed.
`_match_shape` then silently padded or truncated columns so the tensor matched the
model's expectation without raising. The model was being scored on fabricated
inputs and reporting non-zero scores purely from the noise. The rebuild replaces
the entire extraction + training path with `ml/adns_flows/` — a shared module used
identically at corpus-build time and at serve time, restricted to what tshark can
actually observe per conversation.

## 2. CORE INVARIANT (never break this)

**Training features and serving features must come from the same extractor at the
same flow grain, proven by tests.**

The invariant is NOT "matches some external schema." It is that `ml/adns_flows/`
is the only code path that produces model inputs — corpus-building and live scoring
both import from it. Any new feature must be added to `schema.py` and observable
by both `run_pass_a` + `run_pass_b` at serve time. Tests in
`ml/adns_flows/tests/` enforce exact column identity and ordering.

This principle explains every other design choice below.

## 3. FEATURE CONTRACT

**Single source of truth: `ml/adns_flows/schema.py`**

```python
IDENTITY_COLUMNS = ("ts", "src_ip", "dst_ip", "src_port", "dst_port")
# Never fed to the model; kept for joins and labeling only.

FEATURE_COLUMNS = (
    "proto",              # numeric: tcp=6, udp=17, icmp=1, other=0
    "duration",
    "src_bytes",          # canonical-src→dst (after orientation)
    "dst_bytes",          # canonical-dst→src (after orientation)
    "total_bytes",
    "src_pkts",
    "dst_pkts",
    "total_pkts",
    "bytes_ratio",        # dst_bytes / (src_bytes + 1)
    "pkts_ratio",         # dst_pkts / (src_pkts + 1)
    "src_mean_pkt_size",
    "dst_mean_pkt_size",
    "bytes_per_sec",
    "pkts_per_sec",
    "dst_port_bucket",    # 0=well-known(<1024) 1=registered(<49152) 2=ephemeral
    "syn_count", "ack_count", "rst_count", "fin_count", "psh_count", "urg_count",
)
# 21 features total. Order is load-bearing — validate_matrix enforces it.
```

**Orientation rule** (`schema.canonicalize_orientation`): `src = endpoint with the
lower (ip, port) tuple under Python lexicographic ordering`. Total, stable,
deterministic. Both the live scoring path and all corpus builders call
`canonicalize_orientation()` once per flow — never around it. Corpus builders may
pass `prefer_src=<attacker_ip>` to pin the attacker as src; live capture always
omits it and uses the default rule. Eight directional features (`src_bytes`,
`dst_bytes`, `src_pkts`, `dst_pkts`, `bytes_ratio`, `pkts_ratio`,
`src_mean_pkt_size`, `dst_mean_pkt_size`) depend on orientation — getting it wrong
silently mirrors these features relative to training.

**`validate_matrix(data, columns)`** raises `SchemaError` on any column
name/order mismatch. It is the explicit replacement for `_match_shape`'s silent
zero-pad. Call it before any model.predict(). See `test_schema.py` for the pass/
fail contract.

## 4. EXTRACTION

**Current extractor: tshark two-pass** (`ml/adns_flows/extract.py` +
`ml/adns_flows/assemble.py`)

- **Pass A** — `tshark -q -z conv,tcp -z conv,udp` → bidirectional byte/packet
  counts per conversation, returned with neutral names (`ep_a`, `ep_b`, `bytes_ab`,
  `bytes_ba`, `pkts_ab`, `pkts_ba`). No src/dst assignment here.
- **Pass B** — `tshark -T fields -Y "ip && tcp" -e ip.src … -e tcp.flags` →
  TCP flag counts aggregated per `orientation_key` (unordered pair), direction-
  agnostic. Falls back to chunked-pass-B via `editcap` for PCAPs > ~500 K packets
  (timeout guard in `build_corpus.py:run_pass_b_chunked`). Timeout → quarantine
  (`REASON_FLAGS_UNEXTRACTABLE`) — **never zero-fill flags**.
- **Assembly** — `assemble._make_flow` calls `canonicalize_orientation()` once,
  then assigns directional counts, then joins flag counts on `orientation_key`.

**Flow unit**: one bidirectional TCP/UDP conversation = one row. The original bug
was scoring per-packet live telemetry (one DB row per packet, not per conversation)
which produced degenerate feature values that the model had never seen in training.

**Migration in progress**: tshark is being replaced by NFStream (Python library
with C extensions, single-pass, same API for corpus building and live capture).
See §9 CURRENT STATE.

tshark binary probe order (same in `extract.py` and `api/app.py`): PyInstaller
bundle (`_MEIPASS/tshark/tshark.exe`) → `TSHARK_BIN` env → `C:\Program
Files\Wireshark\tshark.exe` → `PATH`.

## 5. CORPUS PIPELINE

All code in `ml/corpus/`. Run via `python -m corpus.build_corpus --dataset
unsw|gotham|cic`.

**Three-way labeling** — every extracted flow gets exactly one outcome:
- `ATTACK (label=1)` — matched ground-truth or in a malicious PCAP directory
- `BENIGN (label=0)` — processed cleanly but no attack label matched; no-match
  is NOT a reason to drop (for UNSW, expected ≥60% of GT rows are from other
  collection days not in our pcap set)
- `DROPPED (n_dropped_unprocessable)` — genuine processing failure: tshark crash
  (`extraction_fail`), unreadable pcap timestamp (`no_timestamp`), pass-B timeout
  (`flags_unextractable`), unexpected exception (`other`). Reason breakdown is
  logged; "no attack label match" is **never** in this bucket.

**Guardrails** (all called before writing parquet):
- `assert_drop_rate()` — HALT if `n_dropped / total_seen > 0.10`. A high rate
  signals an extraction bug; do NOT raise the threshold.
- `assert_sane_balance()` — sanity-check class balance (configurable via
  `allow_skewed=True`).
- **Flood cap** — per-source-IP cap of `DEFAULT_FLOOD_CAP = 3000` on degenerate
  flows (`src_pkts≤1, dst_pkts==0, label==1`). Random sample with `seed=42`.
  Prevents Mirai SYN floods from creating one tshark conversation per SYN and
  dominating the corpus 99:1 in attack rows from degenerate one-sided flows.
- **`--probe-attack` timezone gate** — extract a known attack flow to confirm
  epoch alignment before running the full build.

**Three corpora** (all built through the identical v2 pipeline):

| Dataset | Rows | Attack% | On-disk path |
|---------|------|---------|--------------|
| Gotham Dataset 2025 | 2,873,519 | 98.2% | `outputs/corpus/gotham_flows_v2.parquet` |
| UNSW-NB15 v2 | 2,056,824 | 3.24% | `outputs/corpus/unsw_flows.parquet` |
| CIC-IDS2017 Tuesday | 212,705 | 3.27% | `outputs/corpus/cic_tuesday_flows.parquet` |

**Dataset-specific label quirks:**

*UNSW-NB15:* Ground-truth CSV at `X:\DATA\UNSW\CSV Files\NUSW-NB15_GT.csv`
(filename is literally "NUSW", not "UNSW" — that is how it shipped). 174,347 rows,
all attacks, covering all 9 collection days; our pcap set covers Jan 22 and Feb 17
only. Labels matched via endpoint pair + proto + ±1s time window. OSPF flows
(`sport=0, dsport=0`) were never captured by tshark conversation tracking — 60%
unmatched rate is expected and not a bug. PCAPs in `pcaps 22-1-2015/` (files
6,7,13-53 are pcapng — required the pcapng epoch reader fix in `build_corpus.py:
_pcapng_start_epoch`; returning None instead of 0 was dropping 80% of UNSW flows
silently). Timezone: both PCAP timestamps and GT CSV are UTC — confirmed by probing
GT row 8657 in pcapng 6.pcap.

*Gotham Dataset 2025:* Directory-structure labeling (no GT CSV). `raw/benign/` →
label=0; `raw/malicious/<type>/` → label=1, `attack_cat=<type>`. Mirai SYN flood
PCAPs are 2–3.3 GB and triggered pass-B timeouts → chunked pass-B with editcap.

*CIC-IDS2017 Tuesday:* No CSV label file ships with the dataset. Labels derived
entirely from PCAP forensics (`ml/corpus/cic_labels.py`). Published attacker IP
`205.174.165.73` has **zero packets** in the 11 GB PCAP — confirmed by exhaustive
tshark scan. Real attacker is `172.16.0.1` (NAT gateway). Victim: `192.168.10.50`.
Timezone: ADT (UTC-3), confirmed via `io,stat` burst analysis; label timestamps
must be converted (`label_epoch = published_ADT_time + 10800`). PCAP at
`X:\DATA\CICIDS2017\Tuesday-WorkingHours.pcap`. FTP display-filter counts are zero
on extracted slices (likely VLAN encapsulation) but `conv,tcp` works fine.

## 6. KEY FINDINGS (do not re-litigate)

Cross-eval results (`outputs/corpus/cross_eval_results.log`,
`outputs/corpus/cross_eval_cic.log`):

- **In-domain**: near-perfect for all three corpora (PR-AUC ≥ 0.9999, FPR ≤ 0.05%).
- **Single/partial-domain models FAIL cross-domain**: benign-FPR collapses to
  50–84% in held-out environments. Root cause: benign flow distributions shift
  dramatically across environments (bytes 8–30×, packets 9× between UNSW and
  Gotham). `scale_pos_weight` calibrated at training prevalence misfires at
  deployment prevalence. Score ordering is actually good; calibration is wrong.
  This is a known field-wide problem (arXiv 2402.10974), not a model architecture
  bug.
- **Pooled UNSW+Gotham does NOT generalize to held-out CIC** (Config E1: PR-AUC
  0.0774, benign FPR 50.9%, recall 100% — model flags everything).
- **Three-way pool works** (Config E3: PR-AUC 1.000, recall 99.84%, benign FPR
  0.05% across 17 attack categories from 3 distinct network environments).
- **Do NOT deploy a single-environment model into a different network.**
- Payload-defined attack categories (Shellcode, Backdoor, CoAP-amplification) have
  lower recall than flow-stat attacks — deferred to a future app-layer model.

## 7. SERVING CONSTRAINTS

- **CPU-only inference** — no CUDA/GPU deps in the exe.
- **`xgboost==1.7.6` pinned** (`requirements-desktop.txt`, `api/requirements.txt`).
  Models trained with 1.7.6 will not load under xgboost 2.x. Do not upgrade
  without retraining and re-testing in the frozen exe.
- **scikit-learn should be pinned** — joblib model artifacts are version-sensitive.
- **PyInstaller single-exe bundle** (`ADNS.spec`): bundles tshark/dumpcap DLLs,
  Npcap installer, React build (`frontend/adns-frontend/dist`), model artifacts
  (`api/model_artifacts/`). xgboost requires `collect_all("xgboost")` plus explicit
  `xgboost/lib/xgboost.dll` — both in the spec. Build via
  `pwsh scripts\build_installer.ps1` (runs npm build → PyInstaller → InnoSetup).
- **Npcap** must be installed on the target machine (driver not bundleable);
  installer is shipped alongside the exe.
- No browser localStorage; no heavy deps (torch, tensorflow) in the exe path.

## 8. WORKING PRINCIPLES

These are the discipline rules the project runs on. Violating them is how silent
failures re-enter.

1. **Hunt silent failures.** `None` and `0.0` and an empty dict masquerade as
   observed values. Every default that isn't an explicit observation is a lie to
   the model. Probe before trusting; gate with assertions before writing output.

2. **Gate before trusting.** `assert_drop_rate`, `assert_sane_balance`, the
   timezone probe, `validate_matrix` — these gates must run before any output is
   written. A corpus or model that skipped a gate is not certified.

3. **Never delete a proven component to make it prettier.** tshark extraction
   works and is tested. It stays until NFStream extraction passes the same tests
   on the same data with verified parity. "Cleaner architecture" is not a reason
   to remove a working path mid-migration.

4. **Distinguish "reflects real network behavior" from "tool workaround."** The
   chunked pass-B with editcap is a tshark workaround. The flood-cap is a real
   network behavior artifact (Mirai floods really are degenerate). Workarounds get
   removed when the tool changes; behavior artifacts stay in the pipeline logic.

## 9. CURRENT STATE  ← update this section as work progresses

**NFStream migration — Phase 0: PASS (2026-06-20)**

The frozen exe fork-bombed on the previous run (WinError 1455, paging file
exhausted) because `multiprocessing.freeze_support()` was missing.  NFStream uses
`get_context('spawn')` on Windows and spawns `n_meters` worker processes; without
`freeze_support()` each worker re-ran `main()` exponentially.

Fix applied (all on `main`, not yet committed):
- `nfstream_pkg_test.py`: `freeze_support()` as first call in `__main__`, plus
  `n_meters=1` on every `NFStreamer(...)` call, plus `--mode live` for sustained
  capture testing.
- `launcher.py`: same `freeze_support()` guard.
- `rthook_nfstream.py`: doc updated; freeze_support must NOT go in runtime hooks.
- `run_frozen_guarded.ps1`: safety guard that kills the process tree if child
  count exceeds 12 or wall time exceeds timeout (use this before any frozen test).

Phase 0 test results (exe at `dist_test/nfstream_pkg_test/nfstream_pkg_test.exe`):
- **STEP 3 (pcap read)**: 1 flow from synthetic SYN/SYN-ACK pcap. Exit 0, orphans 0.
- **STEP 4 (live capture 90 s)**: 135 flows, interface
  `\Device\NPF_{E466F43A-35D6-409B-AC2B-A026C362E238}` (Intel Wi-Fi 6E AX211).
  Child count: **exactly 1 for all 18 samples across 92 s — never climbed.**
  Exit 0, orphans 0, guard not triggered.

**Serving config decided**: `n_meters=1, n_dissections=0` for all frozen NFStream
calls. This caps the process tree at root + 1 worker.

**What's next — Phase 1**: Migrate `ml/corpus/build_corpus.py` from the tshark
two-pass + editcap-chunked architecture to NFStream single-pass extraction. All
46+ corpus tests in `ml/corpus/tests/` must pass with NFStream output.  tshark
remains the working path for live scoring in `api/app.py`; nothing is removed
until Phase 1 extraction parity is proven.

Detailed migration plan and per-phase gates: `memory/ml_next_steps.md` +
`memory/nfstream_phase0.md` (Claude auto-memory, loaded at session start).
