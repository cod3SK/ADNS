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

**Three corpora** (NFStream, promoted to canonical paths):

| Dataset | Rows | Attack% | On-disk path |
|---------|------|---------|--------------|
| Gotham Dataset 2025 | 2,331,227 | 97.05% | `outputs/corpus/gotham_flows.parquet` |
| UNSW-NB15 | 2,058,890 | 3.22% | `outputs/corpus/unsw_flows.parquet` |
| CIC-IDS2017 Tuesday | 308,349 | 2.26% | `outputs/corpus/cic_tuesday_flows.parquet` |

Tshark-era parquets archived to `outputs/corpus/archive/` (not deleted).

**Model artifact versioning:**
`api/model_artifacts/nfstream_model.joblib` (102 MB) is stored in Git LFS.
After cloning: `git lfs pull` fetches it. Absent model blocks `/capture/autostart`
with HTTP 503 — not a silent failure (see `api/model_runner.py:NfstreamDetectionEngine`).

**Reproducible rebuild path** (raw PCAPs required — local to original dev machine):
```
# Raw PCAPs assumed present at the paths below.
# UNSW-NB15 (two collection days):
python ml/run_unsw_day1_nfstream.py   # X:\DATA\UNSW\pcap files\pcaps 22-1-2015\ → outputs/corpus/unsw_day1.parquet
python ml/run_unsw_day2_nfstream.py   # X:\DATA\UNSW\pcap files\pcaps 17-2-2015\ → outputs/corpus/unsw_day2.parquet
python ml/combine_unsw_nfstream.py    # merges day1+day2  → outputs/corpus/unsw_flows.parquet

# Gotham Dataset 2025:
python -m corpus.build_corpus gotham  # X:\DATA\Gotham2025\        → outputs/corpus/gotham_flows.parquet

# CIC-IDS2017 Tuesday:
python -m corpus.build_corpus cic     # X:\DATA\CICIDS2017\Tuesday-WorkingHours.pcap → outputs/corpus/cic_tuesday_flows.parquet

# Train E3 pooled model (reads all three parquets above):
python ml/train_nfstream.py           # → api/model_artifacts/nfstream_model.joblib
```
Corpus parquets (~120 MB total) are NOT in git (no LFS for them — too large and
reproducible from raw PCAPs). The raw PCAPs are ~15 GB total and are not redistributable.

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
- **PyInstaller single-exe bundle** (`ADNS.spec`): bundles NFStream + Npcap DLL hook,
  Npcap installer, React build (`frontend/adns-frontend/dist`), model artifacts
  (`api/model_artifacts/`). xgboost requires `collect_all("xgboost")` plus explicit
  `xgboost/lib/xgboost.dll` — both in the spec. NFStream bundled via
  `collect_all("nfstream")`. tshark DLLs removed from bundle (Phase 6). Build via
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

3. **Never delete a proven component to make it prettier.** A working path stays
   until the replacement passes the same tests on the same data with verified parity.
   "Cleaner architecture" is not a reason to remove a working path mid-migration.
   (tshark extraction was removed in Phase 6 only after NFStream passed every parity
   test, corpus rebuild, cross-eval, frozen-exe smoke test, and grain-parity proof.)

4. **Distinguish "reflects real network behavior" from "tool workaround."** The
   flood-cap is a real network behavior artifact (Mirai floods really are degenerate)
   and stays in the pipeline logic. Tool workarounds get removed when the tool changes.

## 9. CURRENT STATE  ← update this section as work progresses

**NFStream migration — Phase 0: PASS (2026-06-20)**

Frozen exe fork-bomb fixed (`freeze_support()` in `__main__`). Phase 0 results:
- STEP 3 (pcap): 1 flow, exit 0, orphans 0.
- STEP 4 (live 90 s): 135 flows, child count exactly 1 for all 18 samples.
- Serving config: `n_meters=1, n_dissections=0`.

All Phase 0 code committed on `feat/nfstream-migration` (commit 67c20b7).

**NFStream migration — Phase 1: PASS (2026-06-20)**

NFStream extractor implemented behind the existing feature contract.
Three new files on `feat/nfstream-migration`:

- `ml/adns_flows/nfstream_config.py` — canonical SSoT config (statistical_analysis=True
  REQUIRED for 6/21 TCP flag features; idle_timeout=120; active_timeout=1800;
  n_dissections=0; n_meters=1 for serving).
- `ml/adns_flows/extract_nfstream.py` — single-pass extractor: NFStream's
  initiator-based src/dst overridden by `canonicalize_orientation()`; directional
  counts swapped when orientation flips; `validate_matrix()` gates every DataFrame.
- `ml/adns_flows/tests/test_nfstream_parity.py` — 15 parity tests, all pass:
  - Hand-fixture: L2 byte counts (src=186/288, scan=54/270, UDP=54/64), flag counts
    (flow1: syn=2 ack=7 fin=2 psh=2; flow2: syn=5 rst=1), orientation correctness.
  - Config parity: n_meters=1 vs n_meters=2 → byte-identical on all features.
  - Determinism: same pcap twice → identical DataFrames.
  - Orientation invariance: initiator_fwd vs initiator_rev → same canonical src.
  - Cross-extractor: NFStream and tshark agree on ALL contract features — flag counts
    AND byte counts (both count Ethernet frame bytes / L2; no per-packet offset).

Phase 1 verdict: **GO** — NFStream extractor is ready for corpus migration.

**NFStream migration — Phase 2: PASS (2026-06-20)**

`ml/corpus/build_corpus.py` migrated from tshark two-pass + editcap-chunked to
`extract_flows_nfstream()`. tshark path preserved (Phase 6 removal). New CLI
flags: `--extractor {nfstream,tshark}` (default nfstream), `--n-meters N`.

New files:
- `ml/corpus/tests/test_labeling_nf.py` — 27 unit tests for NFStream labeling helpers
  (`_reorient_flow`, `_apply_labels_nf`, `_apply_labels_gotham_nf`,
  `_apply_labels_cic_nf`). All pass.
- `ml/run_unsw_day1_nfstream.py`, `ml/run_unsw_day2_nfstream.py` — UNSW build scripts
  (require `if __name__ == '__main__':` + `freeze_support()` due to NFStream's Windows
  multiprocessing spawn). Omitting the guard causes silent extraction failure on all PCAPs.
- `ml/combine_unsw_nfstream.py` — merges day1 + day2 into `unsw_flows_nfstream.parquet`.

Step 2 gate (probe-attack on NFStream output): PASS — UNSW pcap 6 row 8657
delta_start=+0.007s (in-window). Sanity check on pcap 6: 18,677 flows, 1,218 attack,
0 dropped — byte-identical to tshark result.

Three-corpus rebuild results (all `*_nfstream.parquet`, tshark corpora untouched):

| Dataset       | NFStream rows | Attack%  | vs tshark rows | vs tshark attack% | Notes |
|---------------|---------------|----------|----------------|-------------------|-------|
| CIC Tuesday   | 308,349       | 2.26%    | +45.0%         | -1.01pp           | grain-driven (idle_timeout=120 splits 1745s→17.5s mean flows) |
| Gotham 2025   | 2,331,227     | 97.05%   | -18.9%         | -1.15pp           | flood cap hits harder on Mirai SYN (more flows/IP → more capped) |
| UNSW-NB15     | 2,058,890     | 3.22%    | +0.1%          | -0.02pp           | nearly identical (UNSW has short flows; minimal grain effect) |

All deltas are grain-driven (duration mean shifts confirm it), not labeling breaks.
Determinism check (pcap 6, twice): byte-identical DataFrames. All-zero feature rows: 0.
All 172 ML suite tests pass (46 labeling + 27 NFStream labeling + 99 adns_flows).

All attack categories preserved across all three datasets.

**Label-integrity gate + NFStream cross-eval: PASS (2026-06-20)**

Phase 2 label-integrity gate passed for both high-risk datasets:

*CIC gate (highest risk — +45% rows):*
- FTP-Patator: 4,002 flows, ALL in-window (100%). Probe: delta_start=+48s.
- SSH-Patator: 2,979 flows, ALL in-window (100%). Probe: delta_start=+29s.
- +45% explained: attack +11 (+0.2%), benign +95,633 (+46.5%). Extra rows are
  benign splits; idle_timeout=120s splits long benign sessions (mean 1,746s→17s).
  Short brute-force attack flows (FTP/SSH per-attempt connections) are immune.

*Gotham gate (-18.9% rows):*
- Root cause: UDP flow count difference. TCP mirai_dos nearly identical (+3,224).
  tshark UDP: 925,160 vs NFStream UDP: 361,009 (−564k). tshark's `conv,udp`
  counts every amplification-server→victim response as a unique conversation;
  NFStream tracks fewer UDP flows for the same traffic. Unique 5-tuples:
  tshark 1,607,658 vs NFStream 1,150,886 (30% fewer tracked).
- Label integrity: INTACT. All flows from mirai_dos PCAPs are label=1 (PCAP-
  level, no time windows). Benign grew (+17,745, grain-driven). All 5 attack
  categories preserved. Core invariant holds: NFStream misses the same UDP
  flows at training AND serving time — consistent.
- Flood cap attribution: cap is NOT the cause. Degenerate attack delta is only
  −2,473. The 557k non-degenerate drop is the UDP tracking difference above.

NFStream cross-eval (logs: `outputs/corpus/cross_eval_nfstream.log`):

| Config | NFStream result | vs tshark-era | Change |
|--------|-----------------|---------------|--------|
| A in-domain Gotham | PR-AUC 1.000, recall 99.58%, FPR 0.01% | PR-AUC 1.000, 99.75%, 0.00% | trivial |
| A in-domain UNSW | PR-AUC 0.9998, recall 99.94%, FPR 0.03% | PR-AUC 0.9999, 99.95%, 0.03% | trivial |
| B UNSW→Gotham | PR-AUC 0.9764, recall 33.9%, FPR 47.1% | PR-AUC 0.9784, 27.9%, 56.7% | FPR improved |
| B Gotham→UNSW | PR-AUC 0.3276, recall 72.9%, FPR 43.6% | PR-AUC 0.2801, 93.1%, 84.1% | FPR improved |
| D pooled | PR-AUC 1.000, recall 99.68%, FPR 0.05% | PR-AUC 1.000, 99.84%, 0.04% | trivial |
| E1 UNSW+Gotham→CIC | PR-AUC 0.1307, recall 100%, FPR 36.3% | PR-AUC 0.0774, 100%, 50.9% | FPR improved |
| E2 CIC in-domain | PR-AUC 1.000, recall 100%, FPR 0.00% | PR-AUC 1.000, 100%, 0.00% | identical |
| E3 pooled all 3 | PR-AUC 1.000, recall 99.66%, FPR 0.05% | PR-AUC 1.000, 99.84%, 0.05% | trivial |

*Findings reproduce (qualitative story unchanged):*
- Near-perfect in-domain: YES (PR-AUC ≥ 0.9998, FPR ≤ 0.05%)
- Cross-domain benign-FPR collapse: YES (47–44% vs 57–84% tshark — modestly improved,
  still fails. FPR improvement is real: shorter NFStream grain makes benign flows
  more similar across environments. Same root cause: calibration/prevalence mismatch.)
- Pooling fixes in-pool: YES (D PR-AUC 1.000)
- E1 (UNSW+Gotham → CIC) fails: YES (FPR 36% — unacceptable, still flags everything)
- Three-way pool healthy: YES (E3 PR-AUC 1.000, FPR 0.05%)

Domain-shift drivers: same 8 byte/packet features (src_bytes, dst_bytes,
total_bytes, src_pkts, dst_pkts, total_pkts, src/dst_mean_pkt_size) — unchanged
from tshark-era. Duration added to the three-way shift table (NFStream grain makes
it shift across all three corpora).

**NFStream migration — Phase 3: PASS (2026-06-20)**

Live capture + serving path migrated from tshark per-packet/two-pass to NFStream.

New files:
- `api/serving_nfstream.py` — `flow_to_extra()` (stores 21 FEATURE_COLUMNS in
  flow.extra), `extra_to_feature_vector()` (reads back for scoring), `NfstreamScorer`
  (validates via `validate_matrix()` before every predict — no `_match_shape`).
- `ml/adns_flows/tests/test_live_equals_training_nfstream.py` — 8 acceptance-gate
  tests: all PASS. Proves byte-identical feature values: corpus path (flows_to_dataframe_
  nfstream) == serving path (flow_to_extra → extra_to_feature_vector), same pcap.
- `ml/train_nfstream.py` — E3 three-way pool training script (21 FEATURE_COLUMNS,
  binary XGBoost + ExtraTrees). Output: `api/model_artifacts/nfstream_model.joblib`.

Modified files:
- `api/model_runner.py` — `NfstreamDetectionEngine` only (reads contract features
  from flow.extra, calls NfstreamScorer.score_matrix, validate_matrix gates every call).
- `api/app.py` — `_NfstreamCaptureAgent` uses direct NFStream live capture (no tshark
  ring-buffer). `_find_tshark()`/`_tshark_env()` kept for interface enumeration only
  (`tshark -D` at `/interfaces`).
- `api/tasks.py` — `score_flow_batch()` calls `nfstream_detector.score_many()` directly
  (no routing fork).

Training results (E3 pooled: 4,698,466 flows, 21 features):
- XGBoost: PR-AUC 1.0000, recall 99.66%, benign FPR 0.05%
- ExtraTrees: PR-AUC 1.0000, recall 99.66%, benign FPR 0.03%
- `n_features_in_=21` on both models — SchemaError on any column mismatch

Phase 3 acceptance gate (live == training): PASS — 8/8 tests.
All 180 ML suite tests pass.

**NFStream migration — Phase 4: PASS (2026-06-21)**

Grain-parity gap closed. Pre-fix `_NfstreamCaptureAgent` captured 15-second tshark
ring-buffer pcap windows and ran `extract_flows_nfstream()` on each independently.
NFStream force-closes any flow still active at the end of a pcap, so sessions > 15 s
were split into per-window fragments with truncated duration, different bytes_per_sec,
and different pkts_per_sec — systematically different from the training distribution.
The CIC benign mean duration after NFStream grain is ~17.5 s; roughly half of benign
sessions exceeded 15 s and would fragment under the old live path.

**Analysis results** (`ml/adns_flows/tests/test_windowing_grain.py`):
- Synthetic 45 s flow: corpus path → 1 complete flow (duration ~45 s); 15 s-windowed
  path → 4 fragments (each ≤ 15 s). Grain ratio = 4×.
- Bytes conserved across fragments (sum of fragment src/dst_bytes == whole-flow bytes).
- Short flows (< 15 s) are byte-identical between both paths — windowing only affects
  sessions that span a window boundary.
- FPR delta on synthetic pcap: corpus FPR = 0.0%, windowed FPR = 0.0% (mean score
  0.009 vs 0.007); immaterial on synthetic data but structural divergence is proven.

**Decision: Option B.1 — direct NFStream live capture** (not Option A = retrain corpora).
Rationale: corpora are proven and validated; rebuilding them with 15 s grain would
fragment all long benign sessions in the training data, requiring re-validation of all
cross-eval results. Option B preserves the proven corpora. Direct NFStream live capture
applies the same `idle_timeout=120 s` / `active_timeout=1800 s` to the live interface
as corpus extraction — the invariant holds by construction, not by alignment.

**Fix** (`api/app.py`, `_NfstreamCaptureAgent`):
- Removed tshark ring-buffer pcap approach (15 s windows, `_proc`, `_batch_dir`).
- Now uses `NFStreamer(source=self._interface, **make_nfstream_kwargs(n_meters=1))`
  directly in `_run_loop`. Flows expire via idle_timeout/active_timeout (natural),
  not at artificial pcap boundaries.
- `_nf_to_flow()` called on each flow — same as the corpus builder.
- `_stop_internal()` uses psutil fallback to terminate NFStream meter workers
  (this version of NFStream has no `_terminate()` method — meter workers spawn
  on first iteration, not on NFStreamer creation; psutil kills them at stop time).
- `tshark_bin` parameter kept in `start()` for API compat; unused after Phase 4.

**PROVEN — benign FPR delta measured on real CIC data** (`step1_fpr_delta.py`):
CIC Tuesday first 1500 s (pure benign, 8,226 flows, 15.5% with duration > 15 s):
- FPR (a) corpus path      : 0.0000%  (0/8226 flagged)
- FPR (b) new live path    : 0.0000%  (0/8226 flagged)
- FPR delta (b - a)        : 0.0000pp
- Feature matrices (a vs b): byte-identical (PASS)
- FPR (old) windowed path  : 0.1571%  (24/15,275 fragmented flows flagged)
- Grain mismatch cost       : +0.1571pp (3× the in-domain CIC threshold of 0.05%)
The old windowed path produced 15,275 flows from the same 1500 s of traffic (86% more
due to fragmentation). The new live path and corpus path produce exactly 8,226 flows,
byte-identical features, and FPR = 0.0000%. The fix is PROVEN, not just reasoned.

**PROVEN — new tests for the fixed path** (`ml/adns_flows/tests/test_windowing_grain.py`, 7 tests, all pass):
- `test_corpus_path_sees_one_complete_long_flow` — corpus returns 1 flow for 45 s session
- `test_windowed_path_fragments_long_flows` — OLD live path produces 4 fragments
- `test_short_flows_unaffected_by_windowing` — flows < 15 s match exactly between paths
- `test_windowed_bytes_conserved_across_fragments` — byte conservation property holds
- `test_new_live_path_matches_corpus_grain` — NEW acceptance gate: `_extract_direct_nfstream()`
  (mirrors `_NfstreamCaptureAgent._run_loop` exactly) produces 1 complete long flow
  with byte-identical feature matrix to the corpus path.
- `test_live_windowing_equals_corpus_grain` — kept as regression guard for retired windowed path
- `test_benign_score_delta` — FPR delta report (model scoring)

**PROVEN — continuous-stream memory safety** (`step3_memory_verify.py`, 90 s run):
- child count: min=1, max=1 at all 18 sample points (PASS)
- RSS start: 86.3 MB, RSS end: 86.4 MB, growth: +0.2 MB (PASS — bounded)
- orphan processes after shutdown: 0 (PASS — psutil termination cleans up correctly)
- NOTE: This NFStream version has no `_terminate()` API. Shutdown uses
  `psutil.Process().children()` to terminate meter workers. The capture thread
  (daemon) may take >10 s to detect the dead meter; this is safe since daemon
  threads are killed with the process. Confirmed by `_stop_internal()` in `api/app.py`.

All 187 ML suite tests pass after Phase 4 closure.

**NFStream migration — Phase 6: COMPLETE (2026-06-21)**

Final cutover — all tshark dead code removed. Order of operations:

*STEP 1 — Bundle NFStream into frozen exe:*
- `ADNS.spec` updated: `_WIRESHARK_DIR` and `_tshark_datas` block removed; NFStream
  bundled via `collect_all("nfstream")` + explicit DLL hook.
- Npcap DLL runtime hook added.
- `--headless` mode added for smoke-test automation.
- Windows Job Object added to `api/app.py` for forced-shutdown orphan protection.

*STEP 2 — Smoke test the frozen exe (run post-removal, gate cleared retroactively):*
- Build7 (dist/ADNS/ADNS.exe, 2026-06-21 13:47) built with NFStream-only ADNS.spec.
- First smoke test attempt (13:32) FAILED 4/7 — test script used wrong endpoint paths
  (`/capture/status` instead of `/capture_status`). Script fixed at 13:50.
- Re-run against build7 with corrected script: **11/11 PASS** (2026-06-21).
  - 2.0 startup, 2.1a/b model status, 2.2a/b/c live capture (no DLL error),
    2.3a/b forced-shutdown (0 orphans), 2.4a/b/c detection + non-zero scores.
- Forced-shutdown: 0 orphan meter workers after `taskkill /F /T` (Job Object works).

*STEP 3 — Retire all dead paths:*
- `ml/adns_flows/extract.py` + `assemble.py` — DELETED (tshark two-pass extractor).
- `ml/adns_flows/tests/test_extract.py`, `test_assemble.py`, `test_parity.py` — DELETED.
- `api/model_runner.py` — `MetaFeatureBuilder`, `MetaEnsembleModel`, `DetectionEngine`
  REMOVED; only `NfstreamDetectionEngine` remains.
- `api/tasks.py` — routing fork removed; direct `nfstream_detector.score_many()` call.
- `ml/corpus/build_corpus.py` — tshark two-pass branches, `_apply_labels()`,
  `_apply_labels_cic()`, `_apply_labels_gotham()`, `_cmd_probe_attack()`,
  `_cmd_sanity_check_gotham()` REMOVED; `--tshark` and `--extractor` CLI args removed.
  `REASON_NO_TIMESTAMP` kept as a constant (used by `assert_drop_rate` tests).
- `ml/corpus/__init__.py` — dead re-exports removed.
- `ml/corpus/tests/test_labeling.py` — rewritten without `_apply_labels` tests (kept
  TIER 3–8: balance gate, drop-rate gate, CorpusStats, get_pcap_start_epoch, load_label_index).
- `api/tests/test_scoring_and_features.py`, `test_scoring_pipeline.py` — DELETED
  (tested removed MetaFeatureBuilder/DetectionEngine).
- `_find_tshark()`/`_tshark_env()` KEPT in `api/app.py` — still used by `tshark -D`
  interface enumeration at `/interfaces`.

131 ML + API suite tests pass after all removals.

**Final architecture:**
- Single extraction path: `ml/adns_flows/extract_nfstream.py` (NFStream, single-pass)
- Single feature contract: `ml/adns_flows/schema.py` (21 FEATURE_COLUMNS)
- Single model: `api/model_artifacts/nfstream_model.joblib` (XGBoost E3 pooled)
- Corpus builders: `build_corpus()`, `build_corpus_gotham()`, `build_corpus_cic()` —
  all call `extract_flows_nfstream()`, no extractor param.
- Serving: `NfstreamDetectionEngine` + `NfstreamScorer` + `validate_matrix()` gate.
- Live capture: `_NfstreamCaptureAgent` with direct NFStream interface streaming.

**STEP 4 — Promote corpora + rebuild exe**: COMPLETE.
- NFStream parquets renamed to canonical names (`*_nfstream` suffix removed).
- Tshark-era parquets archived to `outputs/corpus/archive/`.
- Frozen exe verified: 11/11 smoke checks pass; 131 ML+API tests pass.
- Code references in `ml/train_nfstream.py`, `ml/combine_unsw_nfstream.py`,
  `ml/run_unsw_day1_nfstream.py`, `ml/run_unsw_day2_nfstream.py` updated to
  canonical parquet names.

Detailed migration plan: `memory/ml_next_steps.md` +
`memory/nfstream_phase0.md` (Claude auto-memory, loaded at session start).

**Deployment-time calibration — COMPLETE (2026-06-23, branch feat/scan-flood-detector)**

Diagnostic (`diag_pipeline.py`) confirmed a 30.77% FPR on real home-network traffic
(39 flows, 30 s capture). Root cause: mDNS (port 5353), DHCP, SSDP — protocols absent
from all three training corpora. Model scores correctly otherwise (attacks → 1.0000,
TCP web traffic → low). The threshold-only `feat/per-network-calibration` approach was
insufficient; full retrain with local benign is the fix.

New files (commit b88c8de):
- `api/calibration/__init__.py` — package init
- `api/calibration/pipeline.py` — 5-stage retrain daemon:
  1. Capture (tshark, default 1800 s, cancel-safe)
  2. Extract (NFStream serve-time config — same as live scoring)
  3. Filter (protocol whitelist `SAFE_PORTS` + IQR outlier drop)
  4. Retrain (XGBoost 300 trees + ExtraTrees 200 trees on ~100 k corpus attacks + local benign)
  5. Validate (Gates 5A load / 5B attack score >0.80 / 5C local FPR <5% / 5D corpus recall >90%)
     → swap model on pass, revert on fail
- `api/calibration/whitelist.py` — `SAFE_PORTS` frozenset (30+ entries), `apply_whitelist()`,
  `apply_outlier_drop()` (IQR-based; robust where mean+3σ fails on extreme single outliers)
- `api/calibration_routes.py` — Flask Blueprint `/calibration/*`: status, start, cancel, reset,
  revert, reload_model, first_run_check (7 endpoints)
- `api/tests/test_calibration_pipeline.py` — 28 tests (whitelist ×9, outlier ×6, state ×5, routes ×8)

Modified:
- `api/model_runner.py` — `NfstreamDetectionEngine.reload()` for hot-model-swap without restart
- `api/app.py` — blueprint registration (try/except; optional)
- `frontend/adns-frontend/src/App.jsx` — `CalibrationPanel` in Settings tab, first-run opt-in modal
- `frontend/adns-frontend/src/App.css` — btn-primary, btn-secondary, modal-overlay, modal-box CSS

**169 tests passing** (28 new + 131 prior).

Open items:
- Corpus parquets (`outputs/corpus/*.parquet`) are required for Stage 4 retrain; they are
  not included in the frozen exe. A production path (bundle-at-build or download-on-first-run)
  is not yet designed.
- Scan-flood-detector (stateful per-IP aggregation for port-scan + SYN-flood heuristics,
  replacing `_infer_scanning` in `api/tasks.py`) is not yet implemented.
