# Model Card — ADNS NFStream Detector

ADNS ships one trained detector: `api/model_artifacts/nfstream_model.joblib`. This
card documents what it is, how it was trained, how it performs, and where it should
not be trusted.

> **Intended use:** an educational/demonstration anomaly-detection platform for
> coursework, workshops, and portfolio review. **Not** intended as a production
> intrusion-detection system or a basis for automated enforcement against real
> traffic.

---

## 1. Model — `nfstream_model.joblib`

| | |
| --- | --- |
| **Type** | Soft-voting ensemble: `XGBClassifier` + `ExtraTreesClassifier` |
| **Trainer** | `ml/train_nfstream.py` |
| **Training data** | E3 pooled corpus: UNSW-NB15 + Gotham Dataset 2025 + CIC-IDS2017 Tuesday (4,698,466 flows total) |
| **Features** | 21 `FEATURE_COLUMNS` from `ml/adns_flows/schema.py` |
| **Target** | Binary: 0 = normal, 1 = attack |
| **Artifact size** | ~102 MB; tracked via Git LFS |
| **Schema gate** | `validate_matrix()` raises `SchemaError` on any column name/order mismatch before every `predict()` call |

### Feature contract

All 21 features are observable from live network flows — no synthesis, hashing, or
imputation:

```
proto, duration, src_bytes, dst_bytes, total_bytes,
src_pkts, dst_pkts, total_pkts,
bytes_ratio, pkts_ratio,
src_mean_pkt_size, dst_mean_pkt_size,
bytes_per_sec, pkts_per_sec,
dst_port_bucket,
syn_count, ack_count, rst_count, fin_count, psh_count, urg_count
```

`src`/`dst` follow the canonical orientation rule: `src = endpoint with the lower
(ip, port) tuple`. Both the corpus builder and the live scoring path call
`canonicalize_orientation()` once per flow — the feature distribution is identical
at training and serve time.

### Training corpus

| Dataset | Flows | Attack% | Source |
|---------|-------|---------|--------|
| UNSW-NB15 | 2,058,890 | 3.22% | Jan 22 + Feb 17 2015 PCAPs |
| Gotham Dataset 2025 | 2,331,227 | 97.05% | 5 attack categories |
| CIC-IDS2017 Tuesday | 308,349 | 2.26% | FTP-Patator, SSH-Patator, Web attacks |
| **Total** | **4,698,466** | **49.6%** | 3 distinct network environments |

### Performance (E3 pooled, 5-fold cross-validation hold-out)

| Estimator | PR-AUC | Recall | Benign FPR |
|-----------|--------|--------|-----------|
| XGBoost | 1.0000 | 99.66% | 0.05% |
| ExtraTrees | 1.0000 | 99.66% | 0.03% |

Cross-environment evaluation (models trained on subset, tested on held-out domain):

| Config | PR-AUC | Recall | Benign FPR | Notes |
|--------|--------|--------|-----------|-------|
| A in-domain Gotham | 1.000 | 99.58% | 0.01% | near-perfect |
| A in-domain UNSW | 0.9998 | 99.94% | 0.03% | near-perfect |
| B UNSW→Gotham | 0.9764 | 33.9% | 47.1% | cross-domain collapse |
| B Gotham→UNSW | 0.3276 | 72.9% | 43.6% | cross-domain collapse |
| E1 UNSW+Gotham→CIC | 0.1307 | 100% | 36.3% | unacceptable; calibration failure |
| E2 CIC in-domain | 1.000 | 100% | 0.00% | near-perfect |
| **E3 pooled all 3** | **1.000** | **99.66%** | **0.05%** | deployed config |

---

## 2. Known limitations

**Cross-domain generalization fails without pooling.** Single-environment or
two-environment models collapse to 43–50% benign FPR on held-out network environments.
Root cause: benign flow distributions shift dramatically across environments (bytes
8–30×, packets 9× between UNSW and Gotham). Score ordering is good; calibration
misfires at deployment prevalence. The three-way pool is the current mitigation —
a fresh network environment would still require re-pooling with local benign data.

**Payload-defined attacks have lower recall.** Categories where the attack signal
lives in application-layer payload (Shellcode, Backdoor, CoAP-amplification) show
lower recall than flow-stat attacks. The 21-feature contract is restricted to what
a passive flow observer can measure — no deep-packet inspection.

**`/simulate` scores 0.0.** The `/simulate` endpoint generates synthetic flows
with a simplified feature format. `extra_to_feature_vector()` returns `None` for
flows without the `_extractor: nfstream` marker, so simulated flows score 0.0.
Real live-captured flows score correctly. Fix requires updating
`generate_attack_flows()` in `api/app.py` to emit the 21-column NFStream contract.

**No automated enforcement on real users.** Active-response endpoints are disabled
by default and token-gated (ADR-0007). Do not wire model output to automatic
blocking against real traffic.

---

## 3. Absent-model behavior

If `nfstream_model.joblib` is missing:
- `NfstreamDetectionEngine` logs at `ERROR` level and sets `is_model_loaded = False`
- `/capture/autostart` returns HTTP 503 — no silent no-scoring
- `/model_status` reports `meta_model_status: "absent"`

Run `git lfs pull` after clone (fetches the 102 MB artifact) or retrain with
`python ml/train_nfstream.py`.

---

## Reproducing

See `CLAUDE.md` §5 for the exact corpus rebuild and training commands. Raw PCAPs
(~15 GB) are required and are not redistributable. Corpus parquets (~120 MB) are
not in git but are reproducible from the PCAPs.
