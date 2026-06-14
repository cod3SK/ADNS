# Model Card — ADNS Detectors

ADNS ships two trained detectors plus a rule-based fallback, selected at runtime
by the cascade described in
[ADR-0003](../design-decisions/0003-three-tier-detection-cascade.md). This card
documents what they are, how they were trained, how they perform, and — just as
importantly — where they should *not* be trusted.

> **Intended use:** an educational/demonstration anomaly-detection platform for
> coursework, workshops, and portfolio review. **Not** intended as a production
> intrusion-detection system or a basis for automated enforcement against real
> traffic.

---

## 1. Meta ensemble — `meta_model_combined.joblib` (primary)

| | |
| --- | --- |
| **Type** | Soft-voting ensemble: `ExtraTreesClassifier` + `XGBClassifier` |
| **Trainer** | `ml/meta/meta_train.py` |
| **Training data** | Merged Zeek/TON_IoT-style flow dataset (`merged_train.csv` / `merged_test.csv`), cleaned by `ml/preprocess/merge_and_clean.py` |
| **Features** | ~46 columns: directional bytes/packets, connection state, DNS/HTTP/SSL/`weird` fields, plus reverse-DNS signals |
| **Selected at runtime** | First — used whenever the artifact and xgboost are present |

**Components**
- **ExtraTrees** — 120 trees, `max_depth=30`, trained chunk-wise on a reduced
  3-class target (`{0→0, 2→1, 3→2}`).
- **XGBoost** — binary `normal` vs `attack` (`max_depth=10`, 500 estimators,
  `lr=0.10`), trained incrementally across chunks.

At inference, `MetaEnsembleModel` averages per-class probabilities across both
estimators and maps the top class to a label.

**Known issues (be honest about these)**
- **Label-space mismatch.** The runtime `CLASS_LABELS` map defines six classes
  (`normal/attack/scanning/dos/injection/ddos`), but the checked-in ExtraTrees was
  trained on a 3-class reduction and XGBoost is binary. The richer labels only
  materialize if a correspondingly multi-class artifact is supplied. Treat
  non-binary labels from the shipped artifact with caution.
- **No checked-in held-out metrics.** `meta_train.py` does not emit a metrics file,
  so this card cannot quote validated ensemble scores. Regenerate and record them
  before relying on this model.
- **Train/serve skew.** Live features are synthesized/hashed from sparse `tshark`
  data (see [ADR-0005](../design-decisions/0005-feature-synthesis-for-sparse-telemetry.md)),
  so the live input distribution differs from training.
- **Portability bug in preprocessing.** `merge_and_clean.py` hard-codes an absolute
  `DATA_DIR`; point it at your data before rerunning. *(Tracked cleanup item.)*

---

## 2. Lightweight flow detector — `flow_detector.joblib` (fallback)

| | |
| --- | --- |
| **Type** | `CalibratedClassifierCV` (isotonic) wrapping balanced `LogisticRegression` |
| **Trainer** | `ml/train_flow_detector.py` |
| **Training data** | UNSW-NB15 training/testing sets |
| **Features** | Only what the live pipeline reliably collects: `total_bytes` (= `sbytes`+`dbytes`), `log_total_bytes`, and one-hot `proto` |
| **Target** | Binary `label` (0 = normal, 1 = attack) |
| **Selected at runtime** | Second — used when the meta artifact is absent |

This model deliberately mirrors the *minimal* live feature set, so it suffers far
less train/serve skew than the meta ensemble — at the cost of using only three
features. Probabilities are calibrated, and decision thresholds are learned
(`threshold_anomaly` ≈ 0.40 from best-F1 on validation; `threshold_watch` ≈ 0.26).

### Performance (`api/model_artifacts/flow_detector_metrics.json`)

| Metric | Validation | Test |
| --- | --- | --- |
| F1 | 0.838 | 0.730 |
| Precision | 0.778 | 0.612 |
| Recall | 0.908 | 0.904 |
| Accuracy | 0.761 | 0.631 |
| ROC-AUC | 0.849 | 0.792 |
| PR-AUC | 0.916 | 0.823 |

**Reading these numbers:** the model is tuned for **high recall** (≈0.90 on test)
at the expense of precision (≈0.61) — appropriate for a detector that prefers to
flag-and-review rather than miss, but it will produce false positives. The
validation→test drop reflects honest generalization on held-out data, not tuning
to the test set.

---

## 3. Heuristic scorer — `FlowScorer` (last resort)

Rule-based score from byte volume, per-source burst rate, traffic direction
(private↔public), protocol, and a small stable jitter (`api/scoring.py`). Requires
no ML dependencies and runs whenever no artifacts are available. It is fully
explainable and serves as both a baseline and the mode used by the test suite
([ADR-0009](../design-decisions/0009-test-strategy-and-ci.md)).

---

## Ethical and operational notes

- **No automated enforcement on real users.** The active-response endpoints are
  disabled by default and token-gated
  ([ADR-0007](../design-decisions/0007-admin-token-gate-for-response-actions.md));
  do not wire model output to automatic blocking against real traffic.
- **Reverse-DNS** lookups send queries about observed peer IPs to a resolver; it is
  configurable and stores only a presence flag/hash, not hostnames.
- **Bias.** Detection quality reflects the public datasets used; performance on a
  given network may differ substantially. Validate on representative data before
  drawing conclusions.

## Reproducing

See the **Training & Data Pipelines** section of the [README](../README.md) for
exact commands to preprocess data and retrain both artifacts.
