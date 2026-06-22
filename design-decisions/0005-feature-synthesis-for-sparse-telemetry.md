# 0005 — Feature synthesis and hashing for sparse live telemetry

- **Status:** Superseded — see amendment below
- **Phase:** 0 — Foundational architecture

## Context

The meta ensemble was trained on a rich, Zeek/TON_IoT-style schema with ~46
columns (directional byte/packet counts, connection state, DNS/HTTP/SSL/`weird`
fields, etc.). Live `tshark` telemetry is far sparser — often little more than a
timestamp, IPs, protocol, total byte count, and whatever ports/service we can
infer. The model expects a fixed-width numeric feature vector, but most of those
columns simply are not observable in real time.

## Decision

`model_runner.MetaFeatureBuilder` constructs the full feature vector from whatever
the flow provides, synthesizing or defaulting the rest:

- **Estimate** unobserved quantities: split total bytes into directional bytes by
  inferred direction (private↔public), estimate packet counts from a mean packet
  size, and estimate duration from byte volume.
- **Hash** categorical/text fields (IPs, service, HTTP URI/agent, DNS query, SSL
  cipher, `weird_*`) into stable numeric buckets with per-field moduli.
- **Default** anything still missing to zero.
- An optional reverse-DNS enrichment (`rdns_exists`/`rdns_hash`) adds a cheap
  signal before scoring.

A `_match_shape` helper pads or truncates the vector to each estimator's expected
width so version skew between artifact and code does not crash inference.

## Consequences

- The trained ensemble can score live traffic at all, which is what makes the
  end-to-end demo work.
- The approach is honest about being an approximation: synthesized and hashed
  features mean the live input distribution differs from the training distribution
  (train/serve skew), so live predictions should be read as indicative, not
  authoritative. This is the project's most important ML limitation and is called
  out in [`../ml/model_card.md`](../ml/model_card.md).
- `_match_shape` trades a hard failure for a silent one: a genuine feature mismatch
  is masked by padding/truncation rather than surfaced. Acceptable for a resilient
  demo; it would need stricter feature contracts before production use.

## Amendment — replaced by NFStream contract (NFStream migration)

`MetaFeatureBuilder` and `_match_shape` have been deleted from `api/model_runner.py`.
The problem they were designed to solve — train/serve skew from unobservable features
— is solved by redesigning what the model is trained on instead of patching at serve time.

**Current approach: the feature contract.**

`ml/adns_flows/schema.py` defines 21 `FEATURE_COLUMNS`, each observable by a
passive flow observer at serve time (directional byte/packet counts, TCP flag counts,
rate features, port bucket). No column requires application-layer dissection,
DNS/HTTP/SSL fields, or `conn_state`.

`ml/adns_flows/extract_nfstream.py` extracts these 21 features identically for:
- Corpus building (`flows_to_dataframe_nfstream()`)
- Live scoring (`flow_to_extra()` stores values in `flow.extra`)

`validate_matrix(data, columns)` is called before every `model.predict()`. It raises
`SchemaError` on any column name or order mismatch — an explicit, loud replacement
for `_match_shape`'s silent zero-pad.

**Train/serve skew: eliminated by construction.** The corpus extractor and the live
serving path import from the same `ml/adns_flows/` module. Tests in
`ml/adns_flows/tests/test_live_equals_training_nfstream.py` (8 tests) prove
byte-identical feature matrices from the same pcap via both paths.
