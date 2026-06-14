# 0005 — Feature synthesis and hashing for sparse live telemetry

- **Status:** Accepted
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
