# Architecture Decision Records (ADRs)

This folder documents the significant design decisions behind ADNS using the
[Architecture Decision Record](https://adr.github.io/) format. Each record is
immutable once accepted; when a decision changes we add a new ADR that supersedes
the old one rather than editing history.

Each ADR follows the same shape: **Context** (the forces at play), **Decision**
(what we chose), **Consequences** (the trade-offs we accepted), and where useful
**Alternatives considered**.

## Start here

**[architecture.md](architecture.md)** — full pipeline map: every component,
call direction, batch size, timing bound, and tuning variable in one place.
Read this first; the ADRs below explain the *why* behind individual decisions.

## Index

### Phase 0 — Foundational architecture
The core capture → ingest → score → visualize loop as originally built.

| ADR | Title | Status |
| --- | --- | --- |
| [0001](0001-microservice-architecture.md) | Microservice architecture and data flow | Accepted |
| [0002](0002-async-scoring-redis-rq.md) | Asynchronous scoring with in-process thread pool (amended from Redis/RQ) | Accepted |
| [0003](0003-three-tier-detection-cascade.md) | Three-tier detection cascade with hot reload | Superseded — see amendment |
| [0004](0004-postgres-persistence-and-retention.md) | PostgreSQL persistence, in-code schema management, and retention | Accepted |
| [0005](0005-feature-synthesis-for-sparse-telemetry.md) | Feature synthesis and hashing for sparse live telemetry | Superseded — see amendment |
| [0006](0006-attack-simulation-subsystem.md) | Built-in attack simulation subsystem | Accepted |

### Phase 1 — Hardening
Closing the gaps that separate a demo from something safe to expose.

| ADR | Title | Status |
| --- | --- | --- |
| [0007](0007-admin-token-gate-for-response-actions.md) | Fail-closed admin-token gate for network-response actions | Accepted |
| [0008](0008-externalized-configuration-and-secrets.md) | Externalized configuration and secrets | Accepted |

### Phase 2 — Quality and confidence
Making correctness observable and regressions catchable.

| ADR | Title | Status |
| --- | --- | --- |
| [0009](0009-test-strategy-and-ci.md) | Lightweight test strategy and continuous integration | Accepted |

### Phase 3 — Desktop packaging and distribution
Shipping a self-contained Windows installer and hardening the update story.

| ADR | Title | Status |
| --- | --- | --- |
| [0010](0010-windows-desktop-packaging.md) | Windows desktop packaging with PyInstaller and Inno Setup | Accepted |
| [0011](0011-tabbed-navigation-layout.md) | Tabbed left-nav layout for the dashboard | Accepted |
| [0012](0012-installer-versioning-and-update-safety.md) | Installer versioning and update safety | Accepted |
| [0013](0013-tshark-4x-batch-conv-format.md) | tshark 4.x batch conv output format compatibility | Superseded — tshark removed |

## Related documents
- [`../ml/model_card.md`](../ml/model_card.md) — model card for the NFStream detector.
- [`../README.md`](../README.md) — project overview and quickstart.
- [`../AGENTS.md`](../AGENTS.md) — component reference and operational notes.
