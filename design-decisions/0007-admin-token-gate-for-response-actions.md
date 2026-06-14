# 0007 — Fail-closed admin-token gate for network-response actions

- **Status:** Accepted
- **Phase:** 1 — Hardening

## Context

ADNS includes active-response endpoints that go beyond detection:
`/block_ip`, `/unblock_ip`, and `/killswitch`. These shell out to `iptables`, and
via `nsenter -t 1 -n` they operate in the **host** network namespace — the API
container runs with `NET_ADMIN`/`SYS_ADMIN`. Combined with wide-open CORS and no
authentication, anyone able to reach the API could drop host traffic or block
arbitrary IPs. That is an unacceptable amount of unauthenticated power, even for a
demo, and exactly the kind of thing a reviewer flags.

## Decision

Gate the destructive endpoints behind a shared admin token, **failing closed**:

- The token is read from `ADNS_ADMIN_TOKEN`.
- When the variable is unset/empty, the endpoints are **disabled** entirely
  (HTTP 403) — the safe default for a fresh clone.
- When set, callers must present the token via `Authorization: Bearer <token>` or
  `X-Admin-Token: <token>`; comparison uses `hmac.compare_digest` to avoid timing
  leaks.
- The read-only `GET /killswitch` (status only) stays open; only the mutating
  `POST` is gated.

Implemented as a `require_admin_token` decorator plus a shared
`_require_admin_token_now()` helper so the per-method gating in `/killswitch` and
the whole-endpoint gating elsewhere share one implementation.

## Consequences

- A default deployment cannot be coerced into firewall changes by an anonymous
  caller; the dangerous capability is opt-in.
- The behavior is fully covered by tests (disabled-without-token, wrong-token,
  valid-token via both headers) in `api/tests/test_admin_endpoints.py`.
- This is a pragmatic shared-secret scheme, not full authn/authz. It is sufficient
  for the project's threat model (a single operator), and a real deployment facing
  multiple users would layer proper identity on top.
