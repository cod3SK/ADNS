# 0007 — Fail-closed admin-token gate for network-response actions

- **Status:** Accepted (amended — see below)
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

Gate `/block_ip` and `/unblock_ip` behind a shared admin token, **failing closed**:

- The token is read from `ADNS_ADMIN_TOKEN`.
- When the variable is unset/empty, those endpoints are **disabled** entirely
  (HTTP 403) — the safe default for a fresh clone.
- When set, callers must present the token via `Authorization: Bearer <token>` or
  `X-Admin-Token: <token>`; comparison uses `hmac.compare_digest` to avoid timing
  leaks.
- The read-only `GET /killswitch` (status only) stays open.

Implemented as a `require_admin_token` decorator plus a shared
`_require_admin_token_now()` helper used by the whole-endpoint gating on
`/block_ip` and `/unblock_ip`.

## Consequences

- A default deployment cannot be coerced into arbitrary IP blocks by an anonymous
  caller; the capability is opt-in.
- The behavior is fully covered by tests (disabled-without-token, wrong-token,
  valid-token via both headers) in `api/tests/test_admin_endpoints.py`.
- This is a pragmatic shared-secret scheme, not full authn/authz. It is sufficient
  for the project's threat model (a single operator), and a real deployment facing
  multiple users would layer proper identity on top.

## Amendment — killswitch ungated

`POST /killswitch` was originally included in the token-gated set.  It has since
been removed from the gate for the following reason: the killswitch is a
**first-responder dashboard action** — the whole point is that an operator can hit
it immediately when an attack is detected, without needing to look up or configure
a token.  Requiring `ADNS_ADMIN_TOKEN` made the button silently fail (HTTP 403) on
a default deployment, which is the opposite of useful for an emergency control.

The risk trade-off is acceptable: anyone who can reach the dashboard on localhost
is already the operator.  The `block_ip`/`unblock_ip` token gate is retained
because those endpoints persistently modify firewall rules for specific IPs, which
is a different threat model.

Additionally, the killswitch scope was expanded from a single configured interface
(`ADNS_KILLSWITCH_INTERFACE`, default `eth0`) to **all non-loopback interfaces**
(`iptables ! -o lo` / `! -i lo` on Linux; `netsh advfirewall` block-all rules on
Windows), so it actually cuts all external traffic rather than one NIC.
