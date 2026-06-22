"""
Orientation canonicalization tests — schema.py contract.

Verifies:
  - orientation_key is symmetric (unordered)
  - canonicalize_orientation is idempotent: (a,b) and (b,a) → same (src,dst)
  - default rule: src = lower (ip, port) — total, stable, deterministic
  - tie-breaking: same IP, lower port wins; identical endpoints → ep_a is src
  - prefer_src override: pins a named IP as src; falls back to default on no match
"""
from __future__ import annotations

from adns_flows.schema import canonicalize_orientation, orientation_key

# ── orientation_key ────────────────────────────────────────────────────────

def test_orientation_key_symmetric():
    assert (
        orientation_key("10.0.0.5", 12345, "10.0.0.9", 80)
        == orientation_key("10.0.0.9", 80, "10.0.0.5", 12345)
    )


def test_orientation_key_self():
    """Degenerate: same endpoint on both sides still returns a valid key."""
    k = orientation_key("1.2.3.4", 80, "1.2.3.4", 80)
    assert k == (("1.2.3.4", 80), ("1.2.3.4", 80))


def test_orientation_key_min_is_first():
    k = orientation_key("10.0.0.9", 80, "10.0.0.5", 12345)
    assert k[0] == ("10.0.0.5", 12345)  # lower IP → first slot
    assert k[1] == ("10.0.0.9", 80)


# ── canonicalize_orientation — default rule ────────────────────────────────

def test_idempotent_ab():
    a = ("10.0.0.5", 12345)
    b = ("10.0.0.9", 80)
    assert canonicalize_orientation(a, b) == canonicalize_orientation(b, a)


def test_default_lower_ip_is_src():
    a = ("10.0.0.5", 12345)
    b = ("10.0.0.9", 80)
    src, dst = canonicalize_orientation(a, b)
    assert src == a     # "10.0.0.5" < "10.0.0.9"
    assert dst == b


def test_default_higher_ip_first_still_gets_same_src():
    a = ("10.0.0.9", 80)
    b = ("10.0.0.5", 12345)
    src, dst = canonicalize_orientation(a, b)
    assert src == b     # "10.0.0.5" < "10.0.0.9" wins regardless of call order
    assert dst == a


def test_tiebreak_same_ip_lower_port_is_src():
    a = ("10.0.0.1", 80)
    b = ("10.0.0.1", 443)
    src, dst = canonicalize_orientation(a, b)
    assert src == a     # port 80 < 443


def test_tiebreak_same_ip_lower_port_still_wins_when_reversed():
    a = ("10.0.0.1", 443)
    b = ("10.0.0.1", 80)
    src, dst = canonicalize_orientation(a, b)
    assert src == b     # port 80 < 443


def test_degenerate_identical_endpoints():
    a = ("1.2.3.4", 80)
    src, dst = canonicalize_orientation(a, a)
    assert src == a     # tie-break: ep_a when equal


# ── prefer_src override ────────────────────────────────────────────────────

def test_prefer_src_pins_matching_endpoint_a():
    a = ("10.0.0.5", 12345)
    b = ("10.0.0.9", 80)
    # a would be src by default (lower IP); prefer_src agrees → no change
    src, dst = canonicalize_orientation(a, b, prefer_src="10.0.0.5")
    assert src == a


def test_prefer_src_overrides_default_to_pin_endpoint_b():
    a = ("10.0.0.5", 12345)
    b = ("10.0.0.9", 80)
    # default: a is src; prefer_src forces b to be src instead
    src, dst = canonicalize_orientation(a, b, prefer_src="10.0.0.9")
    assert src == b
    assert dst == a


def test_prefer_src_no_match_falls_back_to_default():
    a = ("10.0.0.5", 12345)
    b = ("10.0.0.9", 80)
    src, dst = canonicalize_orientation(a, b, prefer_src="192.168.1.1")
    # prefer_src matched neither endpoint → default rule
    assert src == a     # "10.0.0.5" < "10.0.0.9"
