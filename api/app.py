import hmac
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from model_runner import DetectionEngine
from task_queue import enqueue_flow_scoring

app = Flask(__name__)
CORS(app)

# Default to a local SQLite file so the app works without PostgreSQL.
# The launcher (desktop) overrides this with AppData path; production sets the env var.
_instance_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instance")
os.makedirs(_instance_dir, exist_ok=True)
_default_sqlite = "sqlite:///" + os.path.join(_instance_dir, "adns_demo.db").replace("\\", "/")
DEFAULT_DB_URI = _default_sqlite

app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("SQLALCHEMY_DATABASE_URI", DEFAULT_DB_URI)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Allow background scorer threads to use the same SQLite connection pool.
if app.config["SQLALCHEMY_DATABASE_URI"].startswith("sqlite"):
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "connect_args": {"check_same_thread": False},
    }

db = SQLAlchemy(app)

MAX_FLOWS = 400  # keep last N flows when responding to dashboard clients
FLOW_RETENTION_MINUTES = int(os.environ.get("ADNS_FLOW_RETENTION_MINUTES", "30"))
FLOW_RETENTION_MAX_ROWS = int(os.environ.get("ADNS_FLOW_RETENTION_MAX_ROWS", "5000"))
KILL_SWITCH_STATE = {"enabled": False}
USE_NSENTER = os.environ.get("ADNS_NSENTER_HOST", "true").lower() not in {"0", "false", "no"}
# block_ip / unblock_ip shell out to iptables on the host namespace and are
# gated behind a shared admin token. When the token is unset those endpoints
# stay disabled — fail closed rather than open.
# The killswitch is not token-gated: it is a first-responder dashboard action
# and must work without extra configuration.
ADMIN_TOKEN = os.environ.get("ADNS_ADMIN_TOKEN", "").strip()

# Prevents multiple concurrent streaming threads from accumulating (OOM risk)
_STREAM_LOCK = threading.Lock()
_stream_active = False


def _extract_request_token() -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("X-Admin-Token", "").strip()


def _require_admin_token_now():
    """Return an error response if the request lacks a valid admin token, else None."""
    if not ADMIN_TOKEN:
        return (
            jsonify(
                {
                    "error": "endpoint disabled",
                    "detail": "set ADNS_ADMIN_TOKEN to enable network-response actions",
                }
            ),
            403,
        )
    provided = _extract_request_token()
    if not provided or not hmac.compare_digest(provided, ADMIN_TOKEN):
        return jsonify({"error": "unauthorized"}), 401
    return None


def require_admin_token(view):
    """Guard destructive network-response endpoints with a shared token."""

    @wraps(view)
    def wrapper(*args, **kwargs):
        guard = _require_admin_token_now()
        if guard is not None:
            return guard
        return view(*args, **kwargs)

    return wrapper

PROTOCOL_MAP = {
    "1": "ICMP",
    "6": "TCP",
    "17": "UDP",
    "41": "ENCAP",
    "47": "GRE",
    "50": "ESP",
    "51": "AH",
    "58": "ICMPv6",
    "132": "SCTP",
}


class Flow(db.Model):
    __tablename__ = "flows"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime(timezone=True), nullable=False, index=True)
    src_ip = db.Column(db.String(64), nullable=False, index=True)
    dst_ip = db.Column(db.String(64), nullable=False, index=True)
    proto = db.Column(db.String(16), nullable=False)
    bytes = db.Column(db.Integer, nullable=False, default=0)
    extra = db.Column(db.JSON, nullable=True)

    predictions = db.relationship("Prediction", backref="flow", lazy="dynamic", cascade="all, delete-orphan")


class Prediction(db.Model):
    __tablename__ = "predictions"
    __table_args__ = (db.UniqueConstraint("flow_id", name="uq_predictions_flow_id"),)

    id = db.Column(db.Integer, primary_key=True)
    flow_id = db.Column(db.Integer, db.ForeignKey("flows.id"), nullable=False, index=True)
    score = db.Column(db.Float, nullable=True)
    label = db.Column(db.String(32), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class BlockedIP(db.Model):
    __tablename__ = "blocked_ips"

    id = db.Column(db.Integer, primary_key=True)
    ip = db.Column(db.String(64), unique=True, nullable=False)
    active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


def init_db() -> None:
    with app.app_context():
        db.create_all()
        if db.engine.dialect.name == "sqlite":
            db.session.execute(text("PRAGMA journal_mode=WAL"))
            db.session.commit()
        ensure_flow_extra_column()
        ensure_prediction_flow_unique_index()


def ensure_flow_extra_column() -> None:
    """
    Older deployments created the flows table before `extra` existed. Ensure the
    JSON column is present so inserts from the tshark agent succeed.
    """
    try:
        inspector = inspect(db.engine)
        columns = {col["name"] for col in inspector.get_columns("flows")}
    except SQLAlchemyError as exc:  # pragma: no cover - defensive
        app.logger.warning("failed to inspect flows table: %s", exc)
        return

    if "extra" in columns:
        return

    column_type = "JSONB" if db.engine.dialect.name == "postgresql" else "JSON"
    stmt = text(f"ALTER TABLE flows ADD COLUMN extra {column_type}")
    try:
        with db.engine.begin() as conn:
            conn.execute(stmt)
        app.logger.info("added flows.extra column (%s)", column_type)
    except SQLAlchemyError as exc:  # pragma: no cover - defensive
        app.logger.error("failed to add flows.extra column: %s", exc)


def ensure_prediction_flow_unique_index() -> None:
    """Guarantee that predictions.flow_id stays unique for ON CONFLICT logic."""
    index_name = "idx_predictions_flow_id_unique"
    try:
        inspector = inspect(db.engine)
        indexes = inspector.get_indexes("predictions")
    except SQLAlchemyError as exc:  # pragma: no cover - defensive
        app.logger.warning("failed to inspect predictions indexes: %s", exc)
        return

    for idx in indexes:
        columns = idx.get("column_names") or []
        if idx.get("unique") and columns == ["flow_id"]:
            return
        if idx.get("name") == index_name and idx.get("unique"):
            return

    # Prune duplicate rows so the unique index can be created safely.
    removed = 0
    duplicates = (
        db.session.query(Prediction.flow_id)
        .group_by(Prediction.flow_id)
        .having(db.func.count(Prediction.id) > 1)
        .all()
    )
    for (flow_id,) in duplicates:
        dup_ids = (
            db.session.query(Prediction.id)
            .filter(Prediction.flow_id == flow_id)
            .order_by(Prediction.id.asc())
            .all()
        )
        ids_to_delete = [row.id for row in dup_ids[1:]]
        if ids_to_delete:
            removed += (
                Prediction.query.filter(Prediction.id.in_(ids_to_delete)).delete(synchronize_session=False)
            )

    if removed:
        db.session.commit()
        app.logger.info("pruned %d duplicate prediction row(s) before enforcing uniqueness", removed)
    else:
        db.session.rollback()

    stmt = text(
        f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON predictions(flow_id)"
    )
    try:
        with db.engine.begin() as conn:
            conn.execute(stmt)
    except SQLAlchemyError as exc:  # pragma: no cover - defensive
        app.logger.error("failed to create unique predictions index: %s", exc)


def _run_cmd(cmd: list[str]) -> tuple[bool, str]:
    prefixed = cmd
    if USE_NSENTER and shutil.which("nsenter"):
        prefixed = ["nsenter", "-t", "1", "-n"] + cmd
    try:
        proc = subprocess.run(prefixed, check=True, capture_output=True, text=True)
        return True, proc.stdout.strip()
    except FileNotFoundError as exc:
        app.logger.error("command not found: %s", exc)
        return False, "command not found"
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        app.logger.error("command failed (%s): %s", prefixed, stderr)
        return False, stderr or "command failed"


_KILLSWITCH_RULE_IN = "ADNS Killswitch IN"
_KILLSWITCH_RULE_OUT = "ADNS Killswitch OUT"

# Blocks all non-loopback traffic so the monitoring stack stays reachable over localhost.
_KILLSWITCH_IPTABLES = [
    (
        ["iptables", "-C", "OUTPUT", "!", "-o", "lo", "-j", "DROP"],
        ["iptables", "-I", "OUTPUT", "!", "-o", "lo", "-j", "DROP"],
        ["iptables", "-D", "OUTPUT", "!", "-o", "lo", "-j", "DROP"],
    ),
    (
        ["iptables", "-C", "INPUT", "!", "-i", "lo", "-j", "DROP"],
        ["iptables", "-I", "INPUT", "!", "-i", "lo", "-j", "DROP"],
        ["iptables", "-D", "INPUT", "!", "-i", "lo", "-j", "DROP"],
    ),
]


def _ensure_killswitch_windows(enabled: bool) -> bool:
    ok = True
    for rule_name, direction in ((_KILLSWITCH_RULE_IN, "in"), (_KILLSWITCH_RULE_OUT, "out")):
        exists, _ = _run_cmd(["netsh", "advfirewall", "firewall", "show", "rule", f"name={rule_name}"])
        if enabled and not exists:
            success, _ = _run_cmd([
                "netsh", "advfirewall", "firewall", "add", "rule",
                f"name={rule_name}", f"dir={direction}",
                "action=block", "profile=any",
            ])
            ok = ok and success
        elif not enabled and exists:
            success, _ = _run_cmd(["netsh", "advfirewall", "firewall", "delete", "rule", f"name={rule_name}"])
            ok = ok and success
    return ok


def ensure_killswitch_rules_enabled(enabled: bool) -> bool:
    """Drop all non-loopback traffic. Returns True if all OS rules applied successfully.
    Requires NET_ADMIN (Linux) or an Administrator process (Windows)."""
    if sys.platform == "win32":
        return _ensure_killswitch_windows(enabled)
    ok = True
    for check_cmd, add_cmd, remove_cmd in _KILLSWITCH_IPTABLES:
        exists, _ = _run_cmd(check_cmd)
        if enabled and not exists:
            success, _ = _run_cmd(add_cmd)
            ok = ok and success
        elif not enabled and exists:
            success, _ = _run_cmd(remove_cmd)
            ok = ok and success
    return ok


def _block_ip_windows(ip: str, allow: bool) -> tuple[bool, str]:
    """Windows Firewall via netsh advfirewall. Requires an elevated (Administrator) process."""
    rule_in = f"ADNS Block {ip} in"
    rule_out = f"ADNS Block {ip} out"
    all_ok = True

    for rule_name, direction in ((rule_in, "in"), (rule_out, "out")):
        check_cmd = ["netsh", "advfirewall", "firewall", "show", "rule", f"name={rule_name}"]
        exists, _ = _run_cmd(check_cmd)
        if allow:
            if exists:
                ok, msg = _run_cmd(["netsh", "advfirewall", "firewall", "delete", "rule", f"name={rule_name}"])
                all_ok = all_ok and ok
        else:
            if not exists:
                ok, msg = _run_cmd([
                    "netsh", "advfirewall", "firewall", "add", "rule",
                    f"name={rule_name}",
                    f"dir={direction}",
                    "action=block",
                    f"remoteip={ip}",
                ])
                all_ok = all_ok and ok

    return all_ok, "unblocked" if allow else "blocked"


def _block_ip_iptables(ip: str, allow: bool) -> tuple[bool, str]:
    """iptables-based blocking for Linux. Requires NET_ADMIN / root."""
    rules = [
        (
            ["iptables", "-C", "INPUT", "-s", ip, "-j", "DROP"],
            ["iptables", "-I", "INPUT", "-s", ip, "-j", "DROP"],
            ["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"],
        ),
        (
            ["iptables", "-C", "OUTPUT", "-d", ip, "-j", "DROP"],
            ["iptables", "-I", "OUTPUT", "-d", ip, "-j", "DROP"],
            ["iptables", "-D", "OUTPUT", "-d", ip, "-j", "DROP"],
        ),
    ]

    all_ok = True
    messages: list[str] = []

    for check_cmd, add_cmd, remove_cmd in rules:
        exists, _ = _run_cmd(check_cmd)
        if allow:
            if exists:
                ok, msg = _run_cmd(remove_cmd)
                all_ok = all_ok and ok
                if msg:
                    messages.append(msg)
            continue
        if exists:
            continue
        ok, msg = _run_cmd(add_cmd)
        all_ok = all_ok and ok
        if msg:
            messages.append(msg)

    detail = "; ".join(messages) if messages else ""
    return all_ok, detail or ("unblocked" if allow else "blocked")


def block_ip_os(ip: str, allow: bool = False) -> tuple[bool, str]:
    """Apply or remove a firewall rule for the given IP. Best effort; requires elevated privileges."""
    if sys.platform == "win32":
        return _block_ip_windows(ip, allow)
    return _block_ip_iptables(ip, allow)


simulation_detector = DetectionEngine()


def _infer_scanning(flow) -> str | None:
    """
    Lightweight heuristic to nudge likely port scans when the model is neutral.
    Looks for 'scan' service hint or low-byte hits to privileged ports.
    """
    extra = flow.extra or {}
    service = str(extra.get("service", "")).lower()
    if "scan" in service:
        return "scanning"
    try:
        dst_port = int(extra.get("dst_port") or 0)
    except (TypeError, ValueError):
        dst_port = 0
    try:
        src_port = int(extra.get("src_port") or 0)
    except (TypeError, ValueError):
        src_port = 0
    bytes_total = float(flow.bytes or 0.0)
    if (dst_port and dst_port <= 1024) or (src_port and src_port <= 1024):
        if bytes_total <= 20000:
            return "scanning"
    return None

SIMULATION_TYPES = {
    "attack": {
        "label": "Generic attack",
        "default_count": 40,
    },
    "scanning": {
        "label": "Port scanning sweep",
        "default_count": 80,
    },
    "dos": {
        "label": "DoS burst",
        "default_count": 60,
    },
    "ddos": {
        "label": "DDoS swarm",
        "default_count": 120,
    },
    "injection": {
        "label": "Injection attempt",
        "default_count": 40,
    },
}


def _pattern_ip(pattern: str, rng: random.Random) -> str:
    parts = pattern.split(".")
    octets = []
    for part in parts:
        if part in {"x", "*"}:
            octets.append(str(rng.randint(1, 254)))
        elif part == "y":
            octets.append(str(rng.randint(0, 99)))
        else:
            octets.append(part)
    while len(octets) < 4:
        octets.append(str(rng.randint(1, 254)))
    return ".".join(octets[:4])


def generate_attack_flows(kind: str, count: int) -> list[Flow]:
    rng = random.Random()
    now = datetime.now(timezone.utc)
    flows: list[Flow] = []

    def _make_extra(proto: str, src_port: int, dst_port: int, byte_count: int, service_hint: str | None = None):
        total = max(0, int(byte_count))
        reply = int(total * rng.uniform(0.05, 0.3))
        return {
            "src_port": src_port,
            "dst_port": dst_port,
            "service": service_hint or proto.lower(),
            "duration": rng.uniform(2.0, 15.0),
            "src_bytes": total,
            "dst_bytes": reply,
            "src_pkts": max(3, total // 600),
            "dst_pkts": max(3, reply // 600),
        }

    def _add_flow(ts_offset: float, src: str, dst: str, proto: str, byte_count: int, extra: dict | None = None) -> None:
        flow = Flow(
            timestamp=now - timedelta(seconds=ts_offset),
            src_ip=src,
            dst_ip=dst,
            proto=normalize_protocol(proto),
            bytes=max(0, int(byte_count)),
        )
        flow.extra = extra
        flows.append(flow)

    for i in range(count):
        if kind == "ddos":
            dst = rng.choice(["198.51.100.42", "198.51.100.47", "203.0.113.10"])
            src = _pattern_ip("10.x.y.x", rng)
            bytes_val = rng.randint(180_000, 520_000)
            offset = rng.uniform(0, 90)
            src_port = rng.randint(1024, 65000)
            extra = _make_extra("tcp", src_port, rng.choice([80, 443]), bytes_val, "http")
            _add_flow(offset, src, dst, "TCP", bytes_val, extra)
        elif kind == "dos":
            src = rng.choice(["10.0.5.33", "10.0.5.34"])
            dst = rng.choice(["203.0.113.55", "203.0.113.56"])
            bytes_val = rng.randint(90_000, 180_000)
            offset = rng.uniform(0, 60)
            src_port = rng.randint(10_000, 60000)
            extra = _make_extra("tcp", src_port, 443, bytes_val, "https")
            _add_flow(offset, src, dst, "TCP", bytes_val, extra)
        elif kind == "scanning":
            src = rng.choice(["172.16.8.4", "172.16.8.5"])
            dst = f"192.168.{rng.randint(1, 10)}.{(i % 200) + 1}"
            proto = rng.choice(["UDP", "TCP"])
            bytes_val = rng.randint(800, 5000)
            offset = rng.uniform(0, 180)
            dst_port = rng.randint(1, 1024)
            src_port = rng.randint(2000, 9000)
            extra = _make_extra(proto.lower(), src_port, dst_port, bytes_val, "scan")
            _add_flow(offset, src, dst, proto, bytes_val, extra)
        elif kind == "injection":
            src = rng.choice(["10.12.11.7", "10.12.11.8"])
            dst = _pattern_ip("203.0.113.x", rng)
            bytes_val = rng.randint(4_000, 18_000)
            offset = rng.uniform(0, 45)
            dst_port = rng.choice([1433, 3306, 5432, 9200])
            src_port = rng.randint(30000, 65000)
            extra = _make_extra("tcp", src_port, dst_port, bytes_val, "sql")
            extra["http_method"] = "POST"
            extra["http_uri"] = "/login"
            _add_flow(offset, src, dst, "TCP", bytes_val, extra)
        elif kind == "attack":
            src = rng.choice(["10.0.5.33", "10.0.5.34"])
            dst = _pattern_ip("203.0.113.x", rng)
            bytes_val = rng.randint(160_000, 360_000)
            offset = rng.uniform(0, 120)
            src_port = rng.randint(20000, 60000)
            extra = _make_extra("tcp", src_port, 443, bytes_val, "https")
            _add_flow(offset, src, dst, "TCP", bytes_val, extra)
        else:
            raise ValueError(f"unsupported attack type '{kind}'")

    return flows


def parse_timestamp(value) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            # allow trailing Z
            cleaned = value.replace("Z", "+00:00") if value.endswith("Z") else value
            return datetime.fromisoformat(cleaned)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def latest_prediction_score(flow: Flow) -> float:
    pred = flow.predictions.order_by(Prediction.created_at.desc()).first()
    if pred and pred.score is not None:
        return float(pred.score)
    return 0.0


def normalize_protocol(value) -> str:
    if value is None:
        return "OTHER"
    text = str(value).strip()
    if not text:
        return "OTHER"
    if text.isdigit():
        return PROTOCOL_MAP.get(text, f"PROTO_{text}")
    return text.upper()


def _coerce_int(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    base = 16 if text.lower().startswith("0x") else 10
    try:
        return int(text, base)
    except ValueError:
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            return int(digits)
    return None


def _coerce_float(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _clean_text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


EXTRA_INT_FIELDS = {
    "src_port",
    "dst_port",
    "src_bytes",
    "dst_bytes",
    "src_pkts",
    "dst_pkts",
    "dns_qclass",
    "dns_qtype",
    "dns_rcode",
    "http_status_code",
    "http_request_body_len",
    "http_response_body_len",
}

EXTRA_FLOAT_FIELDS = {"duration"}

EXTRA_TEXT_FIELDS = {
    "dns_query",
    "http_method",
    "http_uri",
    "http_referrer",
    "http_version",
    "http_user_agent",
    "http_orig_mime_types",
    "http_resp_mime_types",
    "weird_name",
    "weird_addl",
    "weird_notice",
    "ssl_cipher",
}


def build_flow_extra(rec: dict) -> dict | None:
    extra: dict = {}
    for field in EXTRA_INT_FIELDS:
        val = _coerce_int(rec.get(field))
        if val is not None:
            extra[field] = val

    for field in EXTRA_FLOAT_FIELDS:
        val = _coerce_float(rec.get(field))
        if val is not None:
            extra[field] = val

    for field in EXTRA_TEXT_FIELDS:
        val = _clean_text(rec.get(field))
        if val:
            extra[field] = val

    service = _clean_text(rec.get("service"))
    if service:
        extra["service"] = service.lower()

    ssl_version = _coerce_int(rec.get("ssl_version"))
    if ssl_version is not None:
        extra["ssl_version"] = ssl_version

    # allow callers to set explicit src/dst jitter values later if desired
    return extra or None


def flow_to_dict(flow: Flow) -> dict:
    latest_label = "normal"
    pred = flow.predictions.order_by(Prediction.created_at.desc()).first()
    score = 0.0
    if pred:
        latest_label = pred.label
        score = pred.score or 0.0

    attack_type = None
    if latest_label and latest_label.lower() not in {"normal", "watch", "anomaly"}:
        attack_type = latest_label
    if not attack_type:
        extra_attack = (flow.extra or {}).get("attack_type")
        if extra_attack:
            attack_type = extra_attack

    return {
        "id": flow.id,
        "ts": flow.timestamp.isoformat(),
        "src_ip": flow.src_ip,
        "dst_ip": flow.dst_ip,
        "proto": normalize_protocol(flow.proto),
        "bytes": flow.bytes,
        "score": float(score),
        "label": latest_label,
        "attack_type": attack_type,
        "extra": flow.extra or {},
    }


def is_anomalous_flow(flow: Flow) -> bool:
    pred = flow.predictions.order_by(Prediction.created_at.desc()).first()
    if not pred:
        return False
    label = (pred.label or "").lower()
    if label and label != "normal":
        return True
    return float(pred.score or 0.0) >= 0.6


def get_recent_flows(limit: int = MAX_FLOWS) -> list:
    flows = Flow.query.order_by(Flow.timestamp.desc()).limit(limit).all()
    # maintain chronological order (oldest first) for the dashboard
    return list(reversed(flows))


def enforce_flow_retention() -> int:
    purged = 0
    batch_size = 1000

    def delete_flow_batch(id_list: list[int]) -> int:
        if not id_list:
            return 0
        Prediction.query.filter(Prediction.flow_id.in_(id_list)).delete(synchronize_session=False)
        return Flow.query.filter(Flow.id.in_(id_list)).delete(synchronize_session=False)

    if FLOW_RETENTION_MINUTES > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=FLOW_RETENTION_MINUTES)
        while True:
            stale_ids = (
                Flow.query.with_entities(Flow.id)
                .filter(Flow.timestamp < cutoff)
                .limit(batch_size)
                .all()
            )
            id_list = [row.id for row in stale_ids]
            if not id_list:
                break
            purged += delete_flow_batch(id_list)

    if FLOW_RETENTION_MAX_ROWS > 0:
        total = Flow.query.count()
        if total > FLOW_RETENTION_MAX_ROWS:
            excess = total - FLOW_RETENTION_MAX_ROWS
            while excess > 0:
                chunk = min(excess, batch_size)
                oldest_ids = (
                    Flow.query.order_by(Flow.timestamp.asc())
                    .with_entities(Flow.id)
                    .limit(chunk)
                    .all()
                )
                id_list = [row.id for row in oldest_ids]
                if not id_list:
                    break
                purged += delete_flow_batch(id_list)
                excess -= len(id_list)

    if purged:
        db.session.commit()
    return purged


init_db()

# ---------------------------------------------------------------
# Basic Health Check
# ---------------------------------------------------------------
@app.get("/health")
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------
# Ingest endpoint for tshark agent
# ---------------------------------------------------------------
@app.route("/ingest", methods=["POST"])
def ingest():
    """
    Accepts either:
      - a single flow object
      - a list of flow objects
    and persists them in the flows table.
    """
    payload = request.get_json(force=True, silent=False)

    if isinstance(payload, dict):
        batch = [payload]
    elif isinstance(payload, list):
        batch = payload
    else:
        return jsonify({"error": "invalid payload"}), 400

    blocked = 0
    created = 0
    flow_records: list[Flow] = []

    blocked_set = {row.ip for row in BlockedIP.query.filter_by(active=True).all()}
    for rec in batch:
        src_ip = rec.get("src_ip", "")
        dst_ip = rec.get("dst_ip", "")

        if src_ip in blocked_set or dst_ip in blocked_set:
            blocked += 1
            continue

        extra = build_flow_extra(rec)
        flow = Flow(
            timestamp=parse_timestamp(rec.get("ts")),
            src_ip=src_ip,
            dst_ip=dst_ip,
            proto=normalize_protocol(rec.get("proto", "")),
            bytes=int(rec.get("bytes") or 0),
            extra=extra,
        )
        flow_records.append(flow)
        db.session.add(flow)
        created += 1

    try:
        flow_ids: list[int] = []
        if flow_records:
            db.session.flush()
            flow_ids = [flow.id for flow in flow_records]

        db.session.commit()
    except Exception as exc:  # pragma: no cover
        db.session.rollback()
        app.logger.exception("failed to insert flows: %s", exc)
        return jsonify({"error": "database insert failed"}), 500

    purged = enforce_flow_retention()
    if purged:
        app.logger.info("purged %d old flow(s)", purged)

    enqueued = 0
    if flow_ids:
        try:
            enqueued = enqueue_flow_scoring(flow_ids)
        except Exception as exc:  # pragma: no cover
            app.logger.exception("failed to submit flows for background scoring: %s", exc)

    return jsonify(
        {"status": "ok", "ingested": created, "blocked": blocked, "purged": purged, "queued": enqueued}
    )


@app.post("/simulate")
def simulate_attack():
    payload = request.get_json(silent=True) or {}
    attack_type = str(payload.get("type") or "botnet_flood").strip()
    profile = SIMULATION_TYPES.get(attack_type)
    if not profile:
        return jsonify({"error": f"unknown attack type '{attack_type}'"}), 400

    requested_count = payload.get("count")
    default_count = profile["default_count"]
    try:
        count = int(requested_count) if requested_count is not None else default_count
    except (TypeError, ValueError):
        return jsonify({"error": "count must be an integer"}), 400
    count = max(5, min(count, 250))

    duration_seconds = payload.get("duration_seconds", payload.get("duration"))
    try:
        duration_seconds = int(duration_seconds) if duration_seconds is not None else 0
    except (TypeError, ValueError):
        duration_seconds = 0
    duration_seconds = max(0, min(duration_seconds, 600))
    interval_seconds = payload.get("interval_seconds")
    try:
        interval_seconds = float(interval_seconds) if interval_seconds is not None else 1.0
    except (TypeError, ValueError):
        interval_seconds = 1.0
    interval_seconds = max(0.5, min(interval_seconds, 5.0))

    if duration_seconds > 0:
        global _stream_active
        with _STREAM_LOCK:
            if _stream_active:
                return (
                    jsonify({"error": "a streaming simulation is already running; wait for it to finish"}),
                    409,
                )
            _stream_active = True
        batch_size = max(5, min(count, 200))

        def _stream_simulation() -> None:
            global _stream_active
            try:
                deadline = time.time() + duration_seconds
                with app.app_context():
                    total_generated = 0
                    while time.time() < deadline:
                        flows = generate_attack_flows(attack_type, batch_size)
                        for flow in flows:
                            db.session.add(flow)
                        db.session.flush()

                        for flow in flows:
                            pred = simulation_detector.predict(db.session, flow)
                            if isinstance(pred, (list, tuple)) and len(pred) == 3:
                                score, label, attack_label = pred
                            else:
                                score, label = pred
                                attack_label = None
                            base_labels = {"normal", "watch", "anomaly"}
                            candidate_attack = None
                            if label and label.lower() not in base_labels:
                                candidate_attack = label
                            elif attack_label and label and label.lower() != "normal":
                                candidate_attack = attack_label
                            elif label and label.lower() in {"normal", "watch"}:
                                candidate_attack = _infer_scanning(flow)
                            extras = dict(flow.extra or {})
                            if candidate_attack and candidate_attack.lower() not in base_labels:
                                extras["attack_type"] = candidate_attack
                            else:
                                extras.pop("attack_type", None)
                            flow.extra = extras
                            db.session.add(
                                Prediction(
                                    flow_id=flow.id,
                                    score=score,
                                    label=label,
                                    created_at=datetime.now(timezone.utc),
                                )
                            )

                        db.session.commit()
                        db.session.expunge_all()  # free identity-map refs between batches
                        enforce_flow_retention()
                        total_generated += len(flows)
                        time.sleep(interval_seconds)
                    app.logger.info(
                        "completed streaming simulate: %s flows over %s seconds",
                        total_generated,
                        duration_seconds,
                    )
            finally:
                with _STREAM_LOCK:
                    _stream_active = False

        threading.Thread(target=_stream_simulation, daemon=True).start()
        return jsonify(
            {
                "status": "streaming",
                "type": attack_type,
                "label": profile["label"],
                "duration_seconds": duration_seconds,
                "batch_size": batch_size,
                "interval_seconds": interval_seconds,
            }
        )

    flows = generate_attack_flows(attack_type, count)
    for flow in flows:
        db.session.add(flow)
    db.session.flush()

    scores: list[float] = []
    for flow in flows:
        pred = simulation_detector.predict(db.session, flow)
        if isinstance(pred, (list, tuple)) and len(pred) == 3:
            score, label, attack_label = pred
        else:
            score, label = pred
            attack_label = None
        scores.append(score)
        base_labels = {"normal", "watch", "anomaly"}
        candidate_attack = None
        if label and label.lower() not in base_labels:
            candidate_attack = label
        elif attack_label and label and label.lower() != "normal":
            candidate_attack = attack_label
        elif label and label.lower() in {"normal", "watch"}:
            candidate_attack = _infer_scanning(flow)
        extras = dict(flow.extra or {})
        if candidate_attack and candidate_attack.lower() not in base_labels:
            extras["attack_type"] = candidate_attack
        else:
            extras.pop("attack_type", None)
        flow.extra = extras
        db.session.add(
            Prediction(
                flow_id=flow.id,
                score=score,
                label=label,
                created_at=datetime.now(timezone.utc),
            )
        )

    db.session.commit()
    purged = enforce_flow_retention()

    return jsonify(
        {
            "status": "ok",
            "type": attack_type,
            "label": profile["label"],
            "generated": len(flows),
            "max_score": round(max(scores) if scores else 0.0, 3),
            "purged": purged,
        }
    )


# ---------------------------------------------------------------
# Flows endpoint (dashboard)
#  - uses live buffer if present
#  - falls back to demo data if empty
# ---------------------------------------------------------------
@app.get("/flows")
def flows():
    recent = get_recent_flows()
    if recent:
        payload = [flow_to_dict(f) for f in recent]
        return jsonify(payload)

    demo_flows = [
        {
            "ts": "2025-11-17T11:10:00Z",
            "src_ip": "192.168.1.10",
            "dst_ip": "8.8.8.8",
            "proto": "TCP",
            "bytes": 1500,
            "score": 0.12,
            "label": "normal",
        },
        {
            "ts": "2025-11-17T11:10:05Z",
            "src_ip": "10.0.0.5",
            "dst_ip": "172.217.3.110",
            "proto": "TCP",
            "bytes": 4200,
            "score": 0.98,
            "label": "ddos",
        },
        {
            "ts": "2025-11-17T11:10:09Z",
            "src_ip": "192.168.1.23",
            "dst_ip": "1.1.1.1",
            "proto": "UDP",
            "bytes": 800,
            "score": 0.45,
            "label": "scanning",
        },
    ]
    return jsonify(demo_flows)


@app.get("/anomalous_flows")
def anomalous_flows():
    recent = get_recent_flows()
    if not recent:
        return jsonify([])
    anomalies = [flow for flow in recent if is_anomalous_flow(flow)]
    payload = [flow_to_dict(f) for f in anomalies]
    return jsonify(payload)


# ---------------------------------------------------------------
# Anomaly stats (for now: simple derived stats from buffer or demo)
# ---------------------------------------------------------------
@app.get("/anomalies")
def anomalies():
    data = get_recent_flows()
    if not data:
        # same demo stats as before if nothing ingested yet
        demo_stats = {
            "window": "last 10 min",
            "count": 7,
            "max_score": 0.992,
            "pct_anomalous": 3.1,
        }
        return jsonify(demo_stats)

    scores = [latest_prediction_score(f) for f in data]
    total = len(scores)
    max_score = max(scores) if scores else 0.0
    anomaly_count = sum(1 for s in scores if s > 0.9)
    pct = (anomaly_count / total * 100.0) if total > 0 else 0.0

    stats = {
        "window": "recent buffer",
        "count": anomaly_count,
        "max_score": round(max_score, 3),
        "pct_anomalous": round(pct, 2),
    }
    return jsonify(stats)


@app.post("/block_ip")
def block_ip():
    payload = request.get_json(silent=True) or {}
    ip = str(payload.get("ip") or "").strip()
    if not ip:
        return jsonify({"error": "ip is required"}), 400

    record = BlockedIP.query.filter_by(ip=ip).first()
    now = datetime.now(timezone.utc)
    if record:
        record.active = True
        record.created_at = now
    else:
        db.session.add(BlockedIP(ip=ip, active=True, created_at=now))
    db.session.commit()

    # OS-level iptables block — only attempted when admin token is configured and caller provides it
    os_status = "not_configured"
    provided = _extract_request_token()
    if ADMIN_TOKEN and provided and hmac.compare_digest(provided, ADMIN_TOKEN):
        ok, msg = block_ip_os(ip, allow=False)
        os_status = "ok" if ok else "failed"
    return jsonify({"status": "blocked", "ip": ip, "os_action": os_status})


@app.get("/blocked_ips")
def blocked_ips():
    rows = BlockedIP.query.filter_by(active=True).order_by(BlockedIP.created_at.desc()).all()
    payload = [{"ip": row.ip, "created_at": row.created_at.isoformat()} for row in rows]
    return jsonify(payload)


@app.post("/unblock_ip")
def unblock_ip():
    payload = request.get_json(silent=True) or {}
    ip = str(payload.get("ip") or "").strip()
    if not ip:
        return jsonify({"error": "ip is required"}), 400

    record = BlockedIP.query.filter_by(ip=ip).first()
    if record:
        record.active = False
        db.session.commit()

    os_status = "not_configured"
    provided = _extract_request_token()
    if ADMIN_TOKEN and provided and hmac.compare_digest(provided, ADMIN_TOKEN):
        ok, msg = block_ip_os(ip, allow=True)
        os_status = "ok" if ok else "failed"
    return jsonify({"status": "unblocked", "ip": ip, "os_action": os_status})


@app.route("/killswitch", methods=["GET", "POST"])
def killswitch():
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        enabled = bool(payload.get("enabled"))
        os_ok = ensure_killswitch_rules_enabled(enabled)
        if os_ok:
            KILL_SWITCH_STATE["enabled"] = enabled
        return jsonify({
            "enabled": bool(KILL_SWITCH_STATE.get("enabled", False)),
            "os_action": "ok" if os_ok else "failed",
        })
    return jsonify({"enabled": bool(KILL_SWITCH_STATE.get("enabled", False))})


# ---------------------------------------------------------------
# Frontend static file serving (desktop / self-contained mode)
# Set ADNS_FRONTEND_DIST to the React dist/ directory to enable.
# In dev mode the Vite dev server handles this; this route is a no-op.
# ---------------------------------------------------------------
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    dist_dir = os.environ.get("ADNS_FRONTEND_DIST", "")
    if not dist_dir:
        return jsonify({"status": "api-only mode"}), 200
    target = os.path.join(dist_dir, path) if path else None
    if path and target and os.path.isfile(target):
        return send_from_directory(dist_dir, path)
    return send_from_directory(dist_dir, "index.html")


# ---------------------------------------------------------------
# Main Entrypoint (for direct run; Gunicorn ignores this block)
# ---------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
