from __future__ import annotations

import hashlib
import ipaddress
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "model_artifacts" / "flow_detector.joblib"
DEFAULT_META_MODEL_PATH = BASE_DIR / "model_artifacts" / "meta_model_combined.joblib"

logger = logging.getLogger(__name__)


def _is_private_ip(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_private
    except ValueError:
        return False


def _timestamp_to_epoch(ts: datetime | None) -> float:
    if ts is None:
        return 0.0
    if ts.tzinfo is not None:
        return float(ts.timestamp())
    return float(ts.replace(tzinfo=None).timestamp())


class FlowModel:
    """Wrapper around the legacy sklearn pipeline with byte/proto features."""

    def __init__(self, model_path: str | os.PathLike | None = None) -> None:
        resolved = Path(model_path or os.environ.get("ADNS_MODEL_PATH", DEFAULT_MODEL_PATH))
        if not resolved.exists():
            raise FileNotFoundError(f"model artifact not found at {resolved}")
        payload = joblib.load(resolved)
        self.pipeline = payload["model"]
        self.anomaly_threshold = float(payload.get("threshold_anomaly", 0.6))
        self.watch_threshold = float(
            payload.get("threshold_watch", max(0.2, self.anomaly_threshold * 0.65))
        )
        self.model_path = resolved

    def _feature_dict(self, bytes_count: int, proto: str) -> dict:
        total_bytes = max(0.0, float(bytes_count or 0))
        proto_norm = (proto or "OTHER").upper()
        return {
            "total_bytes": total_bytes,
            "log_total_bytes": math.log1p(total_bytes),
            "proto": proto_norm,
        }

    def _label_for_probability(self, prob: float) -> str:
        if prob >= self.anomaly_threshold:
            return "anomaly"
        if prob >= self.watch_threshold:
            return "watch"
        return "normal"

    def score(self, flow) -> Tuple[float, str]:
        results = self.score_many([flow])
        return results[0] if results else (0.0, "normal")

    def score_many(self, flows: Sequence) -> list[Tuple[float, str, str | None]]:
        if not flows:
            return []

        rows = []
        for flow in flows:
            bytes_count = getattr(flow, "bytes", flow)
            proto = getattr(flow, "proto", getattr(flow, "protocol", ""))
            rows.append(self._feature_dict(bytes_count, proto))
        frame = pd.DataFrame(rows)
        probabilities = self.pipeline.predict_proba(frame)[:, 1]
        return [
            (float(prob), self._label_for_probability(float(prob)))
            for prob in probabilities
        ]


@dataclass
class DirectionalBytes:
    src: float
    dst: float


class MetaFeatureBuilder:
    """
    Builds a feature vector compatible with the combined ExtraTrees/XGBoost
    model bundle. Many TON_IoT columns are unavailable in live telemetry, so we
    approximate or default them to zero until the agent emits richer metadata.
    """

    FEATURE_COLUMNS: Tuple[str, ...] = (
        "ts",
        "src_ip",
        "src_port",
        "dst_ip",
        "dst_port",
        "proto",
        "service",
        "duration",
        "src_bytes",
        "dst_bytes",
        "conn_state",
        "missed_bytes",
        "src_pkts",
        "src_ip_bytes",
        "dst_pkts",
        "dst_ip_bytes",
        "dns_query",
        "dns_qclass",
        "dns_qtype",
        "dns_rcode",
        "dns_AA",
        "dns_RD",
        "dns_RA",
        "dns_rejected",
        "ssl_version",
        "ssl_cipher",
        "ssl_resumed",
        "ssl_established",
        "ssl_subject",
        "ssl_issuer",
        "http_trans_depth",
        "http_method",
        "http_uri",
        "http_referrer",
        "http_version",
        "http_request_body_len",
        "http_response_body_len",
        "http_status_code",
        "http_user_agent",
        "http_orig_mime_types",
        "http_resp_mime_types",
        "weird_name",
        "weird_addl",
        "weird_notice",
        "rdns_exists",
        "rdns_hash",
    )

    PROTO_CODE = {
        "ICMP": 1,
        "TCP": 6,
        "UDP": 17,
        "GRE": 47,
        "ESP": 50,
        "AH": 51,
        "SCTP": 132,
    }

    def __init__(self, avg_packet_bytes: int = 450) -> None:
        self.avg_packet_bytes = max(64, avg_packet_bytes)

    def build(self, flow) -> pd.DataFrame:
        row = self._build_row(flow)
        return pd.DataFrame([row], columns=self.FEATURE_COLUMNS, dtype="float32")

    def build_batch(self, flows: Sequence) -> pd.DataFrame:
        if not flows:
            return pd.DataFrame(columns=self.FEATURE_COLUMNS, dtype="float32")
        rows = [self._build_row(flow) for flow in flows]
        return pd.DataFrame(rows, columns=self.FEATURE_COLUMNS, dtype="float32")

    def _build_row(self, flow) -> dict:
        row = {col: 0.0 for col in self.FEATURE_COLUMNS}
        row["ts"] = _timestamp_to_epoch(getattr(flow, "timestamp", None))
        extra = getattr(flow, "extra", None) or {}

        src_ip = getattr(flow, "src_ip", "") or ""
        dst_ip = getattr(flow, "dst_ip", "") or ""
        row["src_ip"] = float(self._ip_to_int(src_ip))
        row["dst_ip"] = float(self._ip_to_int(dst_ip))

        proto = (getattr(flow, "proto", "") or "").upper()
        row["proto"] = float(self._proto_code(proto))

        service_hint = extra.get("service")
        if service_hint:
            row["service"] = self._encode_text(str(service_hint).lower(), modulus=4099)
        else:
            row["service"] = float(self._stable_hash(proto or "unknown") % 997)
        row["conn_state"] = float(self._stable_hash(f"{proto}:{self._direction_tag(src_ip, dst_ip)}") % 509)

        src_bytes_extra = self._safe_float(extra.get("src_bytes"))
        dst_bytes_extra = self._safe_float(extra.get("dst_bytes"))
        if src_bytes_extra is not None or dst_bytes_extra is not None:
            directional = DirectionalBytes(
                src=max(0.0, src_bytes_extra or 0.0),
                dst=max(0.0, dst_bytes_extra or 0.0),
            )
            total_bytes = directional.src + directional.dst
        else:
            total_bytes = max(0.0, float(getattr(flow, "bytes", 0) or 0))
            directional = self._split_directional_bytes(total_bytes, src_ip, dst_ip)

        duration_extra = self._safe_float(extra.get("duration"))
        row["duration"] = max(0.01, duration_extra) if duration_extra is not None else self._estimate_duration(total_bytes)

        row["src_bytes"] = directional.src
        row["dst_bytes"] = directional.dst
        row["src_ip_bytes"] = directional.src
        row["dst_ip_bytes"] = directional.dst

        src_pkts_extra = self._safe_float(extra.get("src_pkts"))
        dst_pkts_extra = self._safe_float(extra.get("dst_pkts"))
        row["src_pkts"] = max(0.0, src_pkts_extra) if src_pkts_extra is not None else self._estimate_packets(directional.src)
        row["dst_pkts"] = max(0.0, dst_pkts_extra) if dst_pkts_extra is not None else self._estimate_packets(directional.dst)

        for field in ("src_port", "dst_port", "dns_qclass", "dns_qtype", "dns_rcode", "http_status_code",
                      "dns_AA", "dns_RD", "dns_RA", "dns_rejected"):
            val = self._safe_int(extra.get(field))
            if val is not None:
                row[field] = float(val)

        for field in ("http_request_body_len", "http_response_body_len"):
            val = self._safe_float(extra.get(field))
            if val is not None:
                row[field] = float(val)

        dns_query = extra.get("dns_query")
        if dns_query:
            row["dns_query"] = self._encode_text(dns_query, modulus=20011)

        http_method = extra.get("http_method")
        if http_method:
            row["http_method"] = self._encode_text(http_method.upper(), modulus=4001)

        http_uri = extra.get("http_uri")
        if http_uri:
            row["http_uri"] = self._encode_text(http_uri, modulus=65521)

        http_referrer = extra.get("http_referrer")
        if http_referrer:
            row["http_referrer"] = self._encode_text(http_referrer, modulus=49157)

        http_version = extra.get("http_version")
        if http_version:
            row["http_version"] = self._encode_text(http_version, modulus=10139)

        http_user_agent = extra.get("http_user_agent")
        if http_user_agent:
            row["http_user_agent"] = self._encode_text(http_user_agent, modulus=45007)

        http_orig = extra.get("http_orig_mime_types")
        if http_orig:
            row["http_orig_mime_types"] = self._encode_text(http_orig, modulus=32749)

        http_resp = extra.get("http_resp_mime_types")
        if http_resp:
            row["http_resp_mime_types"] = self._encode_text(http_resp, modulus=32749)

        for field in ("weird_name", "weird_addl", "weird_notice"):
            value = extra.get(field)
            if value:
                row[field] = self._encode_text(value, modulus=50021)

        ssl_version = self._safe_int(extra.get("ssl_version"))
        if ssl_version is not None:
            row["ssl_version"] = float(ssl_version)
        ssl_cipher = extra.get("ssl_cipher")
        if ssl_cipher:
            row["ssl_cipher"] = self._encode_text(ssl_cipher, modulus=65267)

        rdns_exists = extra.get("rdns_exists")
        if rdns_exists is not None:
            row["rdns_exists"] = 1.0 if bool(rdns_exists) else 0.0

        rdns_hash = self._safe_int(extra.get("rdns_hash"))
        if rdns_hash is not None:
            row["rdns_hash"] = float(rdns_hash % 10007)

        return row

    def _ip_to_int(self, value: str) -> int:
        try:
            return int(ipaddress.ip_address(value))
        except ValueError:
            if not value:
                return 0
            return self._stable_hash(value) % (2**31)

    def _proto_code(self, proto: str) -> int:
        if proto.isdigit():
            return int(proto)
        return self.PROTO_CODE.get(proto, 0)

    def _stable_hash(self, value: str) -> int:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _direction_tag(self, src: str, dst: str) -> str:
        src_priv = _is_private_ip(src)
        dst_priv = _is_private_ip(dst)
        if src_priv and not dst_priv:
            return "outbound"
        if not src_priv and dst_priv:
            return "inbound"
        if src_priv and dst_priv:
            return "internal"
        return "external"

    def _split_directional_bytes(self, total: float, src: str, dst: str) -> DirectionalBytes:
        direction = self._direction_tag(src, dst)
        if direction == "outbound":
            return DirectionalBytes(src=total, dst=0.0)
        if direction == "inbound":
            return DirectionalBytes(src=0.0, dst=total)
        half = total / 2.0
        return DirectionalBytes(src=half, dst=half)

    def _estimate_packets(self, directional_bytes: float) -> float:
        if directional_bytes <= 0:
            return 0.0
        return max(1.0, directional_bytes / float(self.avg_packet_bytes))

    def _estimate_duration(self, total_bytes: float) -> float:
        if total_bytes <= 0:
            return 0.01
        return max(0.01, total_bytes / 120_000.0)

    @staticmethod
    def _safe_int(value) -> int | None:
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
            return int(digits) if digits else None

    @staticmethod
    def _safe_float(value) -> float | None:
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

    def _encode_text(self, value: str, modulus: int = 10007) -> float:
        return float(self._stable_hash(value) % modulus)


class MetaEnsembleModel:
    """Loads the combined ExtraTrees/XGBoost bundle and produces an averaged score."""

    CLASS_LABELS = {
        0: "normal",
        1: "attack",
        2: "scanning",
        3: "dos",
        4: "injection",
        5: "ddos",
    }

    def __init__(self, model_path: str | os.PathLike | None = None) -> None:
        resolved = Path(model_path or os.environ.get("ADNS_META_MODEL_PATH", DEFAULT_META_MODEL_PATH))
        if not resolved.exists():
            raise FileNotFoundError(f"meta model artifact not found at {resolved}")

        try:
            payload = joblib.load(resolved)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Failed to load meta model bundle because xgboost is missing. "
                "Install xgboost in the API environment."
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError("Unexpected artifact format: expected a dict of estimators.")

        self.models: dict[str, object] = {}
        if "xgboost" in payload:
            xgb_model = payload["xgboost"]
            # Some serialized bundles reference fields that newer xgboost drops;
            # set safe defaults so get_params/predict keep working.
            for attr, default in {
                "use_label_encoder": False,
                "gpu_id": None,
                "predictor": "auto",
            }.items():
                if not hasattr(xgb_model, attr):
                    try:
                        setattr(xgb_model, attr, default)
                    except Exception:  # pragma: no cover
                        pass
            self.models["xgboost"] = xgb_model
        if "extra_trees" in payload:
            self.models["extra_trees"] = payload["extra_trees"]

        if not self.models:
            raise ValueError("meta model bundle did not contain a supported estimator key.")

        self.features = MetaFeatureBuilder()
        self.anomaly_threshold = float(os.environ.get("ADNS_META_ANOMALY_THRESHOLD", "0.82"))
        self.watch_threshold = float(os.environ.get("ADNS_META_WATCH_THRESHOLD", "0.6"))
        self.model_path = resolved

    def _label_for_probability(self, prob: float) -> str:
        if prob >= self.anomaly_threshold:
            return "anomaly"
        if prob >= self.watch_threshold:
            return "watch"
        return "normal"

    def score(self, flow) -> Tuple[float, str]:
        results = self.score_many([flow])
        if not results:
            raise RuntimeError("meta model produced no results")
        return results[0]

    def score_many(self, flows: Sequence) -> list[Tuple[float, str]]:
        if not flows:
            return []

        feature_frame = self.features.build_batch(flows)
        values = feature_frame.to_numpy(dtype="float32")
        probabilities: list[np.ndarray] = []
        class_sets: list[np.ndarray] = []

        for name, model in self.models.items():
            vector = self._match_shape(values, getattr(model, "n_features_in_", values.shape[1]))
            try:
                probs = model.predict_proba(vector)
            except AttributeError:
                logits = model.predict(vector)
                probs = np.array(logits).reshape(-1, 1)

            probs = np.asarray(probs, dtype="float32")
            classes = getattr(model, "classes_", None)
            if classes is None:
                classes = np.arange(probs.shape[1], dtype="int64")
            class_sets.append(np.asarray(classes, dtype="int64"))
            probabilities.append(probs)

        if not probabilities:
            raise RuntimeError("no usable estimators were loaded from the meta model bundle")

        all_classes = sorted({int(c) for classes in class_sets for c in classes})
        class_index = {cls: idx for idx, cls in enumerate(all_classes)}
        combined = np.zeros((values.shape[0], len(all_classes)), dtype="float32")
        class_counts = np.zeros(len(all_classes), dtype="float32")

        for classes, probs in zip(class_sets, probabilities):
            for col_idx, class_id in enumerate(classes):
                target_idx = class_index.get(int(class_id))
                if target_idx is None:
                    continue
                combined[:, target_idx] += probs[:, col_idx]
                class_counts[target_idx] += 1

        for idx, cnt in enumerate(class_counts):
            if cnt > 0:
                combined[:, idx] /= cnt

        results: list[Tuple[float, str, str | None]] = []
        for row in combined:
            top_idx = int(np.argmax(row))
            top_class = all_classes[top_idx]
            top_score = float(row[top_idx])
            label = self.CLASS_LABELS.get(top_class, f"class_{top_class}")
            attack_label: str | None = None
            if len(all_classes) > 1:
                best_non_normal = sorted(
                    ((prob, cls_id) for cls_id, prob in zip(all_classes, row) if cls_id != 0),
                    key=lambda item: item[0],
                    reverse=True,
                )
                if best_non_normal:
                    _, best_cls = best_non_normal[0]
                    attack_label = self.CLASS_LABELS.get(int(best_cls), f"class_{best_cls}")
            results.append((top_score, label, attack_label))
        return results

    @staticmethod
    def _match_shape(arr: np.ndarray, expected: int) -> np.ndarray:
        current = arr.shape[1]
        if current == expected:
            return arr
        if current > expected:
            return arr[:, :expected]
        pad = expected - current
        return np.pad(arr, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)


class DetectionEngine:
    """
    Attempts to load the new meta ensemble first, then the legacy FlowModel,
    finally the heuristic FlowScorer if no artifacts are provisioned.
    """

    def __init__(self) -> None:
        self._mode = "heuristic"
        self.model = None
        self._artifact_mtimes: dict[str, float] = {}
        self._load_model()

    @property
    def mode(self) -> str:
        return self._mode

    def reload(self) -> None:
        self._load_model()

    def reload_if_stale(self) -> None:
        current = self._capture_artifact_mtimes()
        if current != self._artifact_mtimes:
            logger.info("reloading detection engine after model artifact change")
            self._load_model()

    def predict(self, session, flow):
        self.reload_if_stale()
        if self._mode in {"meta", "ml"}:
            return self.model.score(flow)
        return self.model.predict(session, flow)

    def predict_many(self, session, flows: Sequence) -> list[Tuple[float, str]]:
        if not flows:
            return []
        self.reload_if_stale()
        if self._mode in {"meta", "ml"}:
            return self.model.score_many(flows)
        return [self.model.predict(session, flow) for flow in flows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        loaders: Iterable[tuple[str, Callable[[], object]]] = (
            ("meta", MetaEnsembleModel),
            ("ml", FlowModel),
        )

        for mode, factory in loaders:
            try:
                self.model = factory()
                self._mode = mode
                break
            except FileNotFoundError:
                continue
        else:
            from scoring import FlowScorer  # deferred import to avoid optional deps at import time

            self.model = FlowScorer()
            self._mode = "heuristic"

        self._artifact_mtimes = self._capture_artifact_mtimes()
        logger.info("DetectionEngine initialized in %s mode", self._mode)

    def _candidate_paths(self) -> list[Path]:
        paths: list[Path] = []
        meta_path = os.environ.get("ADNS_META_MODEL_PATH")
        ml_path = os.environ.get("ADNS_MODEL_PATH")
        paths.append(Path(meta_path) if meta_path else DEFAULT_META_MODEL_PATH)
        paths.append(Path(ml_path) if ml_path else DEFAULT_MODEL_PATH)
        return [p for p in paths if p]

    def _capture_artifact_mtimes(self) -> dict[str, float]:
        mtimes: dict[str, float] = {}
        for path in self._candidate_paths():
            try:
                resolved = path.resolve()
            except FileNotFoundError:
                continue
            if resolved.exists():
                mtimes[str(resolved)] = resolved.stat().st_mtime
        return mtimes
