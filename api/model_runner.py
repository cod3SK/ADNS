from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class NfstreamDetectionEngine:
    """Scores flows that carry NFStream contract features in flow.extra.

    Reads the 21 FEATURE_COLUMNS values stored by _NfstreamCaptureAgent and calls
    NfstreamScorer.score_matrix().  validate_matrix() is called before every predict()
    inside NfstreamScorer — no silent reconciliation.

    Returns (0.0, 'normal') for any flow whose extra does not carry the
    '_extractor': 'nfstream' marker (pre-Phase-3 flows, or extraction failures).
    """

    def __init__(self) -> None:
        self._scorer = None
        self._load()

    def _load(self) -> None:
        try:
            from serving_nfstream import NfstreamScorer
            self._scorer = NfstreamScorer()
            logger.info("NfstreamDetectionEngine: model loaded")
        except FileNotFoundError:
            logger.info("NfstreamDetectionEngine: model artifact absent, scoring disabled")
        except Exception as exc:
            logger.warning("NfstreamDetectionEngine: failed to load: %s", exc)

    def score_many(self, flows: Sequence) -> list[Tuple[float, str]]:
        """Score a batch of DB Flow objects using contract features from flow.extra."""
        if not flows:
            return []
        if self._scorer is None:
            return [(0.0, "normal")] * len(flows)

        from serving_nfstream import extra_to_feature_vector

        batch_X: list = []
        batch_idx: list[int] = []
        results: dict[int, Tuple[float, str]] = {}

        for i, flow in enumerate(flows):
            extra = getattr(flow, "extra", None) or {}
            vec = extra_to_feature_vector(extra)
            if vec is None:
                results[i] = (0.0, "normal")
            else:
                batch_X.append(vec)
                batch_idx.append(i)

        if batch_X:
            X = np.vstack(batch_X)
            try:
                scored = self._scorer.score_matrix(X)
            except Exception as exc:
                logger.warning("NfstreamDetectionEngine.score_matrix failed: %s", exc)
                scored = [(0.0, "normal")] * len(batch_X)
            for idx, score_label in zip(batch_idx, scored):
                results[idx] = score_label

        return [results[i] for i in range(len(flows))]
