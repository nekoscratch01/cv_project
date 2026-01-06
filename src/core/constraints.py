"""Constraint intent scoring based on Atomic Facts."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

from core.features import TrackFeatures

VALID_INTENTS = {
    "RUNNING",
    "WALKING",
    "STOPPED",
    "APPROACHING",
    "LEAVING",
    "WANDERING",
    "MOVING_LEFT",
    "MOVING_RIGHT",
}


def score_constraints(
    intents: Iterable[str] | None, features: TrackFeatures | None
) -> Tuple[float, Dict[str, float]]:
    if not intents or features is None:
        return 0.0, {}

    scores: Dict[str, float] = {}
    for intent in intents:
        intent = intent.upper().strip()
        if intent not in VALID_INTENTS:
            continue
        scores[intent] = _score_intent(intent, features)

    if not scores:
        return 0.0, {}

    return min(scores.values()), scores


def _score_intent(intent: str, features: TrackFeatures) -> float:
    ns = features.norm_speed
    sc = features.scale_change
    lin = features.linearity
    dx = features.displacement_vec[0] if features.displacement_vec else 0.0

    if intent == "RUNNING":
        return _sigmoid((ns - 1.8) / 0.2)
    if intent == "WALKING":
        return _sigmoid((ns - 0.1) / 0.2) * _sigmoid((1.8 - ns) / 0.2)
    if intent == "STOPPED":
        return _sigmoid((0.4 - ns) / 0.1)
    if intent == "APPROACHING":
        return _sigmoid((sc - 1.2) / 0.1)
    if intent == "LEAVING":
        return _sigmoid((0.8 - sc) / 0.1)
    if intent == "WANDERING":
        return _sigmoid((0.3 - lin) / 0.1)
    if intent == "MOVING_LEFT":
        return _sigmoid(((-dx) - 0.05) / 0.02)
    if intent == "MOVING_RIGHT":
        return _sigmoid((dx - 0.05) / 0.02)
    return 0.0


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
