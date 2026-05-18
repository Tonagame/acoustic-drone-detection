"""Guard-only fusion on top of the existing Option2+Option3 hybrid."""

from dataclasses import dataclass

import numpy as np

from .config_harmonic_guard import (
    HOT_VIEW_THRESHOLD,
    HOT_VIEWS_SPARSE_MAX,
    OPTION2_STRONG_MIN,
    OPTION2_WEAK_MAX,
    OPTION3_MEDIUM_MAX,
    VEHICLE_RISK_STRONG_THRESHOLD,
    VEHICLE_RISK_THRESHOLD,
    VOTE_COUNT_CONFIRM_MIN,
)


@dataclass
class GuardDecision:
    detected: bool
    original_detected: bool
    downgraded: bool
    reason: str
    guard_score: float


def apply_harmonic_guard(hybrid_decision, option3_pred, harmonic_features) -> GuardDecision:
    """Downgrade suspicious vehicle-like detections without touching audio."""
    original = bool(hybrid_decision.detected)
    if not original:
        return GuardDecision(False, False, False, "guard:not_needed", harmonic_features.vehicle_risk_score)

    probs = np.asarray(option3_pred.per_view_probs, dtype=np.float32)
    hot_views = int((probs > HOT_VIEW_THRESHOLD).sum())
    sparse_views = hot_views <= HOT_VIEWS_SPARSE_MAX
    option2_weak = float(hybrid_decision.option2_score) < OPTION2_WEAK_MAX
    option2_strong = float(hybrid_decision.option2_score) >= OPTION2_STRONG_MIN
    option3_medium = float(option3_pred.score) < OPTION3_MEDIUM_MAX
    multi_view_confirmed = int(option3_pred.vote_count) >= VOTE_COUNT_CONFIRM_MIN
    vehicle_risk = float(harmonic_features.vehicle_risk_score)

    strong_drone_evidence = option2_strong and multi_view_confirmed
    if strong_drone_evidence:
        return GuardDecision(
            True,
            True,
            False,
            f"guard:keep_strong_drone,o2={hybrid_decision.option2_score:.3f},votes={option3_pred.vote_count},vehicle={vehicle_risk:.3f}",
            vehicle_risk,
        )

    if vehicle_risk >= VEHICLE_RISK_STRONG_THRESHOLD and (option2_weak or sparse_views):
        return GuardDecision(
            False,
            True,
            True,
            f"guard:downgrade_strong_vehicle,vehicle={vehicle_risk:.3f},o2={hybrid_decision.option2_score:.3f},hot_views={hot_views}",
            vehicle_risk,
        )

    if vehicle_risk >= VEHICLE_RISK_THRESHOLD and option3_medium and (option2_weak or sparse_views):
        return GuardDecision(
            False,
            True,
            True,
            f"guard:downgrade_vehicle_medium_drone,vehicle={vehicle_risk:.3f},o2={hybrid_decision.option2_score:.3f},o3={option3_pred.score:.3f},hot_views={hot_views}",
            vehicle_risk,
        )

    return GuardDecision(
        True,
        True,
        False,
        f"guard:keep,vehicle={vehicle_risk:.3f},o2={hybrid_decision.option2_score:.3f},o3={option3_pred.score:.3f},hot_views={hot_views}",
        vehicle_risk,
    )
