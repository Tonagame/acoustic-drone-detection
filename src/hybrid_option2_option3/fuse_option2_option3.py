"""
Fusion rules and temporal smoothing for the Option2 + Option3 hybrid.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from config_hybrid import (
    ENABLE_TANK_ENGINE_VETO,
    HOP_SAMPLES,
    FS,
    HYBRID_RULE,
    OPTION3_VOTE_THRESHOLD,
    SMOOTHING_MODE,
    VETO_MAINLY_ONE_VIEW_COUNT,
    VETO_OPTION2_MAX,
)


@dataclass
class HybridDecision:
    detected: bool
    rule: str
    option2_score: float
    option3_score: float
    reason: str
    vetoed: bool


def _rule_detected(rule: str, option2_score: float, option3_score: float) -> tuple[bool, str]:
    rule = rule.upper()
    if rule == "A":
        if option3_score > 0.85:
            return True, "A:option3_high"
        if option3_score > 0.65 and option2_score > 0.40:
            return True, "A:option3_plus_option2"
        return False, "A:none"
    if rule == "B":
        if option3_score > 0.90:
            return True, "B:option3_high"
        if option3_score > 0.70 and option2_score > 0.45:
            return True, "B:option3_plus_option2"
        return False, "B:none"
    if rule == "C":
        if option3_score > 0.75 and option2_score > 0.50:
            return True, "C:confirmed"
        return False, "C:none"
    if rule == "D":
        if option3_score > 0.92:
            return True, "D:option3_extreme"
        if option3_score > 0.65 and option2_score > 0.40:
            return True, "D:confirmed_candidate"
        return False, "D:none"
    raise ValueError(f"Unknown hybrid rule: {rule}")


def apply_tank_engine_veto(
    detected: bool,
    option2_score: float,
    option3_probs: np.ndarray,
    vote_threshold: float = OPTION3_VOTE_THRESHOLD,
    option2_max: float = VETO_OPTION2_MAX,
    mainly_one_view_count: int = VETO_MAINLY_ONE_VIEW_COUNT,
) -> tuple[bool, bool, str]:
    if not detected:
        return False, False, ""
    hot_views = int((option3_probs > vote_threshold).sum())
    if option2_score < option2_max and hot_views <= mainly_one_view_count:
        return False, True, f"veto:option2={option2_score:.3f},hot_views={hot_views}"
    return True, False, ""


def fuse_predictions(
    option2_score: float,
    option3_pred,
    rule: str = HYBRID_RULE,
    enable_veto: bool = ENABLE_TANK_ENGINE_VETO,
) -> HybridDecision:
    detected, reason = _rule_detected(rule, option2_score, option3_pred.score)
    vetoed = False
    if enable_veto:
        detected, vetoed, veto_reason = apply_tank_engine_veto(
            detected,
            option2_score,
            option3_pred.per_view_probs,
        )
        if veto_reason:
            reason = f"{reason}|{veto_reason}"
    return HybridDecision(
        detected=detected,
        rule=rule,
        option2_score=float(option2_score),
        option3_score=float(option3_pred.score),
        reason=reason,
        vetoed=vetoed,
    )


class TemporalSmoother:
    def __init__(self, mode: str = SMOOTHING_MODE):
        self.mode = mode
        self.history = deque()
        self.persist_seconds = 1.5

    def reset(self):
        self.history.clear()

    def update(self, detected: bool) -> bool:
        if self.mode == "none":
            return bool(detected)
        if self.mode == "2of3":
            self.history.append(bool(detected))
            while len(self.history) > 3:
                self.history.popleft()
            return len(self.history) == 3 and sum(self.history) >= 2
        if self.mode == "3of5":
            self.history.append(bool(detected))
            while len(self.history) > 5:
                self.history.popleft()
            return len(self.history) == 5 and sum(self.history) >= 3
        if self.mode == "persist_1_5s":
            self.history.append(bool(detected))
            needed = max(1, round(self.persist_seconds / (HOP_SAMPLES / FS)))
            while len(self.history) > needed:
                self.history.popleft()
            return len(self.history) == needed and all(self.history)
        raise ValueError(f"Unknown smoothing mode: {self.mode}")
