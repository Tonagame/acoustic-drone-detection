from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch

from src.phase2v5_real_noise.audio_phase2v5 import AudioPreprocessor
from src.phase2v5_real_noise.model_phase2v5 import DroneCNNV5

from .config_phase3_specialists import (
    PHASE2_BACKBONE_PATH,
    PHASE2_CONFIRM_THR,
    PHASE2_GUARD_PATH,
    PHASE2_STRONG_THR,
    SPARSE_HOT_VIEW_MAX,
    SPECIALIST_BUNDLE_PATH,
    SPECIALIST_FMAX_THR,
    SPECIALIST_SCORE_THR,
    SPECIALIST_STRONG_THR,
    SPECIALIST_VOTES_NEED,
    SPECIALIST_VOTE_THR,
    TEMPORAL_SMOOTHING,
    VEHICLE_RISK_VETO_THR,
    VIEW_NAMES,
    VIEW_WEIGHTS,
)


@dataclass
class SpecialistPrediction:
    score: float
    per_view_probs: np.ndarray
    filtered_max: float
    vote_count: int
    candidate: bool


@dataclass
class Phase2GuardPrediction:
    score: float
    vehicle_risk_score: float
    f0_norm: float
    harmonicity_score: float


@dataclass
class Phase3HybridPrediction:
    detected: bool
    score: float
    reason: str
    specialist: SpecialistPrediction
    guard: Phase2GuardPrediction


class TemporalSmoother:
    def __init__(self, mode: str = TEMPORAL_SMOOTHING):
        self.mode = mode
        self.hist = deque()

    def update(self, detected: bool) -> bool:
        if self.mode == "none":
            return bool(detected)
        n, need = (3, 2) if self.mode == "2_of_3" else (5, 3)
        self.hist.append(bool(detected))
        while len(self.hist) > n:
            self.hist.popleft()
        return len(self.hist) == n and sum(self.hist) >= need


def _phase2_imports():
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.phase2_harmonic_fusion.features_phase2 import harmonic_vector, load_backbone, weighted_cnn_latent
    from src.phase2_harmonic_fusion.model_phase2 import HarmonicFusionHead
    return harmonic_vector, load_backbone, weighted_cnn_latent, HarmonicFusionHead


def load_specialist_bundle(path: Path = SPECIALIST_BUNDLE_PATH, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(str(path), map_location=device, weights_only=False)
    view_names = bundle.get("view_names", VIEW_NAMES)
    models = []
    for vi, _name in enumerate(view_names):
        model = DroneCNNV5(n_classes=2).to(device)
        key = f"model_{vi}_{view_names[vi].replace('-', '_')}"
        if key not in bundle:
            matches = [k for k in bundle if k.startswith(f"model_{vi}_")]
            key = matches[0]
        model.load_state_dict(bundle[key])
        model.eval()
        models.append(model)
    return {"models": models, "device": device, "view_names": view_names, "drone_idx": int(bundle.get("drone_idx", 0)), "path": path}


def load_phase2_guard(phase2_path: Path = PHASE2_GUARD_PATH, backbone_path: Path = PHASE2_BACKBONE_PATH, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    harmonic_vector, load_backbone, weighted_cnn_latent, HarmonicFusionHead = _phase2_imports()
    backbone, _ = load_backbone(backbone_path, device)
    ckpt = torch.load(str(phase2_path), map_location=device, weights_only=False)
    head = HarmonicFusionHead(in_dim=int(ckpt["metadata"]["input_dim"])).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    return {
        "device": device,
        "backbone": backbone,
        "head": head,
        "harmonic_vector": harmonic_vector,
        "weighted_cnn_latent": weighted_cnn_latent,
        "phase2_path": phase2_path,
        "backbone_path": backbone_path,
    }


def load_phase3_hybrid(specialist_path: Path = SPECIALIST_BUNDLE_PATH, phase2_path: Path = PHASE2_GUARD_PATH, backbone_path: Path = PHASE2_BACKBONE_PATH, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "device": device,
        "preproc": AudioPreprocessor(16000),
        "specialists": load_specialist_bundle(specialist_path, device),
        "guard": load_phase2_guard(phase2_path, backbone_path, device),
    }


@torch.no_grad()
def predict_specialists(models_bundle, preproc: AudioPreprocessor, audio: np.ndarray) -> SpecialistPrediction:
    views = preproc.create_audio_views(audio)
    probs = np.zeros(5, dtype=np.float32)
    device = models_bundle["device"]
    drone_idx = models_bundle["drone_idx"]
    for vi, (model, view) in enumerate(zip(models_bundle["models"], views)):
        lm = preproc.audio_to_logmel(view).unsqueeze(0).unsqueeze(0).to(device)
        probs[vi] = torch.softmax(model(lm), dim=1)[0, drone_idx].item()
    score = float(VIEW_WEIGHTS @ probs)
    filtered_max = float(probs[1:].max())
    vote_count = int((probs > SPECIALIST_VOTE_THR).sum())
    candidate = (filtered_max > SPECIALIST_FMAX_THR) or (score > SPECIALIST_SCORE_THR) or (vote_count >= SPECIALIST_VOTES_NEED)
    return SpecialistPrediction(score, probs, filtered_max, vote_count, candidate)


@torch.no_grad()
def predict_phase2_guard(guard, preproc: AudioPreprocessor, audio: np.ndarray) -> Phase2GuardPrediction:
    device = guard["device"]
    cnn = guard["weighted_cnn_latent"](guard["backbone"], preproc, audio, device)
    harm = guard["harmonic_vector"](audio)
    x = torch.from_numpy(np.concatenate([cnn, harm]).astype(np.float32)).unsqueeze(0).to(device)
    score = float(torch.softmax(guard["head"](x), dim=1)[0, 0].item())
    return Phase2GuardPrediction(score, float(harm[6]), float(harm[0]), float(harm[3]))


def fuse_phase3(specialist: SpecialistPrediction, guard: Phase2GuardPrediction) -> Phase3HybridPrediction:
    hot_views = int((specialist.per_view_probs > SPECIALIST_VOTE_THR).sum())
    sparse = hot_views <= SPARSE_HOT_VIEW_MAX
    strong_specialist = specialist.filtered_max >= SPECIALIST_STRONG_THR or specialist.score >= SPECIALIST_STRONG_THR
    guard_confirms = guard.score >= PHASE2_CONFIRM_THR
    guard_strong = guard.score >= PHASE2_STRONG_THR
    vehicle_veto = guard.vehicle_risk_score >= VEHICLE_RISK_VETO_THR and sparse and not guard_strong

    detected = False
    reason = "no_candidate"
    if specialist.candidate and guard_confirms and not vehicle_veto:
        detected = True
        reason = "candidate_confirmed"
    elif strong_specialist and guard.score >= 0.50 and not vehicle_veto:
        detected = True
        reason = "strong_specialist_guard_weak_confirm"
    elif guard_strong and specialist.vote_count >= 1 and not vehicle_veto:
        detected = True
        reason = "strong_guard_one_specialist"
    elif vehicle_veto:
        reason = "vehicle_risk_veto"
    elif specialist.candidate:
        reason = "candidate_guard_reject"

    score = float(0.55 * specialist.score + 0.45 * guard.score)
    return Phase3HybridPrediction(detected, score, reason, specialist, guard)


def fuse_phase3_soft_guard(specialist: SpecialistPrediction, guard: Phase2GuardPrediction) -> Phase3HybridPrediction:
    """
    Softer guard variant for weak mixed positives.

    The original guard can reject low-SNR drones when harmonic/vehicle evidence is
    strong. This version treats vehicle risk as a score penalty when only one
    specialist view is hot, while still allowing strong specialist evidence with
    a weak Phase 2 confirmation.
    """
    hot_views = int((specialist.per_view_probs > SPECIALIST_VOTE_THR).sum())
    sparse = hot_views <= SPARSE_HOT_VIEW_MAX
    multi_view = hot_views >= SPECIALIST_VOTES_NEED
    candidate = (specialist.filtered_max > SPECIALIST_FMAX_THR) or (specialist.score > 0.50) or multi_view
    strong_specialist = specialist.filtered_max >= 0.86 or specialist.score >= 0.86
    guard_strong = guard.score >= PHASE2_STRONG_THR
    vehicle_risk = guard.vehicle_risk_score >= VEHICLE_RISK_VETO_THR and sparse and not guard_strong

    detected = False
    reason = "no_candidate"
    if candidate and guard.score >= 0.55:
        detected = True
        reason = "soft_candidate_confirmed"
    elif strong_specialist and guard.score >= 0.25:
        detected = True
        reason = "soft_strong_specialist_weak_guard"
    elif guard_strong and hot_views >= 1:
        detected = True
        reason = "soft_strong_guard_one_specialist"
    elif candidate:
        reason = "soft_candidate_guard_reject"

    score = float(0.65 * specialist.score + 0.35 * guard.score)
    if vehicle_risk and not multi_view:
        score = float(score - 0.08 * guard.vehicle_risk_score)
        if score < 0.50:
            detected = False
            reason = "soft_vehicle_penalty_reject"
        elif detected:
            reason = f"{reason}_vehicle_penalty"
    return Phase3HybridPrediction(detected, score, reason, specialist, guard)


def predict_phase3_hybrid(detector, audio: np.ndarray) -> Phase3HybridPrediction:
    preproc = detector["preproc"]
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-6:
        audio = audio / peak
    specialist = predict_specialists(detector["specialists"], preproc, audio)
    guard = predict_phase2_guard(detector["guard"], preproc, audio)
    return fuse_phase3(specialist, guard)
