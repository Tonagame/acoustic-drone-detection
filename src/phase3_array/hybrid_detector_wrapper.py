"""Detector wrapper used by Phase 3 beam scanning.

Supports the original Option2+Option3 hybrid and the newer Phase 2 harmonic
fusion guard behind the same mono-window prediction interface.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as FA


def _add_hybrid_path(config):
    if isinstance(config, (str, Path)):
        hybrid_src = Path(config)
    else:
        hybrid_src = Path(getattr(config, "hybrid_src_dir"))
    if str(hybrid_src) not in sys.path:
        sys.path.insert(0, str(hybrid_src))


def load_hybrid_detector(config):
    mode = getattr(config, "detector_mode", "hybrid_option2_option3")
    if mode == "phase2_harmonic":
        return load_phase2_harmonic_detector(config)
    if mode not in ("hybrid_option2_option3", "hybrid"):
        raise ValueError(f"Unknown Phase 3 detector_mode: {mode}")

    _add_hybrid_path(config)
    from load_models import load_hybrid_models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_hybrid_models(
        option2_path=Path(getattr(config, "option2_model_path")),
        option3_path=Path(getattr(config, "option3_model_path")),
        device=device,
    )
    return {
        "models": models,
        "device": device,
        "rule": getattr(config, "hybrid_rule", "B"),
        "option3_score_method": getattr(config, "hybrid_option3_score_method", "weighted_average"),
        "enable_veto": bool(getattr(config, "hybrid_enable_veto", True)),
        "detector_mode": "hybrid_option2_option3",
    }


def load_phase2_harmonic_detector(config):
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.phase2_harmonic_fusion.features_phase2 import (
        harmonic_vector,
        load_backbone,
        weighted_cnn_latent,
    )
    from src.phase2_harmonic_fusion.model_phase2 import HarmonicFusionHead
    from src.phase2v5_real_noise.audio_phase2v5 import AudioPreprocessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_path = Path(getattr(config, "phase2_harmonic_backbone_path"))
    phase2_path = Path(getattr(config, "phase2_harmonic_model_path"))
    if not backbone_path.exists():
        raise FileNotFoundError(f"Phase 2 backbone not found: {backbone_path}")
    if not phase2_path.exists():
        raise FileNotFoundError(f"Phase 2 harmonic model not found: {phase2_path}")

    backbone, _ = load_backbone(backbone_path, device)
    ckpt = torch.load(str(phase2_path), map_location=device, weights_only=False)
    input_dim = int(ckpt["metadata"]["input_dim"])
    head = HarmonicFusionHead(in_dim=input_dim).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    preproc = AudioPreprocessor(16000)
    return {
        "detector_mode": "phase2_harmonic",
        "device": device,
        "backbone": backbone,
        "head": head,
        "preproc": preproc,
        "threshold": float(getattr(config, "phase2_harmonic_threshold", 0.85)),
        "phase2_path": phase2_path,
        "backbone_path": backbone_path,
        "harmonic_vector": harmonic_vector,
        "weighted_cnn_latent": weighted_cnn_latent,
    }


def _resample_if_needed(x: np.ndarray, fs: int, target_fs: int) -> np.ndarray:
    if fs == target_fs:
        return x.astype(np.float32)
    t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
    y = FA.resample(t, fs, target_fs).squeeze(0).numpy()
    return y.astype(np.float32)


def predict_hybrid_on_mono_window(x, fs, hybrid_detector):
    if hybrid_detector.get("detector_mode") == "phase2_harmonic":
        return predict_phase2_harmonic_on_mono_window(x, fs, hybrid_detector)

    _add_hybrid_path(Path(__file__).resolve().parents[2] / "src" / "hybrid_option2_option3")
    from fuse_option2_option3 import fuse_predictions
    from predict_option2 import predict_option2
    from predict_option3 import predict_option3

    models = hybrid_detector["models"]
    target_fs = 16000
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = _resample_if_needed(x, int(fs), target_fs)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1e-6:
        x = x / peak

    o2 = predict_option2(models["option2"], x, hybrid_detector["device"])
    o3 = predict_option3(
        models["option3"],
        x,
        hybrid_detector["device"],
        method=hybrid_detector["option3_score_method"],
    )
    fused = fuse_predictions(
        o2.score,
        o3,
        rule=hybrid_detector["rule"],
        enable_veto=hybrid_detector["enable_veto"],
    )
    debug = {
        "option2_score": o2.score,
        "option2_per_view_probs": o2.per_view_probs,
        "option3_score": o3.score,
        "option3_filtered_max": o3.filtered_max,
        "option3_weighted_average": o3.weighted_average,
        "option3_vote_count": o3.vote_count,
        "option3_per_view_probs": o3.per_view_probs,
        "fusion_rule": hybrid_detector["rule"],
        "fusion_reason": fused.reason,
        "vetoed": fused.vetoed,
    }
    return float(0.5 * o2.score + 0.5 * o3.score), bool(fused.detected), debug


@torch.no_grad()
def predict_phase2_harmonic_on_mono_window(x, fs, detector):
    target_fs = 16000
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = _resample_if_needed(x, int(fs), target_fs)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1e-6:
        x = x / peak
    if peak < 0.002:
        return 0.0, False, {
            "phase2_score": 0.0,
            "detector_mode": "phase2_harmonic",
            "fusion_reason": "phase2:silent",
        }

    device = detector["device"]
    cnn = detector["weighted_cnn_latent"](detector["backbone"], detector["preproc"], x, device)
    harm = detector["harmonic_vector"](x)
    feat = np.concatenate([cnn, harm]).astype(np.float32)
    logits = detector["head"](torch.from_numpy(feat).unsqueeze(0).to(device))
    score = float(torch.softmax(logits, dim=1)[0, 0].item())
    detected = score > float(detector["threshold"])
    debug = {
        "detector_mode": "phase2_harmonic",
        "phase2_score": score,
        "phase2_threshold": float(detector["threshold"]),
        "option2_score": "",
        "option3_score": "",
        "option3_filtered_max": "",
        "option3_weighted_average": "",
        "option3_vote_count": "",
        "option3_per_view_probs": [],
        "vehicle_risk_score": float(harm[6]),
        "f0_norm": float(harm[0]),
        "hps_confidence": float(harm[1]),
        "harmonicity_score": float(harm[3]),
        "upper_harmonic_explained_ratio": float(harm[4]),
        "fusion_reason": f"phase2:{'detected' if detected else 'no_detect'},score={score:.3f},thr={detector['threshold']:.3f},vehicle={harm[6]:.3f}",
        "vetoed": False,
    }
    return score, bool(detected), debug
