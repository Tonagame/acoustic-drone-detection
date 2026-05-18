"""
Model loading for Option 2 generalist and Option 3 specialist ensemble.
"""

from pathlib import Path

import torch
import torch.nn as nn

from config_hybrid import OPTION2_MODEL_PATH, OPTION3_MODEL_PATH, VIEW_NAMES


class DroneCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.fc(self.gap(self.features(x)).view(x.size(0), -1))


def load_option2_generalist(path: Path = OPTION2_MODEL_PATH, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not path.exists():
        raise FileNotFoundError(f"Option 2 model not found: {path}")

    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    classes = ckpt.get("classes", ["drone", "no_drone"])
    model = DroneCNN(len(classes)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    drone_idx = classes.index("drone") if "drone" in classes else int(ckpt.get("drone_idx", 0))
    return {
        "model": model,
        "drone_idx": drone_idx,
        "classes": classes,
        "path": path,
        "phase": ckpt.get("phase", "option2"),
    }


def _option3_key(bundle: dict, vi: int, view_name: str) -> str:
    candidates = [
        f'model_{vi}_{view_name.replace("-", "_").replace("+", "_")}',
        f"model_{vi}_{view_name}",
    ]
    for key in candidates:
        if key in bundle:
            return key
    matches = [k for k in bundle.keys() if k.startswith(f"model_{vi}_")]
    if matches:
        return matches[0]
    raise KeyError(f"Missing Option 3 state dict for view {vi}: {view_name}")


def load_option3_specialists(path: Path = OPTION3_MODEL_PATH, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not path.exists():
        raise FileNotFoundError(f"Option 3 specialist bundle not found: {path}")

    bundle = torch.load(str(path), map_location=device, weights_only=False)
    view_names = bundle.get("view_names", VIEW_NAMES)
    classes = bundle.get("classes", ["drone", "no_drone"])
    drone_idx = int(bundle.get("drone_idx", classes.index("drone") if "drone" in classes else 0))
    models = []
    for vi, view_name in enumerate(view_names):
        model = DroneCNN(len(classes)).to(device)
        model.load_state_dict(bundle[_option3_key(bundle, vi, view_name)])
        model.eval()
        models.append(model)
    return {
        "models": models,
        "drone_idx": drone_idx,
        "classes": classes,
        "path": path,
        "phase": bundle.get("phase", "option3"),
        "view_names": view_names,
        "view_weights": bundle.get("view_weights", None),
    }


def load_hybrid_models(option2_path=OPTION2_MODEL_PATH, option3_path=OPTION3_MODEL_PATH, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "device": device,
        "option2": load_option2_generalist(option2_path, device),
        "option3": load_option3_specialists(option3_path, device),
    }
