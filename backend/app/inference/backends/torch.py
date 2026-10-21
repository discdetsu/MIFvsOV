import sys
from pathlib import Path

import numpy as np
import torch

from .. import models
from ..models import common, yolo


def load_model(path: Path) -> yolo.Model:
    # The checkpoint pickles classes as `models.common.*` / `models.yolo.*`.
    sys.modules.setdefault("models", models)
    sys.modules.setdefault("models.common", common)
    sys.modules.setdefault("models.yolo", yolo)
    # weights_only=False unpickles arbitrary objects: only ever load the trusted, bundled best.pt.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float().fuse().eval()

    # Compatibility updates, as in yolov5's attempt_load()
    for m in model.modules():
        t = type(m)
        if t in (torch.nn.SiLU, yolo.Detect, yolo.Model):
            m.inplace = True
            if t is yolo.Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                m.anchor_grid = [torch.zeros(1)] * m.nl
        elif t is torch.nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None
    return model


class TorchBackend:
    def __init__(self, weights: Path, num_threads: int | None = None):
        if num_threads:
            torch.set_num_threads(num_threads)
        self.model = load_model(weights)
        self.names: list[str] = list(self.model.names)  # type: ignore[arg-type]
        self.stride = int(self.model.stride.max())  # type: ignore[operator]

    def forward(self, x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return self.model(torch.from_numpy(x))[0].numpy()
