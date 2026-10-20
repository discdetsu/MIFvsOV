"""YOLOv5 inference without AutoShape: letterbox -> forward -> NMS -> rescale.

Pre- and post-processing mirror yolov5-master's AutoShape (the 2022 webapp)
so detections match the old stack; see tests/test_detector.py.
"""

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageOps

from app.schemas import Detection

from . import models
from .models import common, yolo

MAX_WH = 7680  # (pixels) class offset for batched NMS, as in yolov5
MAX_NMS = 30000  # maximum number of boxes into NMS


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


def letterbox(im: torch.Tensor, new_shape: tuple[int, int], color: float = 114.0) -> torch.Tensor:
    """Resize a CHW uint8-valued float image to fit `new_shape` (h, w) and pad the rest.

    Port of yolov5's letterbox(auto=False). Uses bilinear without antialiasing and
    rounds back to uint8 values to match cv2.INTER_LINEAR.
    """
    h, w = im.shape[1:]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = round(w * r), round(h * r)
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

    if (w, h) != new_unpad:
        im = F.interpolate(im[None], size=new_unpad[::-1], mode="bilinear", align_corners=False)[0]
        im = im.round().clamp(0, 255)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return F.pad(im, (left, right, top, bottom), value=color)


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def non_max_suppression(pred: torch.Tensor, conf_thres: float, iou_thres: float, max_det: int) -> torch.Tensor:
    """Single-image, best-class-only port of yolov5's NMS. Returns (n, 6): xyxy, conf, cls."""
    x = pred[pred[:, 4] > conf_thres]  # objectness candidates
    if not x.shape[0]:
        return torch.zeros((0, 6))
    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
    box = xywh2xyxy(x[:, :4])
    conf, j = x[:, 5:].max(1, keepdim=True)
    x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
    if x.shape[0] > MAX_NMS:
        x = x[x[:, 4].argsort(descending=True)[:MAX_NMS]]
    i = torchvision.ops.batched_nms(x[:, :4], x[:, 4], x[:, 5].long(), iou_thres)
    return x[i[:max_det]]


def scale_boxes(img1_shape: tuple[int, int], boxes: torch.Tensor, img0_shape: tuple[int, int]) -> torch.Tensor:
    """Rescale xyxy boxes from the letterboxed shape back to the original (h, w), then clip."""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, img0_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, img0_shape[0])
    return boxes


@dataclass
class DetectResult:
    detections: list[Detection]
    width: int
    height: int
    inference_ms: float


class Detector:
    def __init__(
        self,
        weights: Path,
        img_size: int = 640,
        iou: float = 0.45,
        max_det: int = 300,
        num_threads: int | None = None,
    ):
        if num_threads:
            torch.set_num_threads(num_threads)
        self.model = load_model(weights)
        self.names: list[str] = list(self.model.names)  # type: ignore[arg-type]
        self.stride = int(self.model.stride.max())  # type: ignore[operator]
        self.img_size = img_size
        self.iou = iou
        self.max_det = max_det
        self._warmup()

    def _warmup(self) -> None:
        with torch.inference_mode():
            self.model(torch.zeros(1, 3, self.img_size, self.img_size))

    def detect(self, img: Image.Image, conf: float) -> DetectResult:
        """Run detection; boxes are in pixels of the EXIF-oriented original image."""
        t0 = time.perf_counter()
        img = ImageOps.exif_transpose(img).convert("RGB")
        im = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float()  # HWC -> CHW
        shape0 = img.height, img.width
        g = self.img_size / max(shape0)
        shape1 = (
            math.ceil(shape0[0] * g / self.stride) * self.stride,
            math.ceil(shape0[1] * g / self.stride) * self.stride,
        )

        with torch.inference_mode():
            x = (letterbox(im, shape1) / 255)[None]
            pred = self.model(x)[0][0]
            det = non_max_suppression(pred, conf, self.iou, self.max_det)
            det[:, :4] = scale_boxes(shape1, det[:, :4], shape0)

        detections = [
            Detection(
                class_id=int(c),
                label=self.names[int(c)],
                confidence=float(p),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
            )
            for x1, y1, x2, y2, p, c in det.tolist()
        ]
        return DetectResult(detections, img.width, img.height, (time.perf_counter() - t0) * 1000)
