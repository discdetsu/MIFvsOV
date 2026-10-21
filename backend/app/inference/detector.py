"""YOLOv5 inference without AutoShape: letterbox -> forward -> non-maximum suppression -> rescale.

Pre- and post-processing mirror yolov5-master's AutoShape (the 2022 webapp)
so detections match the old stack; see tests/test_detector.py. Everything here
is NumPy, so it is shared by both runtimes (see backends/).
"""

import math
import threading
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from app.config import Settings
from app.schemas import Detection

from .backends import Backend, load_backend

MAX_SUPPRESSION_CANDIDATES = 30000  # maximum number of boxes into non-maximum suppression


def to_rgb_array(img: Image.Image) -> np.ndarray:
    """EXIF-orient and convert to an HWC uint8 RGB array, avoiding Pillow copies where possible."""
    ImageOps.exif_transpose(img, in_place=True)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img)


def letterbox_shape(shape0: tuple[int, int], img_size: int, stride: int) -> tuple[int, int]:
    """Model input (h, w) for an image of `shape0`: longest side `img_size`, both sides a multiple of stride."""
    g = img_size / max(shape0)
    return math.ceil(shape0[0] * g / stride) * stride, math.ceil(shape0[1] * g / stride) * stride


def _linear_indices(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Source indices and weights for 1-D linear resizing, as torch's align_corners=False (half-pixel, clamped)."""
    scale = np.float32(n_in / n_out)
    src = np.maximum((np.arange(n_out, dtype=np.float32) + np.float32(0.5)) * scale - np.float32(0.5), 0)
    i0 = src.astype(np.int64)  # floor, since src >= 0
    i1 = np.minimum(i0 + 1, n_in - 1)
    w1 = src - i0.astype(np.float32)
    return i0, i1, np.float32(1) - w1, w1


def letterbox(im: np.ndarray, new_shape: tuple[int, int], color: float = 114.0) -> np.ndarray:
    """Resize an HWC uint8 image to fit `new_shape` (h, w) and pad the rest. Returns CHW float32 in 0..255.

    Port of yolov5's letterbox(auto=False). Separable bilinear without antialiasing, rounded back to
    uint8 values to match cv2.INTER_LINEAR. Only the source rows that are actually sampled are converted
    to float, so a 12 MP photo needs ~25 MB of scratch instead of a full float copy.
    """
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = (new_shape[1] - nw) / 2, (new_shape[0] - nh) / 2
    top, left = round(dh - 0.1), round(dw - 0.1)

    out = np.full((3, *new_shape), color, dtype=np.float32)
    if (w, h) != (nw, nh):
        y0, y1, wy0, wy1 = _linear_indices(h, nh)
        x0, x1, wx0, wx1 = _linear_indices(w, nw)
        # Blend columns first, then rows: the same order (and float32 rounding) as torch's CPU kernel.
        rows, inv = np.unique(np.concatenate((y0, y1)), return_inverse=True)
        src = im[rows]
        cols = src[:, x0] * wx0[:, None] + src[:, x1] * wx1[:, None]  # (len(rows), nw, 3) float32
        del src
        r0, r1 = inv[:nh], inv[nh:]
        resized = cols[r0] * wy0[:, None, None] + cols[r1] * wy1[:, None, None]
        del cols
        np.clip(np.round(resized, out=resized), 0, 255, out=resized)
    else:
        resized = im
    out[:, top : top + nh, left : left + nw] = resized.transpose(2, 0, 1)
    return out


def preprocess(im: np.ndarray, img_size: int, stride: int) -> tuple[np.ndarray, tuple[int, int]]:
    """HWC uint8 RGB -> model input (1, 3, H, W) float32 in 0..1, and its (H, W)."""
    shape1 = letterbox_shape(im.shape[:2], img_size, stride)
    x = letterbox(im, shape1)
    x /= 255
    return x[None], shape1


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def greedy_non_maximum_suppression(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Greedy non-maximum suppression over xyxy boxes, as torchvision.ops.nms.

    Returns kept indices by descending score.
    """
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        iw = np.maximum(0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        ih = np.maximum(0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = iw * ih
        order = rest[inter / (areas[i] + areas[rest] - inter) <= iou_thres]
    return np.array(keep, dtype=np.int64)


def per_class_non_maximum_suppression(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thres: float
) -> np.ndarray:
    """Per-class non-maximum suppression via torchvision's coordinate trick.

    Offsets each class so boxes never overlap across classes.
    """
    if not boxes.shape[0]:
        return np.zeros(0, dtype=np.int64)
    offsets = classes.astype(boxes.dtype) * (boxes.max() + 1)
    return greedy_non_maximum_suppression(boxes + offsets[:, None], scores, iou_thres)


def non_maximum_suppression(pred: np.ndarray, conf_thres: float, iou_thres: float, max_det: int) -> np.ndarray:
    """Single-image, best-class-only port of yolov5's non-maximum suppression. Returns (n, 6): xyxy, conf, cls."""
    x = pred[pred[:, 4] > conf_thres]  # objectness candidates
    if not x.shape[0]:
        return np.zeros((0, 6), dtype=np.float32)
    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
    box = xywh2xyxy(x[:, :4])
    j = x[:, 5:].argmax(1)
    conf = x[:, 5:].max(1)
    x = np.concatenate((box, conf[:, None], j[:, None].astype(np.float32)), 1)[conf > conf_thres]
    if x.shape[0] > MAX_SUPPRESSION_CANDIDATES:
        x = x[np.argsort(-x[:, 4], kind="stable")[:MAX_SUPPRESSION_CANDIDATES]]
    i = per_class_non_maximum_suppression(x[:, :4], x[:, 4], x[:, 5].astype(np.int64), iou_thres)
    return x[i[:max_det]]


def scale_boxes(img1_shape: tuple[int, int], boxes: np.ndarray, img0_shape: tuple[int, int]) -> np.ndarray:
    """Rescale xyxy boxes from the letterboxed shape back to the original (h, w), then clip."""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img0_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img0_shape[0])
    return boxes


@dataclass
class DetectResult:
    detections: list[Detection]
    width: int
    height: int
    inference_ms: float


class Detector:
    def __init__(self, backend: Backend, img_size: int = 640, iou: float = 0.45, max_det: int = 300):
        self.backend = backend
        self.names = backend.names
        self.stride = backend.stride
        self.img_size = img_size
        self.iou = iou
        self.max_det = max_det
        # One request at a time: concurrent requests would multiply peak memory, for no CPU gain.
        self._lock = threading.Lock()
        self.backend.forward(np.zeros((1, 3, img_size, img_size), dtype=np.float32))  # warmup

    @classmethod
    def from_settings(cls, settings: Settings) -> Detector:
        return cls(load_backend(settings), img_size=settings.img_size, iou=settings.iou, max_det=settings.max_det)

    def detect(self, img: Image.Image, conf: float) -> DetectResult:
        """Run detection; boxes are in pixels of the EXIF-oriented original image.

        Pass the image without keeping another reference to it, so its pixels are freed before inference.
        """
        with self._lock:
            t0 = time.perf_counter()
            im = to_rgb_array(img)
            del img
            shape0 = im.shape[0], im.shape[1]
            x, shape1 = preprocess(im, self.img_size, self.stride)
            del im
            pred = self.backend.forward(x)[0]
            del x
            det = non_maximum_suppression(pred, conf, self.iou, self.max_det)
            det[:, :4] = scale_boxes(shape1, det[:, :4], shape0)
            inference_ms = (time.perf_counter() - t0) * 1000

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
        return DetectResult(detections, shape0[1], shape0[0], inference_ms)
