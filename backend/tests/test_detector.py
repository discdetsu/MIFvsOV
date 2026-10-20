"""Parity against the 2022 stack (torch 1.11 + yolov5 AutoShape).

fixtures/parity.json was produced by running the old `torch.hub.load('yolov5-master', 'custom', ...)`
pipeline at size=640 on fixtures/images (conf 0.25 and 0.10).

The JPEG decoder changed between Pillow 8.4 and today's libjpeg-turbo (pixel diffs up to ~34 levels on
batch.jpg), which alone moves confidences by a few hundredths. To test the *port* strictly, batch.jpg is
also stored as `batch_decoded-pillow8.png`, the exact pixels the old stack saw.
"""

import hashlib
import io

import pytest
from PIL import Image

from app.schemas import Detection

from .conftest import IMAGES

BEST_PT_SHA256 = "3df91454b83f816f119d71b59ab73f4681c806f4f0b40b408cd66e8002f5cc9f"

# fixture file fed to the new detector -> key in parity.json
STRICT_CASES = {
    "Opisthorchis viverrine_1007.jpg": "Opisthorchis viverrine_1007.jpg",
    "batch_decoded-pillow8.png": "batch.jpg",
    "Opisthorchis viverrine_1016-modified.png": "Opisthorchis viverrine_1016-modified.png",
}


def iou(a: dict, b: Detection) -> float:
    ix = max(0.0, min(a["xmax"], b.x2) - max(a["xmin"], b.x1))
    iy = max(0.0, min(a["ymax"], b.y2) - max(a["ymin"], b.y1))
    inter = ix * iy
    area_a = (a["xmax"] - a["xmin"]) * (a["ymax"] - a["ymin"])
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / (area_a + area_b - inter)


def assert_matches(ref: list[dict], got: list[Detection], min_iou: float, conf_tol: float) -> None:
    assert sorted(r["class"] for r in ref) == sorted(d.class_id for d in got)
    unmatched = list(got)
    for r in ref:
        same_class = [d for d in unmatched if d.class_id == r["class"]]
        best = max(same_class, key=lambda d: iou(r, d))
        assert iou(r, best) >= min_iou, (r, best)
        assert best.confidence == pytest.approx(r["confidence"], abs=conf_tol), (r, best)
        unmatched.remove(best)


def test_weights_unchanged(settings, parity):
    digest = hashlib.sha256(settings.weights_path.read_bytes()).hexdigest()
    assert digest == BEST_PT_SHA256 == parity["best_pt_sha256"]


def test_names(detector, parity):
    assert detector.names == parity["names"] == ["MIF", "OV"]


@pytest.mark.parametrize("conf", ["0.25", "0.1"])
@pytest.mark.parametrize("fixture", STRICT_CASES)
def test_parity(detector, parity, fixture, conf):
    ref = parity["images"][STRICT_CASES[fixture]][conf]
    result = detector.detect(Image.open(IMAGES / fixture), float(conf))
    assert_matches(ref, result.detections, min_iou=0.99, conf_tol=0.01)


def test_parity_modern_jpeg_decoder(detector, parity):
    ref = parity["images"]["batch.jpg"]["0.25"]
    result = detector.detect(Image.open(IMAGES / "batch.jpg"), 0.25)
    assert_matches(ref, result.detections, min_iou=0.95, conf_tol=0.05)


def test_result_size_is_exif_oriented(detector):
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90° CW on display
    buf = io.BytesIO()
    Image.new("RGB", (400, 300)).save(buf, "JPEG", exif=exif)
    result = detector.detect(Image.open(buf), 0.25)
    assert (result.width, result.height) == (300, 400)
