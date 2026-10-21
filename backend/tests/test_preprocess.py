"""The NumPy letterbox and non-maximum suppression against the torch/torchvision code they replaced."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

from app.inference.detector import letterbox, letterbox_shape, per_class_non_maximum_suppression, to_rgb_array

from .conftest import IMAGES


def torch_letterbox(im: np.ndarray, new_shape: tuple[int, int], color: float = 114.0) -> np.ndarray:
    """The pre-ONNX letterbox, verbatim apart from the HWC uint8 numpy in/out."""
    t = torch.from_numpy(im.copy()).permute(2, 0, 1).float()
    h, w = t.shape[1:]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = round(w * r), round(h * r)
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if (w, h) != new_unpad:
        t = F.interpolate(t[None], size=new_unpad[::-1], mode="bilinear", align_corners=False)[0]
        t = t.round().clamp(0, 255)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return F.pad(t, (left, right, top, bottom), value=color).numpy()


def assert_letterbox_equal(im: np.ndarray, img_size: int = 640) -> None:
    shape1 = letterbox_shape(im.shape[:2], img_size, 32)
    got, ref = letterbox(im, shape1), torch_letterbox(im, shape1)
    assert got.shape == ref.shape
    # Same formula and float32 order as torch, but torch's vectorised kernel (likely FMA) rounds a few
    # blends differently, which flips an exact .5 by one level. Observed: at most 14 of 1.2 M values
    # (0.0012%, the 1016 fixture), often none. The parity tests show it doesn't matter.
    diff = np.abs(got - ref)
    assert diff.max() <= 1
    assert np.count_nonzero(diff) / diff.size < 1e-4


@pytest.mark.parametrize("fixture", sorted(p.name for p in IMAGES.iterdir()))
def test_letterbox_fixtures(fixture):
    assert_letterbox_equal(to_rgb_array(Image.open(IMAGES / fixture)))


@pytest.mark.parametrize(
    "hw",
    [
        (3024, 4032),  # 12 MP phone photo
        (4032, 3024),  # portrait
        (1800, 2400),
        (1001, 777),  # odd sizes
        (640, 640),  # no resize
        (640, 480),  # no resize, padded
        (300, 400),  # upscaling
        (37, 1000),  # extreme aspect ratio
    ],
)
def test_letterbox_random(hw):
    rng = np.random.default_rng(hw[0] * 10000 + hw[1])
    assert_letterbox_equal(rng.integers(0, 256, (*hw, 3), dtype=np.uint8))


def test_letterbox_smooth_gradients():
    # Smooth content makes exact .5 blends (where summation order could flip a rounding) more likely.
    y, x = np.mgrid[0:3024, 0:4032]
    im = np.stack([(x // 16) % 256, (y // 12) % 256, ((x + y) // 28) % 256], -1).astype(np.uint8)
    assert_letterbox_equal(im)


@pytest.mark.parametrize("seed", range(5))
def test_per_class_non_maximum_suppression_matches_torchvision(seed):
    rng = np.random.default_rng(seed)
    n = 2000
    xy = rng.uniform(0, 600, (n, 2))
    wh = rng.uniform(5, 120, (n, 2))
    boxes = np.concatenate((xy, xy + wh), 1).astype(np.float32)
    scores = rng.uniform(0, 1, n).astype(np.float32)
    classes = rng.integers(0, 2, n)

    got = per_class_non_maximum_suppression(boxes, scores, classes, 0.45)
    ref = torchvision.ops.batched_nms(*map(torch.from_numpy, (boxes, scores, classes)), 0.45)
    np.testing.assert_array_equal(got, ref.numpy())


def test_per_class_non_maximum_suppression_empty():
    kept = per_class_non_maximum_suppression(
        np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64), 0.45
    )
    assert kept.size == 0
