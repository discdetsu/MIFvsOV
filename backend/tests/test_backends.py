"""best.onnx must be an up-to-date, equivalent export of best.pt (see scripts/export_onnx.py)."""

import hashlib

import pytest
from PIL import Image

from app.inference.backends.onnx import OnnxBackend
from app.inference.backends.torch import TorchBackend
from app.inference.detector import preprocess, to_rgb_array
from scripts.export_onnx import MAX_BOX_DIFF, MAX_SCORE_DIFF, output_diff

from .conftest import IMAGES


@pytest.fixture(scope="module")
def torch_backend(settings) -> TorchBackend:
    return TorchBackend(settings.weights_path)


@pytest.fixture(scope="module")
def onnx_backend(settings) -> OnnxBackend:
    return OnnxBackend(settings.onnx_path)


def test_onnx_exported_from_current_weights(settings, onnx_backend):
    meta = onnx_backend.session.get_modelmeta().custom_metadata_map
    assert meta["source_sha256"] == hashlib.sha256(settings.weights_path.read_bytes()).hexdigest(), (
        "best.onnx is stale: run `uv run python -m scripts.export_onnx`"
    )


def test_onnx_metadata(torch_backend, onnx_backend):
    assert onnx_backend.names == torch_backend.names
    assert onnx_backend.stride == torch_backend.stride


@pytest.mark.parametrize("fixture", sorted(p.name for p in IMAGES.iterdir()))
def test_raw_outputs_match(torch_backend, onnx_backend, settings, fixture):
    x, _ = preprocess(to_rgb_array(Image.open(IMAGES / fixture)), settings.img_size, torch_backend.stride)
    a, b = torch_backend.forward(x), onnx_backend.forward(x)
    assert a.shape == b.shape
    box, score = output_diff(a, b)
    assert box < MAX_BOX_DIFF
    assert score < MAX_SCORE_DIFF
