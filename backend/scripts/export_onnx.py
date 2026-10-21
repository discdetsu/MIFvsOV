"""Export weights/best.pt to weights/best.onnx and check that both runtimes agree.

uv run python -m scripts.export_onnx          # export + verify on the test fixtures
uv run python -m scripts.export_onnx --check  # only check that best.onnx was exported from today's best.pt
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from app.config import BACKEND_DIR, Settings
from app.inference.backends.onnx import OnnxBackend
from app.inference.backends.torch import TorchBackend, load_model
from app.inference.detector import preprocess, to_rgb_array
from app.inference.models import yolo

FIXTURES = BACKEND_DIR / "tests" / "fixtures" / "images"
# Raw outputs are xywh in input pixels (up to img_size), then objectness and class scores in 0..1,
# so boxes get their own tolerance: float32 noise on a 640 px coordinate is already ~1e-3.
MAX_SCORE_DIFF = 1e-3
MAX_BOX_DIFF = 0.01  # px


def output_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Max |a - b| over (box columns, score columns) of two raw (1, N, 5 + nc) outputs."""
    d = np.abs(a - b)
    return float(d[..., :4].max()), float(d[..., 4:].max())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(weights: Path, out: Path, img_size: int) -> None:
    model = load_model(weights)
    for m in model.modules():
        if isinstance(m, yolo.Detect):
            m.onnx_dynamic = True  # rebuild the anchor grid from the input shape instead of baking it in
            m.export = True  # return only the concatenated predictions
            # Slice assignment traces poorly; the out-of-place path is numerically identical.
            m.inplace = False  # type: ignore[assignment]

    torch.onnx.export(
        model,
        (torch.zeros(1, 3, img_size, img_size),),
        str(out),
        dynamo=False,
        opset_version=17,
        input_names=["images"],
        output_names=["output"],
        # Letterboxing produces rectangular inputs, e.g. 384x640.
        dynamic_axes={"images": {2: "height", 3: "width"}, "output": {1: "anchors"}},
    )

    proto = onnx.load(str(out))
    meta = {
        "names": json.dumps(list(model.names)),  # type: ignore[arg-type]
        "stride": str(int(model.stride.max())),  # type: ignore[operator]
        "source_sha256": sha256(weights),
        "exported_with": f"torch {torch.__version__}",
    }
    for key, value in meta.items():
        proto.metadata_props.add(key=key, value=value)
    onnx.checker.check_model(proto)
    onnx.save(proto, str(out))


def verify(weights: Path, onnx_path: Path, img_size: int) -> bool:
    """Compare torch and onnxruntime raw outputs on every fixture image."""
    torch_backend, onnx_backend = TorchBackend(weights), OnnxBackend(onnx_path)
    ok = onnx_backend.names == torch_backend.names and onnx_backend.stride == torch_backend.stride
    for path in sorted(FIXTURES.iterdir()):
        x, shape1 = preprocess(to_rgb_array(Image.open(path)), img_size, torch_backend.stride)
        box, score = output_diff(torch_backend.forward(x), onnx_backend.forward(x))
        print(f"  {path.name}: input {shape1[0]}x{shape1[1]}, max |Δ| boxes {box:.2e} px, scores {score:.2e}")
        ok &= box < MAX_BOX_DIFF and score < MAX_SCORE_DIFF
    return ok


def check(weights: Path, onnx_path: Path) -> bool:
    meta = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).get_modelmeta()
    exported_from = meta.custom_metadata_map.get("source_sha256")
    if exported_from != sha256(weights):
        print(f"{onnx_path.name} is stale: exported from {exported_from}, {weights.name} is {sha256(weights)}")
        return False
    print(f"{onnx_path.name} matches {weights.name} ({exported_from})")
    return True


def main() -> int:
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only check source_sha256, don't export")
    args = parser.parse_args()
    if args.check:
        return 0 if check(settings.weights_path, settings.onnx_path) else 1

    export(settings.weights_path, settings.onnx_path, settings.img_size)
    print(f"wrote {settings.onnx_path} ({settings.onnx_path.stat().st_size / 1e6:.1f} MB), verifying:")
    if not verify(settings.weights_path, settings.onnx_path, settings.img_size):
        print(f"FAIL: need names/stride equal, boxes < {MAX_BOX_DIFF} px and scores < {MAX_SCORE_DIFF}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
