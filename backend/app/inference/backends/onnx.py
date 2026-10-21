import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxBackend:
    def __init__(self, path: Path, num_threads: int | None = None):
        opts = ort.SessionOptions()
        if num_threads:
            opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        # Written by scripts/export_onnx.py, so this image never needs to unpickle best.pt.
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names: list[str] = json.loads(meta["names"])
        self.stride = int(meta["stride"])
        self._input = self.session.get_inputs()[0].name

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self._input: x})[0]
