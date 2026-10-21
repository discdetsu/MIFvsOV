"""Inference runtimes behind one interface. Each is imported lazily, so an image only needs the one it runs."""

import importlib
from typing import Protocol

import numpy as np

from app.config import Settings

# runtime -> (package it needs, Docker build arg / uv extra that installs it)
REQUIREMENTS = {"torch": "torch", "onnx": "onnxruntime"}


class Backend(Protocol):
    names: list[str]
    stride: int

    def forward(self, x: np.ndarray) -> np.ndarray:
        """(1, 3, H, W) float32 in 0..1 -> (1, N, 5 + nc) raw predictions: xywh, objectness, class scores."""
        ...


class RuntimeNotInstalledError(RuntimeError):
    pass


def load_backend(settings: Settings) -> Backend:
    package = REQUIREMENTS[settings.runtime]
    try:
        importlib.import_module(package)
    except ModuleNotFoundError as e:
        if e.name != package:
            raise
        raise RuntimeNotInstalledError(
            f"MIFVSOV_RUNTIME={settings.runtime} but {package} is not installed; "
            f"build with RUNTIME={settings.runtime} (or `uv sync --extra {settings.runtime}`)"
        ) from e

    if settings.runtime == "onnx":
        from .onnx import OnnxBackend

        return OnnxBackend(settings.onnx_path, num_threads=settings.num_threads)

    from .torch import TorchBackend

    return TorchBackend(settings.weights_path, num_threads=settings.num_threads)
