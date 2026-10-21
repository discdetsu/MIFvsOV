import sys

import pytest

from app.config import Settings
from app.inference.backends import RuntimeNotInstalledError, load_backend


@pytest.mark.parametrize(("runtime", "package"), [("onnx", "onnxruntime"), ("torch", "torch")])
def test_missing_runtime_fails_clearly(monkeypatch, runtime, package):
    monkeypatch.setitem(sys.modules, package, None)  # makes `import <package>` raise ModuleNotFoundError
    with pytest.raises(RuntimeNotInstalledError, match=f"MIFVSOV_RUNTIME={runtime} but {package} is not installed"):
        load_backend(Settings(runtime=runtime))


def test_runtime_from_env(monkeypatch):
    monkeypatch.setenv("MIFVSOV_RUNTIME", "onnx")
    assert Settings().runtime == "onnx"


def test_unknown_runtime_rejected(monkeypatch):
    monkeypatch.setenv("MIFVSOV_RUNTIME", "tensorflow")
    with pytest.raises(ValueError):
        Settings()
