import json
from pathlib import Path

import pytest

from app.config import Settings
from app.inference.detector import Detector

FIXTURES = Path(__file__).parent / "fixtures"
IMAGES = FIXTURES / "images"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


RUNTIMES = ["torch", "onnx"]


@pytest.fixture(scope="session", params=RUNTIMES)
def detector(request, settings: Settings) -> Detector:
    return Detector.from_settings(settings.model_copy(update={"runtime": request.param}))


@pytest.fixture(scope="session")
def parity() -> dict:
    return json.loads((FIXTURES / "parity.json").read_text())
