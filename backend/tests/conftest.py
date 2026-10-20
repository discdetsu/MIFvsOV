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


@pytest.fixture(scope="session")
def detector(settings: Settings) -> Detector:
    return Detector(settings.weights_path, img_size=settings.img_size, iou=settings.iou)


@pytest.fixture(scope="session")
def parity() -> dict:
    return json.loads((FIXTURES / "parity.json").read_text())
