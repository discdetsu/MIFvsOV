import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

from .conftest import IMAGES

SAMPLE = IMAGES / "Opisthorchis viverrine_1007.jpg"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # No static dir, so only the API is mounted; 1 MB limit so the sample (~160 KB) fits.
    settings = Settings(static_dir=tmp_path_factory.mktemp("nostatic") / "missing", max_upload_mb=1)
    with TestClient(create_app(settings)) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model(client):
    r = client.get("/api/model")
    assert r.status_code == 200
    assert r.json() == {"names": ["MIF", "OV"], "img_size": 640, "conf": 0.25, "detect_conf": 0.1, "iou": 0.45}


def test_detect(client):
    r = client.post("/api/detect", files={"file": (SAMPLE.name, SAMPLE.read_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["image"] == {"width": 3840, "height": 2160}
    assert body["model"] == {"names": ["MIF", "OV"]}
    assert body["inference_ms"] > 0
    assert {d["label"] for d in body["detections"]} == {"MIF", "OV"}
    for d in body["detections"]:
        assert set(d) == {"class_id", "label", "confidence", "x1", "y1", "x2", "y2"}
        assert 0 <= d["x1"] < d["x2"] <= 3840 and 0 <= d["y1"] < d["y2"] <= 2160
        assert d["confidence"] >= 0.10


def test_detect_conf_param(client):
    r = client.post("/api/detect?conf=0.5", files={"file": (SAMPLE.name, SAMPLE.read_bytes(), "image/jpeg")})
    assert r.status_code == 200
    assert [d["label"] for d in r.json()["detections"]] == ["MIF"]


def test_detect_rejects_non_image_type(client):
    r = client.post("/api/detect", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_detect_rejects_undecodable_image(client):
    r = client.post("/api/detect", files={"file": ("fake.jpg", b"not really a jpeg", "image/jpeg")})
    assert r.status_code == 415


def test_detect_rejects_oversized(client):
    big = b"\xff\xd8" + b"\0" * (1024 * 1024 + 1)
    r = client.post("/api/detect", files={"file": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413


def test_detect_rejects_bad_conf(client):
    r = client.post("/api/detect?conf=2", files={"file": (SAMPLE.name, SAMPLE.read_bytes(), "image/jpeg")})
    assert r.status_code == 422
