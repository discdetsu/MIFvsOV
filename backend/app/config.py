from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Runtime config; every field can be overridden with a `MIFVSOV_*` env var."""

    model_config = SettingsConfigDict(env_prefix="MIFVSOV_", env_ignore_empty=True)

    weights_path: Path = BACKEND_DIR / "weights" / "best.pt"
    img_size: int = 640
    # Default display threshold, used by the client slider.
    conf: float = 0.25
    # Server-side floor, kept low so the client can filter without re-requesting.
    detect_conf: float = 0.10
    iou: float = 0.45
    max_det: int = 300
    max_upload_mb: float = 5.0
    num_threads: int | None = None
    static_dir: Path = BACKEND_DIR / "static"
    cors_origins: list[str] = []

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
