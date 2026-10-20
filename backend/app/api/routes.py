from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from app.config import Settings
from app.inference.detector import Detector
from app.schemas import DetectResponse, Health, ImageSize, ModelInfo, ModelNames

router = APIRouter(prefix="/api")

ALLOWED_TYPES = {"image/jpeg", "image/png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}


def _detector(request: Request) -> Detector:
    return request.app.state.detector


def _settings(request: Request) -> Settings:
    return request.app.state.settings


@router.get("/health")
def health(request: Request) -> Health:
    return Health(status="ok", model_loaded=getattr(request.app.state, "detector", None) is not None)


@router.get("/model")
def model_info(request: Request) -> ModelInfo:
    settings = _settings(request)
    return ModelInfo(
        names=_detector(request).names,
        img_size=settings.img_size,
        conf=settings.conf,
        detect_conf=settings.detect_conf,
        iou=settings.iou,
    )


# Sync endpoint: FastAPI runs it in its threadpool, so inference doesn't block the event loop.
@router.post("/detect", responses={413: {"description": "File too large"}, 415: {"description": "Not a JPEG/PNG"}})
def detect(
    request: Request,
    file: Annotated[UploadFile, File()],
    conf: Annotated[float | None, Query(ge=0, le=1)] = None,
) -> DetectResponse:
    settings = _settings(request)
    detector = _detector(request)

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Only JPEG and PNG images are supported")
    if file.size is not None and file.size > settings.max_upload_bytes:
        raise HTTPException(413, f"Image must be at most {settings.max_upload_mb:g} MB")

    try:
        img = Image.open(file.file)
        if img.format not in ALLOWED_FORMATS:
            raise HTTPException(415, "Only JPEG and PNG images are supported")
        img.load()
    except Image.DecompressionBombError as e:
        raise HTTPException(413, "Image dimensions are too large") from e
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(415, "Could not decode image") from e

    result = detector.detect(img, settings.detect_conf if conf is None else conf)
    return DetectResponse(
        image=ImageSize(width=result.width, height=result.height),
        detections=result.detections,
        inference_ms=round(result.inference_ms, 1),
        model=ModelNames(names=detector.names),
    )
