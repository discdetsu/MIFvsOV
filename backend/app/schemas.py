from pydantic import BaseModel, Field


class Detection(BaseModel):
    class_id: int
    label: str
    confidence: float = Field(ge=0, le=1)
    x1: float
    y1: float
    x2: float
    y2: float


class ImageSize(BaseModel):
    width: int
    height: int


class ModelNames(BaseModel):
    names: list[str]


class DetectResponse(BaseModel):
    image: ImageSize
    detections: list[Detection]
    inference_ms: float
    model: ModelNames


class ModelInfo(BaseModel):
    names: list[str]
    img_size: int
    conf: float  # suggested display threshold
    detect_conf: float  # server-side floor: detections below this are never returned
    iou: float


class Health(BaseModel):
    status: str
    model_loaded: bool
