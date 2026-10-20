from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import Settings, get_settings
from app.inference.detector import Detector


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.detector = Detector(
            settings.weights_path,
            img_size=settings.img_size,
            iou=settings.iou,
            max_det=settings.max_det,
            num_threads=settings.num_threads,
        )
        yield

    app = FastAPI(title="MIFvsOV", version="2.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.include_router(router)

    # Only needed in dev, when the Vite server (localhost:5173) calls the API cross-origin.
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"])

    # Production: serve the SvelteKit build. Mounted last so /api routes take precedence.
    if settings.static_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="static")

    return app


app = create_app()
