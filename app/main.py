import logging
from contextlib import asynccontextmanager

import psutil
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import cameras, health, areas, jobs, camera_stream, live_stream
from app.db.schema import init_db
from app.processing.live_session import LiveSessionManager

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def initialize_gpu():
    """Log GPU availability at startup; always proceed regardless."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU AVAILABLE: {gpu_name} (device count: {gpu_count})")
        return {"available": True, "name": gpu_name, "count": gpu_count}
    else:
        logger.warning("GPU NOT AVAILABLE — using CPU fallback")
        return {"available": False, "name": None, "count": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    gpu_info = initialize_gpu()
    app.state.gpu_info = gpu_info

    from ultralytics import YOLO
    model_factory = lambda: YOLO("yolov8m.pt")

    manager = LiveSessionManager(model_factory=model_factory)
    app.state.live_sessions = manager
    logger.info("LiveSessionManager initialised with model factory")

    # Property alias so health endpoint and other code can still use app.state.model
    app.state.model = manager.model

    yield

    # Shutdown (cleanup resources if needed)
    logger.info("SIALAR backend shutting down")
    app.state.model = None


app = FastAPI(
    title="SIALAR Backend",
    description="PoC video analytics with YOLOv8",
    version="0.0.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router registration
app.include_router(health.router, tags=["health"])
app.include_router(cameras.router, tags=["cameras"])
app.include_router(camera_stream.router)
app.include_router(areas.router)
app.include_router(jobs.router)
app.include_router(live_stream.router)


@app.get("/")
async def root():
    return {"status": "ok", "service": "sialar-backend"}
