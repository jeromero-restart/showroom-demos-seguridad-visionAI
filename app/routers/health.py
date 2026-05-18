from datetime import datetime, timezone

import psutil
import torch
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health", tags=["health"])
async def health():
    """Liveness probe — returns 200 if server is alive."""
    return {"status": "ok"}


@router.get("/health/detailed", tags=["health"])
async def health_detailed(request: Request):
    """Diagnostics endpoint with GPU and system status."""
    model_loaded = False
    if hasattr(request.app.state, "model") and request.app.state.model is not None:
        model_loaded = True

    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "model_loaded": model_loaded,
        "disk_free_gb": psutil.disk_usage("/").free / 1e9,
        "memory_usage_percent": psutil.virtual_memory().percent,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
