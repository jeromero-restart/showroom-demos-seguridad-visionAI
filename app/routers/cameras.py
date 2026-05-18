from fastapi import APIRouter, HTTPException

from app.config.cameras import ALL_CAMERAS

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.get("", name="list_cameras")
async def list_cameras():
    """Return list of pre-loaded demo cameras."""
    return {
        "cameras": [
            {
                "id": cam.id,
                "name": cam.name,
                "video_url": f"/api/cameras/{cam.id}/stream",
                "description": cam.description,
                "resolution": cam.resolution,
                "fps": cam.fps,
            }
            for cam in ALL_CAMERAS
        ]
    }


@router.get("/{camera_id}", name="get_camera")
async def get_camera(camera_id: str):
    """Return single camera by ID."""
    for cam in ALL_CAMERAS:
        if cam.id == camera_id:
            return {
                "id": cam.id,
                "name": cam.name,
                "video_url": f"/api/cameras/{cam.id}/stream",
                "description": cam.description,
                "resolution": cam.resolution,
                "fps": cam.fps,
            }
    raise HTTPException(status_code=404, detail="Camera not found")
