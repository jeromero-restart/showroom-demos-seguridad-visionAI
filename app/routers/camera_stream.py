"""
GET /api/cameras/{camera_id}/stream — serves demo video file.

Per D-02: stub that redirects to static demo video from /app/demo_cameras/.
Router prefix is /api/cameras so requests route through nginx /api/ block
(which has proxy_buffering off — needed for video file streaming).
Replace this stub body with RTSP stream logic in a future phase.
"""
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config.cameras import DEMO_CAMERAS

router = APIRouter(prefix="/api/cameras", tags=["camera-stream"])


@router.get("/{camera_id}/stream", name="stream_camera_video")
async def stream_camera_video(camera_id: str):
    """Serve demo MP4 for the given camera_id. Returns video/mp4."""
    for cam in DEMO_CAMERAS:
        if cam.id == camera_id:
            if os.path.exists(cam.video_path):
                return FileResponse(
                    cam.video_path,
                    media_type="video/mp4",
                    headers={"Accept-Ranges": "bytes"},
                )
            raise HTTPException(status_code=503, detail="Video file not available")
    raise HTTPException(status_code=404, detail="Camera not found")
