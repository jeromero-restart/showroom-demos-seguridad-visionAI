from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import Literal


# ---------------------------------------------------------------------------
# Trigger config with param validation (reviews fix: MEDIUM trigger params)
# ---------------------------------------------------------------------------

class TriggerConfig(BaseModel):
    type: Literal["count", "dwell", "direction"]
    params: dict  # validated below per type

    @model_validator(mode="after")
    def validate_params(self) -> "TriggerConfig":
        t = self.type
        p = self.params
        if t == "count":
            n = p.get("min_count")
            if n is None or not isinstance(n, int) or n <= 0:
                raise ValueError("COUNT trigger requires params.min_count (int > 0)")
        elif t == "dwell":
            s = p.get("threshold_s")
            if s is None or not isinstance(s, (int, float)) or float(s) <= 0:
                raise ValueError("DWELL trigger requires params.threshold_s (float > 0)")
        elif t == "direction":
            d = p.get("direction")
            if d not in ("N", "S", "E", "W"):
                raise ValueError("DIRECTION trigger requires params.direction one of: N, S, E, W")
        return self


# ---------------------------------------------------------------------------
# Area
# ---------------------------------------------------------------------------

class AreaCreate(BaseModel):
    camera_id: str
    polygon: list[list[float]] = Field(
        ...,
        description="List of [x, y] pairs in normalized [0.0-1.0] coordinates",
        min_length=3,
    )
    entity_type: Literal["person", "vehicle", "animal"]
    trigger: TriggerConfig


class AreaResponse(BaseModel):
    area_id: str


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

class JobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "done", "error"]


class JobProgress(BaseModel):
    progress_pct: int
    frames_done: int
    total_frames: int
    status: Literal["processing", "done", "error"]
    job_id: str | None = None  # present on final done event


# ---------------------------------------------------------------------------
# detections.json (returned by GET /api/jobs/{id}/results)
# ---------------------------------------------------------------------------

class DetectionObject(BaseModel):
    track_id: int
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] normalized
    in_zone: bool


class EventObject(BaseModel):
    event_id: str
    frame_number: int
    timestamp_s: float
    track_id: int
    class_name: str
    trigger_type: str
    trigger_params: dict


class VideoMetadata(BaseModel):
    fps: float
    processed_fps: int
    total_frames: int
    video_duration: float
    area_id: str


class DetectionResult(BaseModel):
    metadata: VideoMetadata
    frames: dict[str, list[DetectionObject]]  # string frame number -> detections
    events: list[EventObject]
