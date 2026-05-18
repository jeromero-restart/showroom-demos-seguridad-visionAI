"""
YOLOv8 + ByteTrack inference module.

Key design decisions:
- PROCESSED_FPS = 5 (downsample from source FPS via vid_stride — per CLAUDE.md recommendation)
- conf = 0.45 fixed (not exposed in UI — per STATE.md decisions)
- model.track(persist=True) required — NOT model.predict() — track IDs needed for dwell/direction
- opencv-python-headless only (no cv2 GUI functions — Docker headless constraint)
- Shapely Point.within(Polygon) for zone containment (normalized coords throughout)

Frame number semantics (IMPORTANT — RuleEngine depends on this):
- run_detection() uses frame_interval = round(source_fps / PROCESSED_FPS) to compute vid_stride.
- For each result from model.track(stream=True), actual_frame_number = frame_idx * frame_interval.
  Example: source_fps=25, PROCESSED_FPS=5 → frame_interval=5.
  Processed frame 0 → actual frame 0; frame 1 → actual frame 5; frame 2 → actual frame 10, etc.
- Frame keys in detections.json are STRING versions of these actual frame numbers ("0", "5", "10"...).
- source_fps is stored in metadata so RuleEngine can convert actual frame numbers to seconds:
    timestamp_s = actual_frame_number / source_fps
  (e.g., frame 10 at 25fps = 0.4 seconds into the source video)
- RuleEngine receives source_fps from the caller (jobs.py) to perform correct dwell math:
    (current_frame - entry_frame) / source_fps >= threshold_s
"""
from __future__ import annotations

import logging
import os
from typing import Any

import cv2
from shapely.geometry import LineString, Polygon

logger = logging.getLogger(__name__)

PROCESSED_FPS = 5   # frames per second to process (vid_stride computed from source FPS)
DEFAULT_CONF_THRESHOLD = 0.45
CONF_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", str(DEFAULT_CONF_THRESHOLD)))
BOTTOM_EDGE_ZONE_RATIO = 0.01
VEHICLE_TRACK_PROMOTION_THRESHOLD = float(
    os.getenv("VEHICLE_TRACK_PROMOTION_THRESHOLD", str(CONF_THRESHOLD))
)

logger.info(f"YOLOv8 confidence threshold: {CONF_THRESHOLD}")

ENTITY_CLASS_MAP: dict[str, list[int]] = {
    "person":  [0],
    "vehicle": [1, 2, 3, 5, 7],   # bicycle, car, motorcycle, bus, truck
    "animal":  [15, 16, 17, 18, 19],  # cat, dog, horse, sheep, cow
}

MODEL_CLASS_MAP: dict[str, list[int]] = {
    "person": ENTITY_CLASS_MAP["person"],
    "vehicle": [0, *ENTITY_CLASS_MAP["vehicle"]],
    "animal": ENTITY_CLASS_MAP["animal"],
}

TARGET_CLASS_NAMES: dict[str, set[str]] = {
    "person": {"person"},
    "vehicle": {"bicycle", "car", "motorcycle", "bus", "truck"},
    "animal": {"cat", "dog", "horse", "sheep", "cow"},
}


def get_video_metadata(video_path: str) -> dict:
    """Extract video metadata using cv2. Opens and closes VideoCapture immediately."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    duration_s = total_frames / source_fps if source_fps > 0 else 0.0
    return {
        "total_frames": total_frames,
        "source_fps": source_fps,
        "duration_s": round(duration_s, 3),
    }


def extract_detections_from_result(result: Any, zone_polygon: Polygon) -> list[dict]:
    """
    Extract per-frame detection dicts from a single ultralytics Results object.
    Returns list of detection dicts matching D-14 schema.

    Guards against:
    - boxes is None (no detections in frame)
    - boxes.id is None (ByteTrack hasn't confirmed tracks yet — early frames)
    - len(boxes) == 0

    Note: When boxes.id is None, detections are dropped. ByteTrack assigns IDs after
    min_hits frames of consistent detection. This is acceptable for a PoC — early
    unconfirmed detections are not stable enough for dwell/direction evaluation.
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    ids = boxes.id  # Tensor (N,) or None
    if ids is None:
        return []

    bboxes = boxes.xyxyn.tolist()    # [[x1,y1,x2,y2], ...] normalized
    cls_ids = boxes.cls.tolist()
    confs = boxes.conf.tolist()
    track_ids = ids.tolist()

    detections = []
    for track_id, bbox, cls_id, conf in zip(track_ids, bboxes, cls_ids, confs):
        x1, y1, x2, y2 = bbox
        # Use the central portion of the bottom edge instead of a single foot point.
        # This is slightly more permissive near zone boundaries while still grounding
        # the check on the contact area with the floor.
        bbox_width = x2 - x1
        segment_width = bbox_width #* BOTTOM_EDGE_ZONE_RATIO
        segment_x1 = ((x1 + x2) - segment_width) / 2
        segment_x2 = segment_x1 + segment_width
        bottom_segment = LineString([(segment_x1, y2), (segment_x2, y2)])
        overlap = bottom_segment.intersection(zone_polygon).length
        in_zone = overlap >= (segment_width * BOTTOM_EDGE_ZONE_RATIO)
        detections.append({
            "track_id": int(track_id),
            "class_name": result.names[int(cls_id)],
            "confidence": round(float(conf), 4),
            "bbox": [round(float(v), 6) for v in bbox],
            "in_zone": in_zone,
        })
    return detections


def get_model_classes(entity_type: str) -> list[int]:
    """Classes requested from YOLO for the given tracked entity type."""
    return MODEL_CLASS_MAP.get(entity_type, ENTITY_CLASS_MAP.get(entity_type, [0]))


class EntityTrackPostProcessor:
    """
    Keeps lightweight track-level state to smooth class flips across frames.

    For vehicle tracking we intentionally include class 0 ("person") in the YOLO call.
    If a track was confidently classified as a vehicle in previous frames, we keep
    accepting it as a vehicle on later frames even if the current frame flips to
    "person" due to frontal rider views.
    """

    def __init__(self, entity_type: str, promotion_threshold: float = VEHICLE_TRACK_PROMOTION_THRESHOLD) -> None:
        self.entity_type = entity_type
        self.promotion_threshold = promotion_threshold
        self._best_target_conf_by_track: dict[int, float] = {}

    def process(self, detections: list[dict]) -> list[dict]:
        target_class_names = TARGET_CLASS_NAMES.get(self.entity_type, set())
        if not target_class_names:
            return detections

        processed: list[dict] = []
        for detection in detections:
            track_id = detection["track_id"]
            class_name = detection["class_name"]
            confidence = float(detection["confidence"])

            if class_name in target_class_names:
                self._best_target_conf_by_track[track_id] = max(
                    confidence,
                    self._best_target_conf_by_track.get(track_id, 0.0),
                )
                processed.append(detection)
                continue

            if (
                self.entity_type == "vehicle"
                and class_name == "person"
                and self._best_target_conf_by_track.get(track_id, 0.0) >= self.promotion_threshold
            ):
                promoted = dict(detection)
                promoted["class_name"] = "vehicle"
                processed.append(promoted)

        return processed


def run_detection(
    model: Any,
    video_path: str,
    area_config: dict,
    area_id: str,
    progress_callback=None,
) -> dict:
    """
    Run YOLOv8+ByteTrack inference on video_path for the given area_config.

    Args:
        model: Loaded YOLO model (from app.state.model).
        video_path: Absolute path to the MP4 file.
        area_config: Dict with keys: polygon (list of [x,y] normalized), entity_type, trigger.
        area_id: UUID string identifying the area.
        progress_callback: Optional callable(progress_pct: int, frames_done: int, total_frames: int).
                           Called every 5% of processing progress.

    Returns:
        detections.json dict matching D-13 schema (metadata + frames + events).
        events list is EMPTY here — rule evaluation is done by RuleEngine (caller's responsibility).

    Frame number semantics:
        Frame keys in `frames` are actual video frame numbers as strings: "0", "5", "10", ...
        The caller (jobs.py) passes source_fps from metadata to RuleEngine for correct time math.
    """
    meta = get_video_metadata(video_path)
    total_frames = meta["total_frames"]
    source_fps = meta["source_fps"]

    frame_interval = 1 #max(1, round(source_fps / PROCESSED_FPS))
    # Actual processed FPS may differ from PROCESSED_FPS due to rounding
    actual_processed_fps = round(source_fps / frame_interval)

    entity_type = area_config["entity_type"]
    classes = get_model_classes(entity_type)
    post_processor = EntityTrackPostProcessor(entity_type)

    polygon_coords = area_config["polygon"]  # [[x,y], ...] normalized
    zone_polygon = Polygon(polygon_coords)

    logger.info(
        f"Starting detection: video={video_path}, entity={entity_type}, "
        f"source_fps={source_fps:.1f}, frame_interval={frame_interval}, "
        f"estimated_frames_to_process={total_frames // frame_interval}"
    )

    results_gen = model.track(
        source=video_path,
        stream=True,
        persist=True,
        conf=CONF_THRESHOLD,
        classes=classes,
        vid_stride=frame_interval,
        verbose=False,
        iou=0.45,  # Lower NMS threshold to allow overlapping small objects (occlusion)
    )

    frames_dict: dict[str, list[dict]] = {}
    frames_done = 0
    last_reported_pct = -1

    for frame_idx, result in enumerate(results_gen):
        actual_frame_number = frame_idx * frame_interval
        detections = extract_detections_from_result(result, zone_polygon)
        detections = post_processor.process(detections)
        if detections:  # only store frames with detections (saves disk space)
            frames_dict[str(actual_frame_number)] = detections
        frames_done += 1

        # Emit progress every 5% (per D-10)
        if progress_callback and total_frames > 0:
            pct = min(99, int((frames_done * frame_interval / total_frames) * 100))
            rounded_pct = (pct // 5) * 5
            if rounded_pct > last_reported_pct:
                last_reported_pct = rounded_pct
                progress_callback(rounded_pct, frames_done, total_frames // frame_interval)

    logger.info(f"Detection complete: {frames_done} frames processed, {len(frames_dict)} frames with detections")

    return {
        "metadata": {
            "fps": round(float(source_fps), 3),
            "processed_fps": actual_processed_fps,
            "total_frames": total_frames,
            "video_duration": round(meta["duration_s"], 3),
            "area_id": area_id,
        },
        "frames": frames_dict,
        "events": [],  # populated by RuleEngine after this function returns
    }
