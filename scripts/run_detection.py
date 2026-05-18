#!/usr/bin/env python3
"""
CLI entrypoint for standalone YOLOv8+ByteTrack detection validation.

Usage (from backend/ directory):
    python scripts/run_detection.py \
        --video /app/data/uploads/test.mp4 \
        --area-config '{"polygon":[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]],"entity_type":"person","trigger":{"type":"count","params":{"min_count":1}}}' \
        --output /tmp/detections.json

This script proves Phase 2 Success Criterion 1:
"CLI script processes a real MP4 and writes a valid detections.json with frame-indexed detections and track IDs"
"""
import argparse
import json
import logging
import os
import sys

# Allow running from backend/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

VALID_ENTITY_TYPES = {"person", "vehicle", "animal"}
VALID_TRIGGER_TYPES = {"count", "dwell", "direction"}


def validate_area_config(area_config: dict) -> None:
    """Lightweight CLI validation of area_config shape."""
    if "polygon" not in area_config or len(area_config["polygon"]) < 3:
        raise ValueError("area_config.polygon must have at least 3 points")
    if area_config.get("entity_type") not in VALID_ENTITY_TYPES:
        raise ValueError(f"entity_type must be one of {VALID_ENTITY_TYPES}")
    trigger = area_config.get("trigger", {})
    if trigger.get("type") not in VALID_TRIGGER_TYPES:
        raise ValueError(f"trigger.type must be one of {VALID_TRIGGER_TYPES}")


def main():
    parser = argparse.ArgumentParser(description="SIALAR CLI Detection Runner")
    parser.add_argument("--video", required=True, help="Path to MP4 file")
    parser.add_argument(
        "--area-config",
        required=True,
        help='JSON string with keys: polygon, entity_type, trigger',
    )
    parser.add_argument("--output", default="/tmp/detections.json", help="Output path for detections.json")
    args = parser.parse_args()

    area_config = json.loads(args.area_config)
    validate_area_config(area_config)
    area_id = "cli-test-area"

    logger.info(f"Loading YOLOv8s model...")
    from ultralytics import YOLO
    model = YOLO("yolov8m.pt")

    from app.processing.detector import run_detection

    def progress_cb(pct, frames_done, total):
        logger.info(f"Progress: {pct}% ({frames_done}/{total} frames processed)")

    result = run_detection(
        model=model,
        video_path=args.video,
        area_config=area_config,
        area_id=area_id,
        progress_callback=progress_cb,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"detections.json written to {args.output}")
    logger.info(f"Metadata: {result['metadata']}")
    logger.info(f"Frames with detections: {len(result['frames'])}")

    # Validate output structure matches D-13 schema
    assert "metadata" in result, "Missing metadata"
    assert "frames" in result, "Missing frames"
    assert "events" in result, "Missing events"
    assert "fps" in result["metadata"], "Missing metadata.fps"
    assert "processed_fps" in result["metadata"], "Missing metadata.processed_fps"
    assert "total_frames" in result["metadata"], "Missing metadata.total_frames"
    assert "area_id" in result["metadata"], "Missing metadata.area_id"

    if result["frames"]:
        first_frame_key = next(iter(result["frames"]))
        assert isinstance(first_frame_key, str), "Frame keys must be strings (per D-13)"
        first_detection = result["frames"][first_frame_key][0]
        required_fields = {"track_id", "class_name", "confidence", "bbox", "in_zone"}
        missing = required_fields - set(first_detection.keys())
        assert not missing, f"Detection missing fields: {missing}"
        assert len(first_detection["bbox"]) == 4, "bbox must have 4 values [x1,y1,x2,y2]"
        logger.info(f"Schema validation PASSED — sample detection: {first_detection}")
    else:
        logger.warning("No detections found in video (entity not present or wrong entity_type)")

    logger.info("CLI run complete — detections.json is valid.")


if __name__ == "__main__":
    main()
