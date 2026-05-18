"""
Job orchestration endpoints (router prefix: /api/jobs):
  POST /api/jobs                    — accept MP4 upload + area_id, start background inference
  GET  /api/jobs/{id}/progress      — SSE stream of progress events (D-10/D-11/D-12)
  GET  /api/jobs/{id}/results       — return completed detections.json (D-13/D-14/D-15)

Architecture:
- Inference runs in threading.Thread (NEVER async — blocks event loop)
- Progress communicated via queue.Queue (thread-safe sync queue)
- SSE async generator polls queue with asyncio.sleep(0.1) to not block event loop
- Per D-07: GPU available → realtime WebSocket mode (skeleton only in this phase)
              CPU only   → pre-compute mode (SSE + detections.json)
- WebSocket realtime frame streaming is Phase 4 integration work

Sparse frame handling:
- run_detection() stores only frames WITH detections in the frames dict.
- The rule engine is called only for stored frames. Between stored frames, tracks that
  disappear are treated as having exited by _update_dwell_exits() (seen_ids is empty
  for any track not in the current frame's detections).
- This is acceptable for a PoC — the exit detection may lag by up to 1 processed frame
  interval (0.2s at 5fps), which is not user-perceptible for dwell thresholds of 2s+.

DB import path: from app.db import get_connection
    (stable re-export from app.db.__init__, defined in app.db.connection)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.db import get_connection
from app.db.schema import init_db  # only for type reference; actual call is in main.py lifespan
from app.models.schemas import JobResponse
from app.processing.detector import run_detection
from app.processing.rule_engine import RuleEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobCreate(BaseModel):
    area_id: str

# In-memory job state: job_id -> {status, progress_queue, result}
# Per CLAUDE.md: BackgroundTasks + in-memory dict, NOT Celery + Redis
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

UPLOADS_DIR = os.environ.get("UPLOADS_DIR", "/app/data/uploads")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/app/data/results")


def _get_area_config(area_id: str) -> dict:
    """Fetch area config from SQLite. Raises 404 if not found."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM areas WHERE area_id = ?", (area_id,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"Area {area_id} not found")
    return {
        "camera_id": row["camera_id"],
        "polygon": json.loads(row["polygon"]),
        "entity_type": row["entity_type"],
        "trigger": json.loads(row["trigger"]),
    }


def _update_job_db(
    job_id: str,
    status: str,
    progress_pct: int,
    result_path: str | None = None,
    error_msg: str | None = None,
) -> None:
    """Update job row in SQLite. Creates new connection (safe for background thread)."""
    conn = get_connection()
    try:
        conn.execute(
            """
            UPDATE jobs
            SET status=?, progress_pct=?, result_path=?, error_msg=?, updated_at=?
            WHERE job_id=?
            """,
            (
                status,
                progress_pct,
                result_path,
                error_msg,
                datetime.now(timezone.utc).isoformat(),
                job_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _run_inference_thread(
    job_id: str,
    video_path: str,
    area_config: dict,
    area_id: str,
    model: Any,
    progress_q: queue.Queue,
) -> None:
    """
    Background thread: run YOLOv8 detection + rule engine, write results to disk.

    Uses its own SQLite connection per call (per Pitfall 5 in RESEARCH.md —
    do NOT share connections across threads).
    Puts progress dicts into progress_q for SSE endpoint to consume.

    RuleEngine receives source_fps from detections["metadata"]["fps"] for correct
    dwell timing and timestamp_s calculation (see rule_engine.py docstring).
    """
    total_frames_processed = 0
    try:
        logger.info(f"Job {job_id}: inference thread started")
        _update_job_db(job_id, "processing", 0)

        # --- Detection phase ---
        def progress_cb(pct: int, frames_done: int, total: int) -> None:
            nonlocal total_frames_processed
            total_frames_processed = total
            progress_q.put({
                "progress_pct": pct,
                "frames_done": frames_done,
                "total_frames": total,
                "status": "processing",
            })
            _update_job_db(job_id, "processing", pct)

        detections_result = run_detection(
            model=model,
            video_path=video_path,
            area_config=area_config,
            area_id=area_id,
            progress_callback=progress_cb,
        )

        # Clear batch job tracker state so the next live session or batch job starts clean.
        try:
            if hasattr(model, "predictor") and model.predictor is not None:
                model.predictor = None
                logger.debug("cleared model.predictor after batch job run_detection")
        except Exception:
            pass

        # --- Rule engine phase ---
        trigger_config = area_config["trigger"]
        entity_type = area_config["entity_type"]
        source_fps = detections_result["metadata"]["fps"]  # actual video fps (e.g. 25.0)

        rule_engine = RuleEngine(
            trigger_config=trigger_config,
            source_fps=source_fps,  # NOT processed_fps — required for correct dwell math
            area_id=area_id,
            entity_class=entity_type,
        )

        all_events: list[dict] = []
        # Track frames that had at least one in_zone=True detection — used as a
        # safety-net diagnostic when 0 events are generated (helps surface polygon
        # mis-configuration without flooding logs).
        frames_with_in_zone = 0
        frames_with_dets = 0
        # Sort frame keys numerically to ensure chronological evaluation order
        for frame_key in sorted(detections_result["frames"].keys(), key=lambda k: int(k)):
            frame_detections = detections_result["frames"][frame_key]
            frame_number = int(frame_key)
            if frame_detections:
                frames_with_dets += 1
                if any(d.get("in_zone") for d in frame_detections):
                    frames_with_in_zone += 1
            events = rule_engine.evaluate(frame_number=frame_number, detections=frame_detections)
            # Filter None returns from deduplication (per D-05/D-07)
            all_events.extend([e for e in events if e is not None])

        detections_result["events"] = all_events
        logger.info(f"Job {job_id}: rule engine generated {len(all_events)} events")

        # Safety-net diagnostic: surface the most common configuration mistake
        # (polygon doesn't overlap any detection foot points) without spamming
        # logs in normal operation.
        if not all_events and frames_with_dets > 0 and frames_with_in_zone == 0:
            logger.warning(
                f"Job {job_id}: 0 frames had in_zone=True out of {frames_with_dets} "
                f"frames with detections — polygon may not overlap detection foot points."
            )

        # --- Persist events to SQLite (single transaction) ---
        conn = get_connection()
        try:
            for event in all_events:
                conn.execute(
                    """
                    INSERT INTO events (event_id, job_id, frame_number, timestamp_s, track_id,
                                        class_name, trigger_type, trigger_params)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event["event_id"],
                        job_id,
                        event["frame_number"],
                        event["timestamp_s"],
                        event["track_id"],
                        event["class_name"],
                        event["trigger_type"],
                        json.dumps(event["trigger_params"]),
                    ),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        # --- Write detections.json to disk ---
        os.makedirs(RESULTS_DIR, exist_ok=True)
        result_path = os.path.join(RESULTS_DIR, f"{job_id}.json")
        with open(result_path, "w") as f:
            json.dump(detections_result, f)
        logger.info(f"Job {job_id}: detections.json written to {result_path}")

        # Store result in memory for fast retrieval
        with _jobs_lock:
            _jobs[job_id]["result"] = detections_result
            _jobs[job_id]["status"] = "done"

        _update_job_db(job_id, "done", 100, result_path=result_path)

        # Final SSE event (D-12): {progress_pct: 100, status: "done", job_id}
        progress_q.put({
            "progress_pct": 100,
            "frames_done": total_frames_processed,
            "total_frames": total_frames_processed,
            "status": "done",
            "job_id": job_id,
        })

    except Exception as exc:
        logger.exception(f"Job {job_id}: inference thread failed: {exc}")
        _update_job_db(job_id, "error", 0, error_msg=str(exc))
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
        progress_q.put({
            "progress_pct": 0,
            "frames_done": 0,
            "total_frames": 0,
            "status": "error",
            "job_id": job_id,
            "error": str(exc),
        })
    finally:
        progress_q.put(None)  # sentinel: SSE generator stops iteration


@router.post("", response_model=JobResponse, status_code=202)
def create_job(request: Request, body: JobCreate):
    """
    Accept area_id JSON body. Resolve demo video from DEMO_CAMERAS. Start background inference. Return job_id immediately.

    Per D-05: area must already exist via POST /api/areas.
    Per D-07: backend auto-selects pre-compute (CPU) or realtime (GPU) mode.
    Per D-08: returns {job_id, status: "queued"}.
    Phase 4: accepts JSON-only body with area_id — no file upload.
    """
    # Validate area exists
    area_config = _get_area_config(body.area_id)
    camera_id = area_config["camera_id"]

    # Resolve demo video from DEMO_CAMERAS
    from app.config.cameras import DEMO_CAMERAS
    demo_camera = next((c for c in DEMO_CAMERAS if c.id == camera_id), None)
    if not demo_camera:
        raise HTTPException(status_code=404, detail=f"No demo video configured for camera {camera_id}")
    video_path = demo_camera.video_path

    job_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    # Insert job record into SQLite
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO jobs (job_id, area_id, video_path, status, progress_pct, created_at, updated_at)
            VALUES (?, ?, ?, 'queued', 0, ?, ?)
            """,
            (job_id, body.area_id, video_path, created_at, created_at),
        )
        conn.commit()
    finally:
        conn.close()

    # Initialize in-memory job state
    progress_q: queue.Queue = queue.Queue()
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "progress_queue": progress_q,
            "result": None,
        }

    # Auto-select processing mode (D-07)
    model = request.app.state.live_sessions.model
    gpu_available = getattr(request.app.state, "gpu_info", {}).get("available", False)

    # Launch background inference thread
    # Note: realtime WebSocket streaming for GPU mode is Phase 4 integration work.
    # Both modes run the same background thread and produce detections.json.
    thread = threading.Thread(
        target=_run_inference_thread,
        args=(job_id, video_path, area_config, body.area_id, model, progress_q),
        daemon=True,
        name=f"inference-{job_id[:8]}",
    )
    thread.start()
    logger.info(f"Job {job_id}: inference thread started (gpu_available={gpu_available})")

    return JobResponse(job_id=job_id, status="queued")


@router.get("/{job_id}/progress")
async def stream_progress(job_id: str):
    """
    SSE stream of processing progress events.

    Per D-10: emits one event every 5% of processing progress.
    Per D-11: event payload {progress_pct, frames_done, total_frames, status}.
    Per D-12: final event has {progress_pct: 100, status: "done", job_id}.
    Client calls GET /api/jobs/{id}/results after receiving the done event.

    SSE headers include X-Accel-Buffering: no to override nginx response buffering
    at the application level (belt-and-suspenders alongside nginx proxy_buffering off).
    """
    if job_id not in _jobs:
        # Check DB as fallback (job may have been created before process restart)
        conn = get_connection()
        try:
            row = conn.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        finally:
            conn.close()
        if not row:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        status = row["status"]
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} has status '{status}' — SSE stream only available during active processing",
        )

    q: queue.Queue = _jobs[job_id]["progress_queue"]

    async def event_generator():
        while True:
            try:
                msg = q.get_nowait()
                if msg is None:  # sentinel — inference thread finished
                    break
                yield f"data: {json.dumps(msg)}\n\n"
            except queue.Empty:
                await asyncio.sleep(0.1)  # yield control to event loop without blocking

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # overrides nginx buffering per-response
        },
    )


@router.get("/{job_id}/results")
def get_results(job_id: str):
    """
    Return completed detections.json after processing finishes.

    Per D-12: client calls this endpoint after receiving {status: "done"} SSE event.
    Returns full D-13 structure: {metadata, frames, events}.
    """
    # Check in-memory cache first
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job and job.get("result"):
        return job["result"]

    # Fallback: load from disk (handles container restart scenario)
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT status, result_path, error_msg FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if row["status"] == "error":
        raise HTTPException(status_code=500, detail=f"Job failed: {row['error_msg']}")

    if row["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Job is still {row['status']} — results not yet available",
        )

    result_path = row["result_path"]
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=500, detail="Result file not found on disk")

    with open(result_path) as f:
        return json.load(f)
