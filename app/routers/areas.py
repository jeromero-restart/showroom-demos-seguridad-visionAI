"""
POST /api/areas — save area configuration to SQLite.

Per D-05: Area config is saved separately before submitting a job.
Per D-06: Body schema is {camera_id, polygon, entity_type, trigger}. Returns {area_id}.

Router prefix is /api/areas — consistent with existing nginx /api/ location block (Phase 1).
"""
import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.db import get_connection
from app.models.schemas import AreaCreate, AreaResponse

router = APIRouter(prefix="/api/areas", tags=["areas"])


@router.get("", name="list_areas")
def list_areas(camera_id: str | None = None):
    """
    Return saved area configs. Filtered by camera_id if provided.
    Returns most recent area per camera (LIMIT 1) — PoC scope is 1 area per camera.
    Returns all areas (no LIMIT) when camera_id is not provided.
    """
    conn = get_connection()
    try:
        if camera_id:
            rows = conn.execute(
                "SELECT area_id, camera_id, polygon, entity_type, trigger, created_at "
                "FROM areas WHERE camera_id = ? ORDER BY created_at DESC LIMIT 1",
                (camera_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT area_id, camera_id, polygon, entity_type, trigger, created_at "
                "FROM areas ORDER BY created_at DESC",
            ).fetchall()
    finally:
        conn.close()

    result = []
    for row in rows:
        result.append({
            "area_id": row[0],
            "camera_id": row[1],
            "polygon": json.loads(row[2]),
            "entity_type": row[3],
            "trigger": json.loads(row[4]),
            "created_at": row[5],
        })
    return result


@router.post("", response_model=AreaResponse, status_code=201)
def create_area(body: AreaCreate):
    """
    Save area configuration. Returns area_id for use in POST /api/jobs.

    Polygon coordinate validation is performed by AreaCreate Pydantic model (min 3 points).
    Trigger param validation is performed by TriggerConfig @model_validator.
    Additional coordinate range check (all values in [0.0, 1.0]) done here before DB insert.
    """
    area_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    # Validate all polygon coordinates are in [0.0, 1.0]
    for point in body.polygon:
        if not (len(point) == 2 and 0.0 <= point[0] <= 1.0 and 0.0 <= point[1] <= 1.0):
            raise HTTPException(
                status_code=422,
                detail=f"All polygon coordinates must be in [0.0, 1.0] normalized range. Got: {point}",
            )

    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO areas (area_id, camera_id, polygon, entity_type, trigger, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                area_id,
                body.camera_id,
                json.dumps(body.polygon),
                body.entity_type,
                json.dumps(body.trigger.model_dump()),
                created_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return AreaResponse(area_id=area_id)
