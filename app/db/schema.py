from app.db.connection import get_connection

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS areas (
    area_id     TEXT PRIMARY KEY,
    camera_id   TEXT NOT NULL,
    polygon     TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    trigger     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,
    area_id      TEXT NOT NULL REFERENCES areas(area_id),
    video_path   TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'queued',
    progress_pct INTEGER NOT NULL DEFAULT 0,
    result_path  TEXT,
    error_msg    TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id       TEXT PRIMARY KEY,
    job_id         TEXT NOT NULL REFERENCES jobs(job_id),
    frame_number   INTEGER NOT NULL,
    timestamp_s    REAL NOT NULL,
    track_id       INTEGER NOT NULL,
    class_name     TEXT NOT NULL,
    trigger_type   TEXT NOT NULL,
    trigger_params TEXT NOT NULL
);
"""


def init_db(db_path: str | None = None) -> None:
    """Create all tables if they do not exist. Safe to call multiple times."""
    kwargs = {"db_path": db_path} if db_path else {}
    conn = get_connection(**kwargs)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()
