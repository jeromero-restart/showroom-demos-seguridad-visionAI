import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("DB_PATH", "/app/data/sialar.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create a new SQLite connection. Call once per thread — do NOT share across threads."""
    try:
        db_dir = os.path.dirname(db_path)
        # Diagnostics are useful when permissions misbehave but get_connection() is
        # called per progress callback and per SSE poll, so keep them at DEBUG to
        # avoid flooding the console during normal operation. ERROR-level logs in
        # the except block below remain unconditional.
        logger.debug(f"Creating DB at: {db_path}")
        logger.debug(f"DB directory: {db_dir}")
        os.makedirs(db_dir, exist_ok=True)
        logger.debug(f"DB directory exists: {os.path.exists(db_dir)}")
        logger.debug(f"DB directory writable: {os.access(db_dir, os.W_OK)}")

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # safe for concurrent reads during inference write
        conn.execute("PRAGMA foreign_keys=ON")
        logger.debug(f"DB connection established: {db_path}")
        return conn
    except Exception as e:
        logger.error(f"Failed to create DB connection: {e}")
        logger.error(f"DB path: {db_path}")
        logger.error(f"CWD: {os.getcwd()}")
        logger.error(f"Exists: {os.path.exists(db_path)}")
        raise
