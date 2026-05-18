# Stable re-export: both Plan 02-01 and Plan 02-02 import get_connection from here.
# Do NOT import directly from app.db.connection in application code.
from app.db.connection import get_connection  # noqa: F401
