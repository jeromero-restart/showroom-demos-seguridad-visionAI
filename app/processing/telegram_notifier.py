"""
Telegram alerts: post each rule-engine alarm, with the annotated frame that fired it,
to a fixed Telegram chat through a bot.

Enabled only when TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set. Sending runs on its
own daemon thread so a slow Telegram API never stalls the video loop, and alerts are
throttled to one per camera every TELEGRAM_MIN_INTERVAL_S seconds: demo videos loop
and Telegram rate-limits messages per chat.

Never log the request URL — it carries the bot token.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable

logger = logging.getLogger(__name__)

API_URL = "https://api.telegram.org/bot{token}/sendPhoto"
SEND_TIMEOUT_S = 10
QUEUE_SIZE = 8
MAX_EVENTS_IN_CAPTION = 3

ENTITY_LABELS = {"person": "Persona", "vehicle": "Vehículo", "animal": "Animal"}
DIRECTION_LABELS = {"N": "Norte", "S": "Sur", "E": "Este", "W": "Oeste"}


def _event_line(event: dict) -> str:
    entity = ENTITY_LABELS.get(event.get("class_name"), event.get("class_name") or "Objeto")
    p = event.get("trigger_params") or {}
    ttype = event.get("trigger_type")
    if ttype == "count":
        reason = f"{p.get('actual_count')} en la zona (mínimo {p.get('min_count')})"
    elif ttype == "dwell":
        reason = f"permanencia de {p.get('actual_s')} s (umbral {p.get('threshold_s')} s)"
    elif ttype == "direction":
        reason = f"ingreso desde el {DIRECTION_LABELS.get(p.get('direction'), p.get('direction'))}"
    else:
        reason = "detección"
    return f"{entity} #{event.get('track_id')} · {reason}"


def build_caption(camera_name: str, events: list[dict]) -> str:
    """Caption for one alert photo; several events on the same frame share it."""
    lines = [f"🚨 Alerta · {camera_name}"]
    lines += [_event_line(ev) for ev in events[:MAX_EVENTS_IN_CAPTION]]
    if len(events) > MAX_EVENTS_IN_CAPTION:
        lines.append(f"+{len(events) - MAX_EVENTS_IN_CAPTION} más")
    return "\n".join(lines)


def _multipart(fields: dict[str, str], photo: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        for name, value in fields.items()
    ]
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; filename="alerta.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n".encode() + photo + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class TelegramNotifier:
    """Throttled, non-blocking sender of alert photos to one Telegram chat."""

    def __init__(
        self,
        token: str,
        chat_id: str,
        min_interval_s: float = 30.0,
        sender: Callable[[str, bytes], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._token = token
        self._chat_id = chat_id
        self._min_interval_s = min_interval_s
        self._send = sender or self._send_photo
        self._clock = clock
        self._last_sent: dict[str, float] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[tuple[str, bytes]] = queue.Queue(maxsize=QUEUE_SIZE)
        self._thread: threading.Thread | None = None

    @classmethod
    def from_env(cls) -> TelegramNotifier | None:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not token or not chat_id:
            logger.info("Telegram alerts disabled (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set)")
            return None
        interval = float(os.environ.get("TELEGRAM_MIN_INTERVAL_S", "30"))
        logger.info(f"Telegram alerts enabled (one per camera every {interval:.0f}s)")
        return cls(token, chat_id, min_interval_s=interval)

    def notify(self, key: str, caption: str, encode: Callable[[], bytes | None]) -> bool:
        """
        Queue an alert photo unless `key` (the camera) alerted less than min_interval_s ago.
        `encode` produces the JPEG and only runs when the alert is actually sent.
        Returns True when the alert was queued.
        """
        now = self._clock()
        with self._lock:
            last = self._last_sent.get(key)
            if last is not None and now - last < self._min_interval_s:
                return False
            self._last_sent[key] = now
            if self._thread is None:
                self._thread = threading.Thread(target=self._worker, daemon=True, name="telegram-notifier")
                self._thread.start()
        photo = encode()
        if not photo:
            return False
        try:
            self._queue.put_nowait((caption, photo))
        except queue.Full:
            logger.warning("Telegram queue full; dropping alert")
            return False
        return True

    def _worker(self) -> None:
        while True:
            caption, photo = self._queue.get()
            try:
                self._send(caption, photo)
            except Exception:
                logger.exception("Telegram alert failed")

    def _send_photo(self, caption: str, photo: bytes) -> None:
        body, content_type = _multipart({"chat_id": self._chat_id, "caption": caption}, photo)
        req = urllib.request.Request(
            API_URL.format(token=self._token),
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=SEND_TIMEOUT_S) as resp:
                resp.read()
        except urllib.error.HTTPError as exc:
            # Telegram explains the failure in the JSON body (bad chat id, rate limit...).
            try:
                detail = json.loads(exc.read()).get("description", "")
            except Exception:
                detail = ""
            logger.warning(f"Telegram alert rejected: HTTP {exc.code} {detail}")
        except urllib.error.URLError as exc:
            logger.warning(f"Telegram unreachable: {exc.reason}")


_notifier: TelegramNotifier | None = None
_notifier_loaded = False
_notifier_lock = threading.Lock()


def get_notifier() -> TelegramNotifier | None:
    """Process-wide notifier, built from the environment on first use (None if disabled)."""
    global _notifier, _notifier_loaded
    with _notifier_lock:
        if not _notifier_loaded:
            _notifier = TelegramNotifier.from_env()
            _notifier_loaded = True
        return _notifier
