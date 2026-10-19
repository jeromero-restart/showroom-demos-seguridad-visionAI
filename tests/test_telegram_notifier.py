"""
Unit tests for TelegramNotifier — captions, per-camera throttle, queueing.

Run with: pytest tests/test_telegram_notifier.py -v
No network: the sender is replaced by a fake that records what would be posted.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading

import pytest
from app.processing import telegram_notifier
from app.processing.telegram_notifier import TelegramNotifier, build_caption


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeSender:
    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes]] = []
        self.done = threading.Event()

    def __call__(self, caption: str, photo: bytes) -> None:
        self.sent.append((caption, photo))
        self.done.set()


def make_event(trigger_type: str, params: dict, track_id: int = 3, class_name: str = "person") -> dict:
    return {
        "event_id": "e",
        "frame_number": 50,
        "timestamp_s": 2.0,
        "track_id": track_id,
        "class_name": class_name,
        "trigger_type": trigger_type,
        "trigger_params": params,
    }


@pytest.fixture
def notifier():
    clock, sender = FakeClock(), FakeSender()
    n = TelegramNotifier("token", "-100", min_interval_s=30, sender=sender, clock=clock)
    return n, clock, sender


# ---------------- captions ----------------

def test_caption_dwell():
    caption = build_caption("Perímetro edificio", [make_event("dwell", {"threshold_s": 5, "actual_s": 5.2})])
    assert caption == "🚨 Alerta · Perímetro edificio\nPersona #3 · permanencia de 5.2 s (umbral 5 s)"


def test_caption_count_and_direction():
    count = build_caption("Peatonal", [make_event("count", {"min_count": 2, "actual_count": 4})])
    assert count.endswith("Persona #3 · 4 en la zona (mínimo 2)")
    direction = build_caption("Cruce", [make_event("direction", {"direction": "W"}, class_name="vehicle")])
    assert direction.endswith("Vehículo #3 · ingreso desde el Oeste")


def test_caption_caps_events_per_photo():
    events = [make_event("dwell", {"threshold_s": 5, "actual_s": 6}, track_id=i) for i in range(5)]
    lines = build_caption("Cam", events).split("\n")
    assert len(lines) == 1 + 3 + 1
    assert lines[-1] == "+2 más"


# ---------------- throttle & queue ----------------

def test_sends_photo_with_caption(notifier):
    n, _, sender = notifier
    assert n.notify("cam_001", "hola", lambda: b"jpeg") is True
    assert sender.done.wait(2)
    assert sender.sent == [("hola", b"jpeg")]


def test_throttles_per_camera(notifier):
    n, clock, _ = notifier
    encodes = []

    def encode():
        encodes.append(1)
        return b"jpeg"

    assert n.notify("cam_001", "a", encode) is True
    clock.now += 10
    assert n.notify("cam_001", "b", encode) is False   # same camera, inside the window
    assert n.notify("cam_002", "c", encode) is True    # other camera is independent
    clock.now += 21
    assert n.notify("cam_001", "d", encode) is True    # window elapsed
    assert len(encodes) == 3                           # throttled alert never encodes


def test_failed_encode_is_not_sent(notifier):
    n, _, sender = notifier
    assert n.notify("cam_001", "a", lambda: None) is False
    assert sender.sent == []


# ---------------- configuration ----------------

def test_disabled_without_env(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100")
    assert TelegramNotifier.from_env() is None


def test_enabled_with_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100")
    monkeypatch.setenv("TELEGRAM_MIN_INTERVAL_S", "45")
    n = TelegramNotifier.from_env()
    assert n is not None and n._min_interval_s == 45


def test_multipart_body_carries_fields_and_photo():
    body, content_type = telegram_notifier._multipart({"chat_id": "-100", "caption": "Alerta · cámara"}, b"\xff\xd8jpeg")
    boundary = content_type.split("boundary=")[1]
    assert body.startswith(f"--{boundary}\r\n".encode())
    assert body.endswith(f"--{boundary}--\r\n".encode())
    assert 'name="caption"\r\n\r\nAlerta · cámara\r\n'.encode() in body
    assert b'filename="alerta.jpg"' in body and b"\xff\xd8jpeg" in body
