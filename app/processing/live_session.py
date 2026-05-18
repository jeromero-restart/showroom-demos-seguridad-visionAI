"""
Live MJPEG streaming session for one (camera_id, area_id) pair.

A single daemon thread decodes the source video at source_fps using cv2,
runs YOLO every `yolo_stride` frames (reusing last detections in between),
burns bboxes + zone polygon onto each frame, JPEG-encodes, and fans out
to N subscribers. RuleEngine events are fanned out on a separate per-subscriber
queue so the SSE endpoint can expose them in real time.

Why this shape:
- The browser's <img> MJPEG decoder ties video + bboxes together perfectly:
  whatever the user sees IS the frame YOLO just ran on.
- Events go on a parallel SSE channel for the side panel; tiny ordering jitter
  vs. video frames is acceptable because the panel is cumulative.
- A process-singleton manager lets the MJPEG endpoint and the SSE endpoint share
  the same YOLO loop — otherwise each connection would re-process the video
  independently and the two would drift.

Lifecycle (per session):
    creating -> running -> (EOF reached) frozen -> stopped
                ^                         |
                +---- subscribers > 0 ----+
    When subscribers drop to 0 the session.stop_event is set and the manager
    deregisters the session. A future request creates a fresh session.

Concurrency caveats (POC scope):
- model.track(persist=True) keeps tracker state on the model object. Two
  concurrent sessions on different areas/cameras share that state and may
  see track ids leak between sessions. Acceptable for a single-user demo.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import queue
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import torch
from shapely.geometry import Polygon
from ultralytics.trackers.basetrack import BaseTrack

from app.processing.detector import (
    CONF_THRESHOLD,
    EntityTrackPostProcessor,
    PROCESSED_FPS,
    extract_detections_from_result,
    get_model_classes,
)
from app.processing.rule_engine import RuleEngine

_TRACKER_CFG = os.path.join(os.path.dirname(__file__), "bytetrack_nogmc.yaml")

logger = logging.getLogger(__name__)

JPEG_QUALITY = 75
JPEG_QUEUE_SIZE = 5     # ~200ms of buffering at 25fps before we drop
EVENT_QUEUE_SIZE = 256  # events are sparse; this just guards backlog spikes
FROZEN_FRAME_INTERVAL_S = 1.0  # how often to re-emit the "stream finalizado" frame
STREAM_FPS = max(1, int(os.getenv("STREAM_FPS", "10")))  # max frames/s emitted to subscribers

# BGR drawing colors
COLOR_IN_ZONE = (0, 64, 220)
COLOR_OUT_ZONE = (180, 180, 180)
COLOR_ZONE = (0, 220, 220)
COLOR_LABEL_BG = (0, 0, 0)
COLOR_LABEL_TEXT = (255, 255, 255)


class Subscriber:
    """One HTTP request's view onto a LiveSession."""

    __slots__ = ("sub_id", "jpeg_q", "event_q")

    def __init__(self) -> None:
        self.sub_id = uuid.uuid4().hex[:8]
        # `None` is a sentinel meaning "session is gone, stop iterating"
        self.jpeg_q: queue.Queue[bytes | None] = queue.Queue(maxsize=JPEG_QUEUE_SIZE)
        self.event_q: queue.Queue[dict | None] = queue.Queue(maxsize=EVENT_QUEUE_SIZE)


class LiveSession:
    """One daemon-threaded YOLO loop shared by all subscribers of a (camera, area)."""

    def __init__(
        self,
        camera_id: str,
        area_id: str,
        video_path: str,
        area_config: dict,
        model: Any,
    ) -> None:
        self.camera_id = camera_id
        self.area_id = area_id
        self.video_path = video_path
        self.area_config = area_config
        self.model = model

        self._subscribers: set[Subscriber] = set()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._last_jpeg: bytes | None = None
        # Manager assigns this after construction; called when subscriber count hits 0.
        self._on_empty: Callable[[], None] | None = None
        # Trigger hot-swap: set by update_trigger(), consumed inside _loop_video without restart.
        self._pending_trigger: dict | None = None
        self._trigger_lock = threading.Lock()

    # ---------------- public API ----------------

    @property
    def tag(self) -> str:
        return f"{self.camera_id}/{self.area_id[:8]}"

    def config_matches(self, other_config: dict) -> bool:
        """
        Check if the given area_config matches this session's stored config.
        Compares only the trigger-relevant fields: entity_type and trigger.
        Returns True if configs match, False if they differ.
        """
        try:
            # Compare entity_type
            if self.area_config.get("entity_type") != other_config.get("entity_type"):
                logger.debug(
                    f"[{self.tag}] config mismatch: entity_type {self.area_config.get('entity_type')} "
                    f"!= {other_config.get('entity_type')}"
                )
                return False

            # Compare trigger (deep comparison of dicts)
            if self.area_config.get("trigger") != other_config.get("trigger"):
                logger.debug(
                    f"[{self.tag}] config mismatch: trigger {self.area_config.get('trigger')} "
                    f"!= {other_config.get('trigger')}"
                )
                return False

            return True
        except Exception as e:
            logger.warning(f"[{self.tag}] error comparing configs: {e}")
            return False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"live-{self.tag}"
        )
        self._thread.start()

    def is_dying(self) -> bool:
        return self._stop_event.is_set() or self._stopped

    def _can_hot_swap(self, other_config: dict) -> bool:
        """True when only the trigger changed: entity_type and polygon are identical."""
        return (
            self.area_config.get("entity_type") == other_config.get("entity_type")
            and self.area_config.get("polygon") == other_config.get("polygon")
        )

    def update_trigger(self, new_trigger: dict) -> None:
        """Hot-swap the trigger config without restarting the session or the tracker."""
        with self._trigger_lock:
            self._pending_trigger = new_trigger
        self.area_config["trigger"] = new_trigger
        logger.info(f"[{self.tag}] trigger update queued (hot-swap)")

    def _remap(self, new_area_id: str, new_area_config: dict) -> None:
        """Reassign area_id and queue a trigger swap. Tracker state and video position preserved."""
        old_tag = self.tag
        self.area_id = new_area_id
        self.area_config = dict(new_area_config)
        with self._trigger_lock:
            self._pending_trigger = new_area_config["trigger"]
        logger.info(f"[{old_tag}] remapped → [{self.tag}] (tracker preserved, trigger queued)")

    def subscribe(self) -> Subscriber:
        sub = Subscriber()
        with self._lock:
            self._subscribers.add(sub)
            primer = self._last_jpeg
            count = len(self._subscribers)
        # Prime so the <img> renders something immediately rather than a blank.
        if primer is not None:
            try:
                sub.jpeg_q.put_nowait(primer)
            except queue.Full:
                pass
        logger.info(f"[{self.tag}] subscriber attached: {sub.sub_id} (total={count})")
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        with self._lock:
            self._subscribers.discard(sub)
            remaining = len(self._subscribers)
        # Sentinels so consumers blocked on .get() exit cleanly.
        for q in (sub.jpeg_q, sub.event_q):
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        logger.info(f"[{self.tag}] subscriber detached: {sub.sub_id} (remaining={remaining})")
        if remaining == 0:
            self.stop()

    def stop(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        if self._on_empty is not None:
            self._on_empty()

    # ---------------- thread loop ----------------

    def _run(self) -> None:
        try:
            self._loop_video()
        except Exception:
            logger.exception(f"[{self.tag}] live session crashed")
            self._fanout_event({"type": "error", "message": "Internal error in live session"})
        finally:
            self._stopped = True
            with self._lock:
                subs = list(self._subscribers)
            for sub in subs:
                for q in (sub.jpeg_q, sub.event_q):
                    try:
                        q.put_nowait(None)
                    except queue.Full:
                        pass
            logger.info(f"[{self.tag}] live session thread exited")

    def _loop_video(self) -> None:
        # The model is always fresh at session start (LiveSessionManager._fresh_model() was called
        # before this session was created). No surgical predictor/tracker reset needed here.
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        try:
            source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Stream stride: only 1 in stream_stride source frames is decoded and emitted.
            # The rest are grabbed (no decode, no JPEG encode) — cuts CPU ~60-70%.
            stream_stride = max(1, round(source_fps / STREAM_FPS))
            effective_fps = source_fps / stream_stride
            frame_period = 1.0 / effective_fps          # advance only on emitted frames
            yolo_stride = max(1, round(effective_fps / PROCESSED_FPS))  # relative to streamed frames

            entity_type = self.area_config["entity_type"]
            classes = get_model_classes(entity_type)
            post_processor = EntityTrackPostProcessor(entity_type)
            zone_polygon = Polygon(self.area_config["polygon"])
            zone_pixels = np.array(
                [[int(x * frame_w), int(y * frame_h)] for x, y in self.area_config["polygon"]],
                dtype=np.int32,
            )

            rule_engine = RuleEngine(
                trigger_config=self.area_config["trigger"],
                source_fps=source_fps,
                area_id=self.area_id,
                entity_class=entity_type,
            )

            last_detections: list[dict] = []
            frame_idx = 0        # counts ALL source frames (including grabbed)
            streamed_idx = 0     # counts only emitted/decoded frames
            next_tick = time.monotonic()
            yolo_called_once = False  # flag to reset tracker after first YOLO inference

            logger.info(
                f"[{self.tag}] live loop start: source_fps={source_fps:.1f}, "
                f"stream_stride={stream_stride}, effective_fps={effective_fps:.1f}, "
                f"yolo_stride={yolo_stride}, entity={entity_type}"
            )

            while not self._stop_event.is_set():
                # Hot-swap trigger config if update_trigger() was called while running.
                with self._trigger_lock:
                    if self._pending_trigger is not None:
                        rule_engine = RuleEngine(
                            trigger_config=self._pending_trigger,
                            source_fps=source_fps,
                            area_id=self.area_id,
                            entity_class=entity_type,
                        )
                        self._pending_trigger = None
                        logger.info(f"[{self.tag}] trigger config swapped in-place")

                # Skip non-output source frames cheaply (no decode, no encode).
                if frame_idx % stream_stride != 0:
                    grabbed = cap.grab()
                    if not grabbed:
                        break  # EOF during grab
                    frame_idx += 1
                    continue

                # Decode this source frame (it is an output frame).
                ok, frame = cap.read()
                if not ok:
                    break  # EOF

                # YOLO stride is relative to streamed (decoded) frames.
                if streamed_idx % yolo_stride == 0:
                    try:
                        results = self.model.track(
                            source=frame,
                            persist=True,
                            conf=CONF_THRESHOLD,
                            classes=classes,
                            verbose=False,
                            tracker=_TRACKER_CFG,
                            iou=0.45,  # Lower NMS threshold to allow overlapping small objects (occlusion)
                        )

                        # Reset the global BaseTrack ID counter on first inference so track IDs
                        # start at 1. The model is already fresh (reinstantiated by the manager),
                        # but BaseTrack._count is a class-level counter shared across all instances.
                        if not yolo_called_once:
                            yolo_called_once = True
                            try:
                                BaseTrack.reset_id()
                                logger.debug(f"[{self.tag}] reset BaseTrack global ID counter")
                            except Exception as e:
                                logger.warning(f"[{self.tag}] failed to reset BaseTrack ID: {e}")

                        if results:
                            last_detections = extract_detections_from_result(results[0], zone_polygon)
                            last_detections = post_processor.process(last_detections)
                            if last_detections:
                                logger.info(
                                    f"[{self.tag}] frame {frame_idx}: {len(last_detections)} detection(s) "
                                    f"(in_zone: {sum(1 for d in last_detections if d.get('in_zone'))})"
                                )
                            events = rule_engine.evaluate(frame_idx, last_detections)
                            for ev in events:
                                if ev is not None:
                                    self._fanout_event({**ev, "type": "event"})
                    except cv2.error as exc:
                        # Frame-size mismatch in ByteTrack GMC (e.g. variable-resolution video).
                        # Reset tracker so the next frame starts fresh instead of crashing the session.
                        logger.warning(
                            f"[{self.tag}] cv2 error at frame {frame_idx}, resetting tracker: {exc}"
                        )
                        if hasattr(self.model, "predictor") and self.model.predictor is not None:
                            try:
                                self.model.predictor = None
                            except AttributeError:
                                pass
                        last_detections = []

                annotated = self._annotate(frame, last_detections, zone_pixels, frame_w, frame_h)
                ok2, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if ok2:
                    payload = buf.tobytes()
                    self._last_jpeg = payload
                    self._fanout_jpeg(payload)

                # Throttle to effective_fps. next_tick only advances on emitted frames.
                next_tick += frame_period
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    if self._stop_event.wait(timeout=sleep_for):
                        return
                else:
                    next_tick = time.monotonic()

                frame_idx += 1
                streamed_idx += 1

            # EOF reached cleanly: announce end-of-stream and freeze on a static frame.
            if not self._stop_event.is_set():
                self._fanout_event({"type": "ended"})
                frozen = self._build_ended_frame(frame_w, frame_h)
                ok3, buf = cv2.imencode(".jpg", frozen, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if ok3:
                    self._last_jpeg = buf.tobytes()
                while not self._stop_event.is_set():
                    if self._last_jpeg is not None:
                        self._fanout_jpeg(self._last_jpeg)
                    if self._stop_event.wait(timeout=FROZEN_FRAME_INTERVAL_S):
                        return
        finally:
            cap.release()

    # ---------------- annotation ----------------

    def _annotate(
        self,
        frame: np.ndarray,
        detections: list[dict],
        zone_pixels: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> np.ndarray:
        out = frame  # cap.read() returns a fresh buffer each call; mutate in place

        if zone_pixels.shape[0] >= 3:
            overlay = out.copy()
            cv2.fillPoly(overlay, [zone_pixels], COLOR_ZONE)
            cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)
            cv2.polylines(out, [zone_pixels], isClosed=True, color=COLOR_ZONE, thickness=2)

        for det in detections:
            x1n, y1n, x2n, y2n = det["bbox"]
            x1, y1 = int(x1n * frame_w), int(y1n * frame_h)
            x2, y2 = int(x2n * frame_w), int(y2n * frame_h)
            color = COLOR_IN_ZONE if det.get("in_zone") else COLOR_OUT_ZONE
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"{det['class_name']} #{det['track_id']}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly = max(y1 - 6, th + 4)
            cv2.rectangle(out, (x1, ly - th - 4), (x1 + tw + 6, ly + 2), COLOR_LABEL_BG, -1)
            cv2.putText(
                out, label, (x1 + 3, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LABEL_TEXT, 1, cv2.LINE_AA,
            )

        return out

    def _build_ended_frame(self, w: int, h: int) -> np.ndarray:
        """Dim the last decoded frame and overlay 'Stream finalizado'."""
        text = "Stream finalizado"
        if self._last_jpeg is not None:
            decoded = cv2.imdecode(np.frombuffer(self._last_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is not None:
                canvas = (decoded * 0.45).astype(np.uint8)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
                cv2.putText(
                    canvas, text,
                    ((canvas.shape[1] - tw) // 2, (canvas.shape[0] + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA,
                )
                return canvas
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cv2.putText(
            canvas, text, ((w - tw) // 2, (h + th) // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA,
        )
        return canvas

    # ---------------- fanout ----------------

    def _fanout_jpeg(self, payload: bytes) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for sub in subs:
            try:
                sub.jpeg_q.put_nowait(payload)
            except queue.Full:
                # Slow consumer: drop oldest, push newest. Live > complete.
                try:
                    sub.jpeg_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    sub.jpeg_q.put_nowait(payload)
                except queue.Full:
                    pass

    def _fanout_event(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for sub in subs:
            try:
                sub.event_q.put_nowait(event)
            except queue.Full:
                logger.warning(f"[{self.tag}] event queue full for sub {sub.sub_id}; dropping")


class LiveSessionManager:
    """Process-wide registry of LiveSessions keyed by (camera_id, area_id)."""

    def __init__(self, model_factory: Callable[[], Any]) -> None:
        self._factory = model_factory
        self._model: Any = model_factory()
        self._sessions: dict[tuple[str, str], LiveSession] = {}
        self._lock = threading.Lock()

    @property
    def model(self) -> Any:
        return self._model

    def _fresh_model(self) -> None:
        """Reinstantiate YOLO completely — clears all ByteTrack state at every level."""
        logger.info("reinstantiating YOLO model for clean tracker state")
        self._model = self._factory()

    def _dispose_session(self, session: "LiveSession") -> None:
        """
        Fully tear down a LiveSession that has ALREADY been popped from
        `self._sessions` by the caller.

        Sequence:
          1. Set the session's stop_event so the daemon loop exits at its
             next wait/iteration.
          2. Snapshot + clear subscribers under the session's own lock,
             then push the None sentinel onto each subscriber queue using
             the same drop-oldest-then-push pattern as `_fanout_jpeg`.
          3. Bounded thread.join(timeout=5.0); warn-log if still alive.
          4. Drop the per-session model reference, gc.collect(),
             torch.cuda.empty_cache() (guarded + try/except).

        Invariants (load-bearing for deadlock safety):
          - Caller MUST have already removed `session` from `self._sessions`.
          - Caller MUST NOT be holding `self._lock` when invoking this.
            The thread.join() and any on_empty callbacks must run free of
            the manager lock or they will deadlock against subscribe/unsubscribe.
          - Never touches `self._model` (the manager-level reference used by
            the /health 503 guard) or `self._factory`.
        """
        thread = session._thread

        # 1) Tell the loop to exit at the next wait/iteration.
        session._stop_event.set()

        # 2) Detach subscribers and signal them so MJPEG/SSE generators
        #    return immediately rather than waiting on their queues.
        with session._lock:
            subs = list(session._subscribers)
            session._subscribers.clear()
        for sub in subs:
            for q in (sub.jpeg_q, sub.event_q):
                # Drop oldest if full, then push the None sentinel — same
                # slow-consumer pattern as _fanout_jpeg.
                try:
                    q.put_nowait(None)
                except queue.Full:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        q.put_nowait(None)
                    except queue.Full:
                        pass

        # 3) Join with a bounded timeout so HTTP requests can't hang.
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning(
                    f"[{session.tag}] dispose: thread did not exit within 5s; "
                    "leaving as daemon (will die on process exit)"
                )

        # 4) Drop the per-session model reference and reclaim memory.
        session.model = None
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                # Never let cache cleanup break the dispose path.
                logger.warning(f"[{session.tag}] torch.cuda.empty_cache() failed: {exc}")

        logger.info(
            f"[{session.tag}] dispose complete: "
            f"thread_alive={thread.is_alive() if thread else False}"
        )

    def kill_session(self, camera_id: str, area_id: str) -> bool:
        """
        Force-evict the LiveSession at (camera_id, area_id), join its thread, and
        release its YOLO model so VRAM/RAM is reclaimed.

        Idempotent: returns False when no session is registered for the key.
        Returns True when a session was found and killed (regardless of whether
        the daemon thread joined within the timeout — the operator gets a
        synchronous "stopped" confirmation either way).

        Never touches `self._model` (used by /health and the live endpoints'
        503 guard) or `self._factory` — only the per-session model reference.
        """
        key = (camera_id, area_id)

        # Pop from the registry under the lock; dispose outside the lock
        # (mirrors the get_or_create pattern — join must not hold the manager lock).
        with self._lock:
            session = self._sessions.pop(key, None)

        if session is None:
            return False

        self._dispose_session(session)
        return True

    def get_or_create(
        self,
        camera_id: str,
        area_id: str,
        video_path: str,
        area_config: dict,
    ) -> LiveSession:
        key = (camera_id, area_id)
        sessions_to_dispose: list[LiveSession] = []
        reusable_session: LiveSession | None = None

        # ---- Phase A: scan ALL sessions under the manager lock. ----
        # Classify each as either reusable (exact match / hot-swap / cross-area
        # remap on same camera) or to-dispose. Any session on a different camera
        # is a conflict in this single-user PoC and is queued for dispose.
        with self._lock:
            for k in list(self._sessions.keys()):
                s = self._sessions[k]
                if s.is_dying():
                    self._sessions.pop(k, None)
                    continue

                if k[0] == camera_id:
                    # Same camera — preserve all existing reuse paths.
                    if k == key:
                        # Same (camera, area_id): exact match or compatible swap?
                        if s.config_matches(area_config):
                            reusable_session = s
                            break
                        if s._can_hot_swap(area_config):
                            s.update_trigger(area_config["trigger"])
                            reusable_session = s
                            break
                        # entity_type or polygon changed → full restart.
                        logger.info(f"[{s.tag}] config incompatible; stopping for restart")
                        self._sessions.pop(k)
                        sessions_to_dispose.append(s)
                    else:
                        # Different area_id, same camera.
                        if s._can_hot_swap(area_config):
                            # Trigger changed via new area POST — remap to new key
                            # and swap the rule engine. Tracker + video position preserved.
                            self._sessions.pop(k)
                            s._remap(area_id, area_config)

                            def on_empty_remap(sess: LiveSession = s, nk: tuple = key) -> None:
                                with self._lock:
                                    if self._sessions.get(nk) is sess:
                                        self._sessions.pop(nk, None)
                                logger.info(f"live session deregistered: {nk[0]}/{nk[1][:8]}")

                            s._on_empty = on_empty_remap
                            self._sessions[key] = s
                            reusable_session = s
                            break
                        # Different entity_type or polygon → stop old, start fresh.
                        logger.info(
                            f"[{s.tag}] incompatible with new area {area_id[:8]}; stopping"
                        )
                        self._sessions.pop(k)
                        sessions_to_dispose.append(s)
                else:
                    # Different camera entirely. In this single-user PoC, only one
                    # live session may run at a time — auto-stop the prior one
                    # before starting the new one (no Detener click required).
                    logger.info(
                        f"[{s.tag}] auto-stopping: new live session requested "
                        f"on different camera {camera_id}"
                    )
                    self._sessions.pop(k)
                    sessions_to_dispose.append(s)

        # ---- Phase B: dispose conflicting sessions OUTSIDE the manager lock. ----
        # _dispose_session does the bounded join + gc + cuda cache reclaim.
        # Doing it under self._lock would deadlock against on_empty callbacks.
        for s in sessions_to_dispose:
            self._dispose_session(s)

        # Reusable session was found on the same camera and is, by construction,
        # NOT in sessions_to_dispose. Safe to return now that conflicts are gone.
        if reusable_session is not None:
            return reusable_session

        # ---- Phase C: re-acquire lock and create the new session. ----
        with self._lock:
            # Re-check: a concurrent request may have created the session
            # in the gap between phases.
            existing = self._sessions.get(key)
            if existing is not None and not existing.is_dying():
                if existing._can_hot_swap(area_config) or existing.config_matches(area_config):
                    return existing
                # Rare race: someone created an incompatible session.
                # Pop it so we can dispose it after releasing the lock.
                self._sessions.pop(key, None)
                race_loser = existing
            else:
                race_loser = None

            # Each new LiveSession gets its own fresh model instance from the factory.
            # This guarantees completely clean ByteTrack state (tracked_stracks, GMC,
            # Kalman filter) per session — no shared state between cameras or after restarts.
            session = LiveSession(
                camera_id=camera_id,
                area_id=area_id,
                video_path=video_path,
                area_config=area_config,
                model=self._factory(),
            )
            # Capture-by-default closure so a late-firing on_empty from a stale
            # session can't evict its successor.
            def on_empty(s: LiveSession = session, k: tuple[str, str] = key) -> None:
                with self._lock:
                    if self._sessions.get(k) is s:
                        self._sessions.pop(k, None)
                logger.info(f"live session deregistered: {k[0]}/{k[1][:8]}")

            session._on_empty = on_empty
            self._sessions[key] = session
            session.start()

        # Dispose the rare-race loser outside the manager lock.
        if race_loser is not None:
            self._dispose_session(race_loser)

        return session
