"""
Pure Python rule engine for live cam detection video analytics.

NO FastAPI imports — this module is unit-testable in isolation.

Implements three trigger types per CONTEXT.md D-01/D-02/D-03/D-04:
- COUNT: fire once when simultaneous in-zone count >= N, re-arm when count drops below N
- DWELL: fire once when entity in zone >= T consecutive seconds, reset after entity exits
- DIRECTION: fire when entity enters zone from configured cardinal direction (N/S/E/W)
             Direction computed from velocity vector averaged over last 3-5 frames before entry
             Fires ONLY on actual outside→inside transition (tracked via _prev_in_zone per track_id)

Frame number semantics (CRITICAL for correct dwell math):
    Frame numbers passed to evaluate() are ACTUAL video frame numbers (e.g., 0, 5, 10 at 5fps processed
    from a 25fps source). They are NOT sequential processed-frame indices (0, 1, 2...).

    Time conversion: timestamp_s = frame_number / source_fps
    Dwell check:     (frame_number - entry_frame) / source_fps >= threshold_s

    The caller (jobs.py) passes source_fps from detections["metadata"]["fps"] when constructing RuleEngine.

Sparse frame handling:
    evaluate() must be called once per processed frame, even frames with zero detections.
    Zero-detection frames signal that ALL previously tracked entities have left the camera view.
    - DWELL: _update_dwell_exits() is called with an empty out_zone_ids set on zero-detection frames,
      but the caller passes all previously known track IDs as exited via _handle_all_exited().
    - DIRECTION: _prev_in_zone is cleared for unseen tracks on zero-detection frames.
    In practice, run_detection() only stores frames with detections in the frames dict, so the caller
    (jobs.py) must explicitly call evaluate() for stored frames only and accept that between stored
    frames, exits are inferred when a track_id disappears across consecutive frames.
"""
from __future__ import annotations

import uuid
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RuleEngine:
    """
    Stateful rule evaluator. Call evaluate() once per processed frame in order.

    Args:
        trigger_config: {"type": "count"|"dwell"|"direction", "params": {...}}
        source_fps: Source video FPS (e.g. 25.0). Used to convert actual frame numbers to seconds.
                    IMPORTANT: this is the SOURCE video fps, not the processed fps (5).
                    Obtain from detections["metadata"]["fps"] in the caller.
        area_id: Area UUID — embedded in generated events
        entity_class: Expected entity class name (e.g. "person")
    """

    # Minimum seconds between alerts of the same trigger type, regardless of
    # track_id. Debounces tracker ID switches / detection flicker for the demo.
    _TRIGGER_COOLDOWN_S = 3.0

    def __init__(
        self,
        trigger_config: dict,
        source_fps: float,
        area_id: str = "",
        entity_class: str = "person",
    ) -> None:
        self.trigger = trigger_config
        self.source_fps = source_fps
        self.area_id = area_id
        self.entity_class = entity_class

        # COUNT state
        self._count_armed: bool = True
        self._count_above_frames: int = 0  # consecutive frames at or above threshold

        # DWELL state — per track_id
        self._dwell_entry_frame: dict[int, int] = {}   # track_id -> actual frame number when entered zone
        self._dwell_fired: set[int] = set()             # tracks that fired (cleared on exit)

        # DIRECTION state — per track_id
        # Stores last N centroid positions BEFORE zone entry: (frame_number, cx, cy)
        self._pre_zone_positions: dict[int, list[tuple[int, float, float]]] = {}
        self._direction_fired: set[int] = set()         # tracks that already fired direction event
        # Track previous in-zone state to detect actual outside→inside transitions
        self._prev_in_zone: dict[int, bool] = {}        # track_id -> was_in_zone in previous frame

        # DEDUPLICATION state — per (track_id, trigger_type) pair
        self._recent_events: dict[tuple[int, str], tuple[float, str]] = {}
        # key: (track_id, trigger_type)
        # value: (last_event_timestamp_s, event_id)

        # Per-trigger-type cooldown: last fire time per trigger_type, regardless of
        # track_id. Absorbs tracker ID switches and detection flicker that would
        # otherwise spam alerts for the SAME physical entity under new track ids.
        self._last_type_fire_s: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Deduplication helpers
    # ------------------------------------------------------------------

    def _should_suppress_duplicate(self, track_id: int, trigger_type: str, timestamp_s: float) -> bool:
        """
        Check if event for this entity+trigger fired within last 2 seconds.
        Returns True if event should be suppressed (duplicate).

        Per D-06: temporal gate = 2 seconds.
        """
        # For COUNT trigger (track_id=-1), key by trigger_type only
        if track_id == -1:
            key = (-1, trigger_type)
        else:
            key = (track_id, trigger_type)

        if key not in self._recent_events:
            return False
        last_timestamp_s, _ = self._recent_events[key]
        time_since_last = timestamp_s - last_timestamp_s
        # Suppress if within 2-second window (per D-06)
        return time_since_last < 2.0

    def _record_event(self, track_id: int, trigger_type: str, timestamp_s: float, event_id: str) -> None:
        """Record event in dedup memory for future checks."""
        # For COUNT trigger (track_id=-1), key by trigger_type only
        if track_id == -1:
            key = (-1, trigger_type)
        else:
            key = (track_id, trigger_type)
        self._recent_events[key] = (timestamp_s, event_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, frame_number: int, detections: list[dict]) -> list[dict]:
        """
        Evaluate trigger conditions for one frame.

        Args:
            frame_number: The ACTUAL video frame number (e.g., 0, 5, 10 at 5fps processed
                          from a 25fps source). NOT a sequential processed-frame index.
            detections: List of detection dicts with fields: track_id, class_name,
                        confidence, bbox [x1,y1,x2,y2] normalized, in_zone bool.
                        May be empty (zero-detection frame — all entities absent from frame).

        Returns:
            List of event dicts (may be empty) matching D-15 schema.
        """
        trigger_type = self.trigger["type"]
        in_zone = [d for d in detections if d.get("in_zone")]
        out_zone_ids = {d["track_id"] for d in detections if not d.get("in_zone")}
        seen_ids = {d["track_id"] for d in detections}

        events: list[dict] = []

        if trigger_type == "count":
            events.extend(self._eval_count(frame_number, in_zone))
        elif trigger_type == "dwell":
            # On zero-detection frames, all previously tracked entities are treated as having exited
            self._update_dwell_exits(out_zone_ids, seen_ids)
            events.extend(self._eval_dwell(frame_number, in_zone))
        elif trigger_type == "direction":
            events.extend(self._eval_direction(frame_number, detections, in_zone, seen_ids))

        return events

    # ------------------------------------------------------------------
    # COUNT trigger (D-01)
    # ------------------------------------------------------------------

    # Require this many consecutive frames above threshold before COUNT fires.
    # Prevents single-frame YOLO false positives from triggering alarms.
    _COUNT_STABLE_FRAMES = 2

    def _eval_count(self, frame_number: int, in_zone: list[dict]) -> list[dict]:
        params = self.trigger["params"]
        min_count = int(params.get("min_count", 1))
        current_count = len(in_zone)

        events = []
        if current_count >= min_count:
            self._count_above_frames += 1
        else:
            self._count_above_frames = 0

        if self._count_armed and self._count_above_frames >= self._COUNT_STABLE_FRAMES:
            ev = self._make_event(
                frame_number=frame_number,
                track_id=-1,
                trigger_params={"min_count": min_count, "actual_count": current_count},
            )
            if ev is not None:
                events.append(ev)
                self._count_armed = False
                self._count_above_frames = 0
                logger.debug(f"COUNT fired at frame {frame_number}: count={current_count}>={min_count}")

        elif not self._count_armed and current_count < min_count:
            self._count_armed = True
            self._count_above_frames = 0
            logger.debug(f"COUNT re-armed at frame {frame_number}: count={current_count}<{min_count}")

        return events

    # ------------------------------------------------------------------
    # DWELL trigger (D-02)
    # ------------------------------------------------------------------

    def _update_dwell_exits(self, out_zone_ids: set[int], seen_ids: set[int]) -> None:
        """
        Clear dwell state for tracks that have left the zone or disappeared entirely.

        out_zone_ids: track_ids seen in this frame but not in-zone.
        seen_ids: ALL track_ids seen in this frame (in-zone + out-of-zone).

        A track that was previously in _dwell_entry_frame but is NOT in seen_ids at all
        is treated as having exited (sparse frame / track lost).
        """
        # Tracks explicitly out-of-zone this frame
        for tid in list(self._dwell_entry_frame.keys()):
            if tid in out_zone_ids or tid not in seen_ids:
                del self._dwell_entry_frame[tid]
                self._dwell_fired.discard(tid)  # re-arm: entity can trigger again on re-entry
                logger.debug(f"DWELL re-armed for track {tid} (exited zone or disappeared)")

    def _eval_dwell(self, frame_number: int, in_zone: list[dict]) -> list[dict]:
        params = self.trigger["params"]
        threshold_s = float(params.get("threshold_s", 3))

        events = []
        for detection in in_zone:
            tid = detection["track_id"]
            if tid not in self._dwell_entry_frame:
                self._dwell_entry_frame[tid] = frame_number
                logger.debug(f"DWELL tracking started for track {tid} at frame {frame_number}")

            if tid in self._dwell_fired:
                continue  # already fired for this entry — entity must exit to re-arm

            # Convert actual frame numbers to seconds using source_fps
            # Example: frame_number=50, entry_frame=0, source_fps=25 → 50/25 = 2.0s
            elapsed_s = (frame_number - self._dwell_entry_frame[tid]) / self.source_fps
            if elapsed_s >= threshold_s:
                events.append(self._make_event(
                    frame_number=frame_number,
                    track_id=tid,
                    trigger_params={"threshold_s": threshold_s, "actual_s": round(elapsed_s, 2)},
                ))
                self._dwell_fired.add(tid)
                logger.debug(f"DWELL fired for track {tid} at frame {frame_number}: {elapsed_s:.1f}s>={threshold_s}s")

        return events

    # ------------------------------------------------------------------
    # DIRECTION trigger (D-03/D-04)
    # ------------------------------------------------------------------

    def _centroid(self, bbox: list[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _compute_entry_direction(self, positions: list[tuple[int, float, float]]) -> str | None:
        """
        Compute cardinal direction from centroid velocity averaged over last 3-5 positions.
        positions: [(frame_number, cx, cy), ...] ordered by frame number.
        Returns "N", "S", "E", "W" or None if insufficient data.
        """
        if len(positions) < 2:
            return None
        # Use last min(5, len) positions for velocity average
        window = positions[-5:]
        dx_sum = 0.0
        dy_sum = 0.0
        for i in range(1, len(window)):
            dx_sum += window[i][1] - window[i - 1][1]  # cx delta
            dy_sum += window[i][2] - window[i - 1][2]  # cy delta
        steps = len(window) - 1
        mean_dx = dx_sum / steps
        mean_dy = dy_sum / steps

        if abs(mean_dx) >= abs(mean_dy):
            return "E" if mean_dx > 0 else "W"
        else:
            return "S" if mean_dy > 0 else "N"

    def _eval_direction(
        self,
        frame_number: int,
        all_detections: list[dict],
        in_zone: list[dict],
        seen_ids: set[int],
    ) -> list[dict]:
        """
        Evaluate DIRECTION trigger for this frame.

        Key correctness rules (fixes Issue 1 from code review):
        1. Pre-zone positions are accumulated ONLY for tracks that are currently OUTSIDE the zone.
        2. A direction event fires ONLY on an actual outside→inside transition:
           the track was NOT in-zone last frame (_prev_in_zone.get(tid) is False or absent)
           AND is in-zone this frame.
        3. _prev_in_zone is updated at the end of this method for all seen tracks.
        4. Tracks not seen this frame are removed from _prev_in_zone (lost track = exit).
        """
        params = self.trigger["params"]
        configured_dir = str(params.get("direction", "N")).upper()

        in_zone_ids = {d["track_id"] for d in in_zone}

        # Accumulate pre-zone position history for entities currently OUTSIDE the zone
        for detection in all_detections:
            tid = detection["track_id"]
            if tid not in in_zone_ids and tid not in self._direction_fired:
                cx, cy = self._centroid(detection["bbox"])
                if tid not in self._pre_zone_positions:
                    self._pre_zone_positions[tid] = []
                self._pre_zone_positions[tid].append((frame_number, cx, cy))
                # Keep only last 5 positions to bound memory
                self._pre_zone_positions[tid] = self._pre_zone_positions[tid][-5:]

        events = []
        for detection in in_zone:
            tid = detection["track_id"]

            if tid in self._direction_fired:
                continue  # already fired direction event for this track

            # Only fire on actual outside→inside transition.
            # If _prev_in_zone[tid] is True, the entity was already in-zone last frame — no transition.
            was_in_zone = self._prev_in_zone.get(tid, False)
            if was_in_zone:
                continue  # entity was already inside — not an entry event

            # This is the first frame this track enters the zone (outside→inside transition)
            positions = self._pre_zone_positions.get(tid, [])
            entry_dir = self._compute_entry_direction(positions)

            if entry_dir is None:
                logger.debug(f"DIRECTION: insufficient position history for track {tid}")
                # Still update _prev_in_zone below — don't skip the state update
            elif entry_dir == configured_dir:
                events.append(self._make_event(
                    frame_number=frame_number,
                    track_id=tid,
                    trigger_params={"direction": configured_dir, "computed_direction": entry_dir},
                ))
                self._direction_fired.add(tid)
                logger.debug(f"DIRECTION fired for track {tid}: entry={entry_dir}=={configured_dir}")
            else:
                logger.debug(f"DIRECTION no match for track {tid}: entry={entry_dir}!={configured_dir}")

        # Update _prev_in_zone for all seen tracks
        for tid in seen_ids:
            self._prev_in_zone[tid] = tid in in_zone_ids

        # Remove lost tracks from _prev_in_zone (treat as exited)
        for tid in list(self._prev_in_zone.keys()):
            if tid not in seen_ids:
                del self._prev_in_zone[tid]
                self._pre_zone_positions.pop(tid, None)
                logger.debug(f"DIRECTION: track {tid} lost — cleared state")

        return events

    # ------------------------------------------------------------------
    # Event factory
    # ------------------------------------------------------------------

    def _make_event(self, frame_number: int, track_id: int, trigger_params: dict) -> dict | None:
        """
        Build a D-15 compliant event dict, or None if duplicate.

        timestamp_s uses source_fps for correct wall-clock time:
            frame_number=50, source_fps=25 → timestamp_s=2.0s

        Returns None if event is a duplicate within the 2-second window (D-06/D-07).
        Caller filters None returns.
        """
        event = {
            "event_id": str(uuid.uuid4()),
            "frame_number": frame_number,
            "timestamp_s": round(frame_number / self.source_fps, 3),
            "track_id": track_id,
            "class_name": self.entity_class,
            "trigger_type": self.trigger["type"],
            "trigger_params": trigger_params,
        }

        # NEW: Deduplication check (D-06/D-07)
        if self._should_suppress_duplicate(track_id, event["trigger_type"], event["timestamp_s"]):
            logger.debug(f"Suppressed duplicate: track {track_id}, {event['trigger_type']} at {event['timestamp_s']}s")
            return None  # Caller filters None returns

        # Per-trigger-type cooldown: suppress repeats regardless of track_id so a
        # single entity whose track id keeps switching doesn't spam alerts.
        ttype = event["trigger_type"]
        last_type_fire = self._last_type_fire_s.get(ttype)
        if last_type_fire is not None and (event["timestamp_s"] - last_type_fire) < self._TRIGGER_COOLDOWN_S:
            logger.debug(f"Suppressed by cooldown: {ttype} at {event['timestamp_s']}s")
            return None

        # Record event in dedup memory
        self._record_event(track_id, event["trigger_type"], event["timestamp_s"], event["event_id"])
        self._last_type_fire_s[ttype] = event["timestamp_s"]
        return event
