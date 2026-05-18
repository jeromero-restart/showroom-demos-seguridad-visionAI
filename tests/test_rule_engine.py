"""
Unit tests for RuleEngine — Phase 2 Success Criterion 2.

Tests run with: pytest tests/test_rule_engine.py -v
(from backend/ directory, or inside the Docker container)

All tests use mocked detection sequences — no real video or YOLOv8 needed.

Frame number semantics used in tests:
    source_fps=25, processed at 5fps → frame_interval=5.
    Processed frames are at actual video frame numbers: 0, 5, 10, 15, 20, 25, 30...
    1 second = 25 source frames = 5 processed frames.
    Dwell threshold_s=2.0 → fires when (frame_number - entry_frame) / 25 >= 2.0
                            → fires when frame_number - entry_frame >= 50
                            → fires at frame 50 if entry at frame 0.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.processing.rule_engine import RuleEngine

SOURCE_FPS = 25.0    # source video fps
FRAME_INTERVAL = 5   # vid_stride (source_fps / processed_fps = 25/5 = 5)


def make_detection(track_id: int, cx: float, cy: float, in_zone: bool) -> dict:
    """Helper: build a minimal detection dict for testing."""
    half = 0.05
    return {
        "track_id": track_id,
        "class_name": "person",
        "confidence": 0.85,
        "bbox": [cx - half, cy - half, cx + half, cy + half],
        "in_zone": in_zone,
    }


def frame(n: int) -> int:
    """Convert processed-frame index n to actual video frame number. frame(0)=0, frame(1)=5, etc."""
    return n * FRAME_INTERVAL


# ---------------------------------------------------------------------------
# COUNT trigger tests (D-01)
# ---------------------------------------------------------------------------

class TestCountTrigger:
    def setup_method(self):
        self.engine = RuleEngine(
            trigger_config={"type": "count", "params": {"min_count": 2}},
            source_fps=SOURCE_FPS,
        )

    def test_count_fires_when_threshold_reached(self):
        """COUNT fires after 2 consecutive frames at or above threshold (false-positive filter)."""
        detections = [
            make_detection(1, 0.5, 0.5, in_zone=True),
            make_detection(2, 0.5, 0.6, in_zone=True),
        ]
        events_f0 = self.engine.evaluate(frame_number=frame(0), detections=detections)
        assert len(events_f0) == 0  # first frame: not yet stable
        events_f1 = self.engine.evaluate(frame_number=frame(1), detections=detections)
        assert len(events_f1) == 1
        assert events_f1[0]["trigger_type"] == "count"
        assert events_f1[0]["trigger_params"]["actual_count"] == 2

    def test_count_no_fire_below_threshold(self):
        """1 entity in zone, threshold=2: no event."""
        detections = [make_detection(1, 0.5, 0.5, in_zone=True)]
        events = self.engine.evaluate(frame_number=frame(0), detections=detections)
        assert len(events) == 0

    def test_count_no_double_fire_while_sustained(self):
        """Count stays >= threshold across frames: fires only once (after 2-frame stability)."""
        detections = [
            make_detection(1, 0.5, 0.5, in_zone=True),
            make_detection(2, 0.5, 0.6, in_zone=True),
        ]
        events_f0 = self.engine.evaluate(frame_number=frame(0), detections=detections)
        events_f1 = self.engine.evaluate(frame_number=frame(1), detections=detections)
        events_f2 = self.engine.evaluate(frame_number=frame(2), detections=detections)
        assert len(events_f0) == 0  # frame 1 of 2 — not yet stable
        assert len(events_f1) == 1  # frame 2 of 2 — fires
        assert len(events_f2) == 0  # no second fire

    def test_count_rearms_after_drop_below_threshold(self):
        """Count drops below threshold then rises again: fires a second time after 2-frame stability."""
        detections_2 = [
            make_detection(1, 0.5, 0.5, in_zone=True),
            make_detection(2, 0.5, 0.6, in_zone=True),
        ]
        # First cycle: needs 2 stable frames to fire
        events_f0 = self.engine.evaluate(frame_number=frame(0), detections=detections_2)
        assert len(events_f0) == 0
        events_f1 = self.engine.evaluate(frame_number=frame(1), detections=detections_2)
        assert len(events_f1) == 1  # fires

        # Drop below threshold (re-arm)
        detections_1 = [make_detection(1, 0.5, 0.5, in_zone=True)]
        events_f2 = self.engine.evaluate(frame_number=frame(2), detections=detections_1)
        assert len(events_f2) == 0  # re-arms, doesn't fire

        # Rise again — needs 2 stable frames again
        events_f3 = self.engine.evaluate(frame_number=frame(3), detections=detections_2)
        assert len(events_f3) == 0
        events_f4 = self.engine.evaluate(frame_number=frame(4), detections=detections_2)
        assert len(events_f4) == 1  # fires again


# ---------------------------------------------------------------------------
# DWELL trigger tests (D-02)
# ---------------------------------------------------------------------------

class TestDwellTrigger:
    def setup_method(self):
        # threshold_s=2.0, source_fps=25 → fires when (frame_number - entry_frame) / 25 >= 2.0
        # → fires when delta >= 50 actual frames = 10 processed frames at 5fps
        self.engine = RuleEngine(
            trigger_config={"type": "dwell", "params": {"threshold_s": 2.0}},
            source_fps=SOURCE_FPS,
        )

    def _dwell_frames(self, track_id: int, start_processed_idx: int, count: int) -> list[dict]:
        """Feed count in-zone frames for track_id, starting at processed index start_processed_idx."""
        all_events = []
        for i in range(count):
            fn = frame(start_processed_idx + i)  # actual video frame number
            detections = [make_detection(track_id, 0.5, 0.5, in_zone=True)]
            events = self.engine.evaluate(frame_number=fn, detections=detections)
            all_events.extend(events)
        return all_events

    def test_dwell_fires_after_threshold(self):
        """
        Entity dwells 3 seconds (15 processed frames at 5fps = actual frames 0,5,...,70):
        threshold=2s → fires when frame_delta / 25 >= 2.0 → delta >= 50 → frame 50 (processed idx 10).
        """
        events = self._dwell_frames(track_id=1, start_processed_idx=0, count=15)
        assert len(events) == 1, f"Expected 1 event, got {len(events)}"
        assert events[0]["trigger_type"] == "dwell"
        assert events[0]["track_id"] == 1
        # Event fires at frame 50 (processed index 10): timestamp = 50/25 = 2.0s
        assert events[0]["frame_number"] == frame(10)
        assert abs(events[0]["timestamp_s"] - 2.0) < 0.01

    def test_dwell_no_fire_before_threshold(self):
        """Entity dwells 1.8 seconds (9 processed frames = frame 0 to frame 40):
        delta=40, 40/25=1.6s < 2.0s threshold → no fire."""
        events = self._dwell_frames(track_id=1, start_processed_idx=0, count=9)
        assert len(events) == 0

    def test_dwell_no_double_fire_sustained(self):
        """Entity dwells 5 seconds (25 processed frames): fires exactly once."""
        events = self._dwell_frames(track_id=1, start_processed_idx=0, count=25)
        assert len(events) == 1

    def test_dwell_rearms_after_exit_and_reentry(self):
        """Entity dwells (fires), exits, re-enters and dwells again: fires again."""
        # First dwell — fires at frame 50 (processed idx 10)
        events1 = self._dwell_frames(track_id=1, start_processed_idx=0, count=15)
        assert len(events1) == 1

        # Exit zone (entity out-of-zone for a frame)
        exit_fn = frame(16)
        exit_detections = [make_detection(1, 0.5, 0.5, in_zone=False)]
        self.engine.evaluate(frame_number=exit_fn, detections=exit_detections)

        # Re-enter and dwell again — should fire again (re-armed on exit)
        events2 = self._dwell_frames(track_id=1, start_processed_idx=17, count=15)
        assert len(events2) == 1


# ---------------------------------------------------------------------------
# DIRECTION trigger tests (D-03/D-04)
# ---------------------------------------------------------------------------

class TestDirectionTrigger:
    def _make_engine(self, direction: str) -> RuleEngine:
        return RuleEngine(
            trigger_config={"type": "direction", "params": {"direction": direction}},
            source_fps=SOURCE_FPS,
        )

    def _simulate_approach_and_enter(
        self,
        engine: RuleEngine,
        pre_zone_positions: list[tuple[float, float]],
        entry_cx: float,
        entry_cy: float,
    ) -> list[dict]:
        """Feed entity approaching from outside zone, then entering."""
        all_events = []
        # Pre-zone frames (in_zone=False) — builds _pre_zone_positions
        for i, (cx, cy) in enumerate(pre_zone_positions):
            detections = [make_detection(1, cx, cy, in_zone=False)]
            events = engine.evaluate(frame_number=frame(i), detections=detections)
            all_events.extend(events)
        # Entry frame (in_zone=True) — actual outside→inside transition
        entry_fn = frame(len(pre_zone_positions))
        detections = [make_detection(1, entry_cx, entry_cy, in_zone=True)]
        events = engine.evaluate(frame_number=entry_fn, detections=detections)
        all_events.extend(events)
        return all_events

    def test_direction_fires_on_east_entry(self):
        """Entity moving left→right (dx>0) enters zone: configured E → fires."""
        engine = self._make_engine("E")
        pre_zone = [(0.1, 0.5), (0.2, 0.5), (0.3, 0.5), (0.4, 0.5)]
        events = self._simulate_approach_and_enter(engine, pre_zone, entry_cx=0.5, entry_cy=0.5)
        assert len(events) == 1
        assert events[0]["trigger_type"] == "direction"
        assert events[0]["trigger_params"]["computed_direction"] == "E"

    def test_direction_no_fire_wrong_direction(self):
        """Entity moving left→right but configured W → does NOT fire."""
        engine = self._make_engine("W")
        pre_zone = [(0.1, 0.5), (0.2, 0.5), (0.3, 0.5), (0.4, 0.5)]
        events = self._simulate_approach_and_enter(engine, pre_zone, entry_cx=0.5, entry_cy=0.5)
        assert len(events) == 0

    def test_direction_north_entry(self):
        """Entity moving top→bottom (dy>0) enters zone: configured S → fires."""
        engine = self._make_engine("S")
        pre_zone = [(0.5, 0.1), (0.5, 0.2), (0.5, 0.3), (0.5, 0.4)]
        events = self._simulate_approach_and_enter(engine, pre_zone, entry_cx=0.5, entry_cy=0.5)
        assert len(events) == 1
        assert events[0]["trigger_params"]["computed_direction"] == "S"

    def test_direction_velocity_averaged_not_single_frame(self):
        """Last frame shows jitter (slightly west) but overall average is east → fires E."""
        engine = self._make_engine("E")
        # 3 frames strongly east, 1 frame slight jitter west, net average still strongly east
        pre_zone = [(0.1, 0.5), (0.2, 0.5), (0.3, 0.5), (0.29, 0.5)]
        events = self._simulate_approach_and_enter(engine, pre_zone, entry_cx=0.5, entry_cy=0.5)
        assert len(events) == 1  # average wins over jitter

    def test_direction_no_fire_on_sustained_in_zone(self):
        """
        Entity already in zone stays in zone across frames: only fires on first entry,
        NOT on subsequent in-zone frames (tests _prev_in_zone transition guard).
        """
        engine = self._make_engine("E")
        pre_zone = [(0.1, 0.5), (0.2, 0.5), (0.3, 0.5)]
        # Entry frame
        all_events = self._simulate_approach_and_enter(engine, pre_zone, entry_cx=0.5, entry_cy=0.5)
        first_count = len(all_events)
        assert first_count == 1

        # Subsequent in-zone frames — must NOT fire again
        for i in range(5):
            fn = frame(len(pre_zone) + 1 + i)
            detections = [make_detection(1, 0.5, 0.5, in_zone=True)]
            events = engine.evaluate(frame_number=fn, detections=detections)
            assert len(events) == 0, f"Direction fired again at frame {fn} (should fire only on entry)"

    def test_direction_no_fire_insufficient_history(self):
        """
        Entity enters zone with only 1 pre-zone position (insufficient for velocity):
        no direction event fired (None from _compute_entry_direction).
        """
        engine = self._make_engine("E")
        # Only 1 pre-zone position — not enough for velocity average (need >= 2)
        pre_zone = [(0.4, 0.5)]
        events = self._simulate_approach_and_enter(engine, pre_zone, entry_cx=0.5, entry_cy=0.5)
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Event schema validation
# ---------------------------------------------------------------------------

class TestEventSchema:
    def test_event_has_all_d15_fields(self):
        """Every generated event must have all D-15 required fields."""
        engine = RuleEngine(
            trigger_config={"type": "count", "params": {"min_count": 1}},
            source_fps=SOURCE_FPS,
            area_id="test-area-id",
            entity_class="person",
        )
        detections = [make_detection(1, 0.5, 0.5, in_zone=True)]
        engine.evaluate(frame_number=frame(1), detections=detections)  # first stable frame
        events = engine.evaluate(frame_number=frame(2), detections=detections)
        assert len(events) == 1
        event = events[0]
        required_fields = {
            "event_id", "frame_number", "timestamp_s", "track_id",
            "class_name", "trigger_type", "trigger_params",
        }
        missing = required_fields - set(event.keys())
        assert not missing, f"Event missing D-15 fields: {missing}"
        assert isinstance(event["event_id"], str)
        # frame(2) = actual frame 10; timestamp = 10 / 25 = 0.4s
        assert event["timestamp_s"] == pytest.approx(frame(2) / SOURCE_FPS, abs=0.001)
        assert event["frame_number"] == frame(2)

    def test_timestamp_uses_source_fps_not_processed_fps(self):
        """timestamp_s = frame_number / source_fps (NOT frame_number / processed_fps)."""
        engine = RuleEngine(
            trigger_config={"type": "count", "params": {"min_count": 1}},
            source_fps=SOURCE_FPS,
        )
        detections = [make_detection(1, 0.5, 0.5, in_zone=True)]
        fn = frame(10)  # actual frame 50 (10 * 5)
        engine.evaluate(frame_number=frame(9), detections=detections)  # first stable frame
        events = engine.evaluate(frame_number=fn, detections=detections)
        assert len(events) == 1
        # Correct: 50 / 25 = 2.0s
        # Wrong:   50 / 5  = 10.0s  (would happen if processed_fps was used)
        assert events[0]["timestamp_s"] == pytest.approx(2.0, abs=0.001)
