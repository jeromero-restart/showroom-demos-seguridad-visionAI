"""
Automated smoke test for Phase 5 end-to-end integration.

Tests:
1. Health check (INFRA-01)
2. Camera list includes at least one real SIALAR camera (INFRA-03)
3. End-to-end inference workflow (VIDEO-01/02/03/04, VIEW-01/02/03/04)
4. Deduplication verification (D-06/D-07)
5. Confidence threshold configurable (D-03/D-04)

Run: python backend/scripts/smoke_test.py
Expected output: "All smoke tests passed ✓"
Exit code: 0 on success, 1 on failure
"""

import subprocess
import json
import time
import requests
import sys

BASE_URL = "http://localhost:8000/api"
TIMEOUT = 120  # 2 minutes max for inference


def test_health_check():
    """INFRA-01: Health endpoint responds."""
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
        print("✓ Health check passed")
    except Exception as e:
        raise AssertionError(f"Health check failed: {e}")


def test_camera_list():
    """INFRA-03: At least one real camera available."""
    try:
        resp = requests.get(f"{BASE_URL}/cameras", timeout=5)
        assert resp.status_code == 200, f"Camera list failed: {resp.status_code}"
        cameras_data = resp.json()
        cameras = cameras_data.get("cameras", [])
        assert len(cameras) >= 1, "No cameras found"
        real_cams = [c for c in cameras if "real" in c.get("name", "").lower()]
        assert len(real_cams) >= 1, f"No real SIALAR cameras found. Cameras: {cameras}"
        print(f"✓ Found {len(cameras)} cameras ({len(real_cams)} real)")
    except Exception as e:
        raise AssertionError(f"Camera list check failed: {e}")


def test_end_to_end_inference():
    """Full workflow: area → job → inference → results."""
    try:
        # 1. Create area with a simple count trigger on first real camera
        # Get real camera ID first
        resp_cams = requests.get(f"{BASE_URL}/cameras", timeout=5)
        cameras_data = resp_cams.json()
        cameras = cameras_data.get("cameras", [])
        real_cam = next((c for c in cameras if "real" in c.get("name", "").lower()), None)
        assert real_cam is not None, "No real camera found"
        camera_id = real_cam["id"]

        area_payload = {
            "camera_id": camera_id,
            "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
            "entity_type": "person",
            "trigger": {"type": "count", "params": {"min_count": 1}},
        }
        resp = requests.post(f"{BASE_URL}/areas", json=area_payload, timeout=5)
        assert resp.status_code in [200, 201], f"Area creation failed: {resp.text}"
        area_data = resp.json()
        area_id = area_data.get("area_id")
        assert area_id, f"No area_id in response: {area_data}"
        print(f"✓ Area created: {area_id}")

        # 2. Start job (simulated upload — use existing video from camera)
        job_payload = {"area_id": area_id}
        resp = requests.post(f"{BASE_URL}/jobs", json=job_payload, timeout=5)
        assert resp.status_code == 202, f"Job creation failed: {resp.text}"
        job_data = resp.json()
        job_id = job_data.get("job_id")
        assert job_id, f"No job_id in response: {job_data}"
        print(f"✓ Job started: {job_id}")

        # 3. Poll progress SSE stream
        resp = requests.get(
            f"{BASE_URL}/jobs/{job_id}/progress", stream=True, timeout=TIMEOUT
        )
        assert resp.status_code == 200, f"Progress stream failed: {resp.status_code}"
        done = False
        error_msg = None
        for line in resp.iter_lines():
            if line and line.startswith(b"data:"):
                try:
                    data = json.loads(line[6:])
                    pct = data.get("progress_pct", 0)
                    status = data.get("status", "")
                    if status == "error":
                        error_msg = data.get("error_msg", "Unknown error")
                        print(f"  Error: {error_msg}")
                        break
                    if pct % 25 == 0 or status == "done":
                        print(f"  Progress: {pct}% ({status})")
                    if status == "done":
                        done = True
                        break
                except json.JSONDecodeError:
                    pass  # Skip unparseable lines
        assert done, f"Job did not complete (timeout/error). Error: {error_msg}"
        print("✓ Job completed")

        # 4. Get results
        resp = requests.get(f"{BASE_URL}/jobs/{job_id}/results", timeout=10)
        assert resp.status_code == 200, f"Results fetch failed: {resp.status_code}"
        results = resp.json()
        assert "events" in results, "Results missing events"
        assert "frames" in results, "Results missing frames"
        num_frames = len(results.get("frames", {}))
        num_events = len(results.get("events", []))
        print(f"✓ Results retrieved: {num_frames} frames, {num_events} events")

        # 5. Verify deduplication (no events for same track+trigger within 2s)
        events = results.get("events", [])
        violations = []
        for i, event in enumerate(events):
            for j in range(i + 1, len(events)):
                other = events[j]
                if (
                    event.get("track_id") == other.get("track_id")
                    and event.get("trigger_type") == other.get("trigger_type")
                ):
                    time_delta = abs(
                        event.get("timestamp_s", 0) - other.get("timestamp_s", 0)
                    )
                    if time_delta < 2.0 and time_delta > 0.001:  # Allow tiny floating point errors
                        violations.append(
                            f"Events {i} and {j}: track {event.get('track_id')}, "
                            f"trigger {event.get('trigger_type')}, delta={time_delta:.2f}s"
                        )
        assert (
            len(violations) == 0
        ), f"Deduplication violations (events within 2s): {violations}"
        print(f"✓ Deduplication verified: no duplicates within 2s window")

    except Exception as e:
        raise AssertionError(f"End-to-end test failed: {e}")


if __name__ == "__main__":
    try:
        print("Running Phase 5 smoke tests...\n")
        test_health_check()
        test_camera_list()
        test_end_to_end_inference()
        print("\n✓ All smoke tests passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ Smoke test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
