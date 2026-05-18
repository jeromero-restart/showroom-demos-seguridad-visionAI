import os
import sys

from shapely.geometry import Polygon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.processing.detector import EntityTrackPostProcessor, extract_detections_from_result


class _FakeTensor:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeBoxes:
    def __init__(self, xyxyn, cls, conf, ids):
        self.xyxyn = _FakeTensor(xyxyn)
        self.cls = _FakeTensor(cls)
        self.conf = _FakeTensor(conf)
        self.id = _FakeTensor(ids)

    def __len__(self):
        return len(self.cls.tolist())


class _FakeResult:
    def __init__(self, xyxyn, cls=None, conf=None, ids=None, names=None):
        self.boxes = _FakeBoxes(
            xyxyn=xyxyn,
            cls=cls or [0],
            conf=conf or [0.9],
            ids=ids or [1],
        )
        self.names = names or {0: "person"}


def test_extract_detections_marks_in_zone_when_half_bottom_edge_is_inside():
    zone_polygon = Polygon([(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)])
    result = _FakeResult([[0.4, 0.2, 0.6, 0.8]])

    detections = extract_detections_from_result(result, zone_polygon)

    assert len(detections) == 1
    assert detections[0]["in_zone"] is True


def test_extract_detections_marks_out_of_zone_when_less_than_half_bottom_edge_is_inside():
    zone_polygon = Polygon([(0.0, 0.0), (0.44, 0.0), (0.44, 1.0), (0.0, 1.0)])
    result = _FakeResult([[0.4, 0.2, 0.6, 0.8]])

    detections = extract_detections_from_result(result, zone_polygon)

    assert len(detections) == 1
    assert detections[0]["in_zone"] is False


def test_vehicle_post_processor_keeps_vehicle_track_after_person_flip():
    post_processor = EntityTrackPostProcessor("vehicle", promotion_threshold=0.5)

    first_frame = [
        {
            "track_id": 7,
            "class_name": "motorcycle",
            "confidence": 0.81,
            "bbox": [0.4, 0.2, 0.6, 0.8],
            "in_zone": True,
        }
    ]
    second_frame = [
        {
            "track_id": 7,
            "class_name": "person",
            "confidence": 0.76,
            "bbox": [0.41, 0.2, 0.61, 0.8],
            "in_zone": True,
        }
    ]

    assert post_processor.process(first_frame)[0]["class_name"] == "motorcycle"
    promoted = post_processor.process(second_frame)
    assert len(promoted) == 1
    assert promoted[0]["class_name"] == "vehicle"


def test_vehicle_post_processor_does_not_promote_person_without_prior_vehicle_evidence():
    post_processor = EntityTrackPostProcessor("vehicle", promotion_threshold=0.5)

    detections = [
        {
            "track_id": 9,
            "class_name": "person",
            "confidence": 0.91,
            "bbox": [0.4, 0.2, 0.6, 0.8],
            "in_zone": True,
        }
    ]

    assert post_processor.process(detections) == []
