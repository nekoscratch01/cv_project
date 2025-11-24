import sys
import types
import numpy as np


# Stub heavy deps so unit tests can import dataclasses without GPU/GPU models.
class _DummyTracker:
    def update(self, detections, frame=None):
        return np.empty((0, 5))


class _DummyYOLO:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        return [types.SimpleNamespace(boxes=None)]


if "boxmot" not in sys.modules:
    sys.modules["boxmot"] = types.SimpleNamespace(create_tracker=lambda *args, **kwargs: _DummyTracker())

if "ultralytics" not in sys.modules:
    sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=_DummyYOLO)

if "transformers" not in sys.modules:
    sys.modules["transformers"] = types.SimpleNamespace(SiglipModel=object, SiglipProcessor=object)
