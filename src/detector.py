"""Thin wrapper around a YOLOv8 model that returns annotated image + detections + severity."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .severity import score


@dataclass
class DetectionResult:
    annotated_rgb: np.ndarray
    detections: list  # [{"box": (x1,y1,x2,y2), "confidence": float, "class": str}]
    coverage_pct: float
    severity: str


class PotholeDetector:
    def __init__(self, weights: str | Path = "models/best.pt"):
        self.model = YOLO(str(weights))

    def predict(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.5,
                mode: str = "union") -> DetectionResult:
        """`image` is an RGB numpy array (as provided by Gradio)."""
        res = self.model.predict(image[..., ::-1], conf=conf, iou=iou, verbose=False)[0]
        h, w = res.orig_shape
        boxes, dets = [], []
        for b in res.boxes:
            xyxy = [float(v) for v in b.xyxy[0].tolist()]
            boxes.append(xyxy)
            dets.append({
                "box": tuple(int(v) for v in xyxy),
                "confidence": float(b.conf[0]),
                "class": res.names[int(b.cls[0])],
            })
        s = score(boxes, w, h, mode=mode)
        annotated_rgb = res.plot()[..., ::-1]  # BGR -> RGB
        return DetectionResult(annotated_rgb, dets, s["coverage_pct"], s["severity"])
