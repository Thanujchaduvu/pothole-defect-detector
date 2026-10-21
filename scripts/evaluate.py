"""Evaluate trained weights on the val or test split and print/save metrics.

Usage:
    python scripts/evaluate.py --split test
"""
import argparse
import json

from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--weights", default="models/best.pt")
p.add_argument("--data", default="datasets/pothole/data.yaml")
p.add_argument("--split", default="val", choices=["val", "test"])
a = p.parse_args()

m = YOLO(a.weights).val(data=a.data, split=a.split, plots=True, project="runs", name=f"eval_{a.split}")
out = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
       "precision": float(m.box.mp), "recall": float(m.box.mr)}
print(json.dumps(out, indent=2))
with open(f"runs/metrics_{a.split}.json", "w") as f:
    json.dump(out, f, indent=2)
