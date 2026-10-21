"""Fine-tune a pretrained YOLOv8 model on the pothole dataset.

Usage:
    python scripts/train.py --data datasets/pothole/data.yaml --epochs 50
"""
import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datasets/pothole/data.yaml")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--freeze", type=int, default=0,
                   help="Freeze first N layers (10 = YOLOv8 backbone). 0 = full fine-tune.")
    p.add_argument("--name", default="pothole_v8n")
    a = p.parse_args()

    model = YOLO(a.weights)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        freeze=a.freeze or None, project="runs", name=a.name, patience=15, plots=True,
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    Path("models").mkdir(exist_ok=True)
    shutil.copy(best, "models/best.pt")
    print("Copied best weights to models/best.pt")

    m = YOLO("models/best.pt").val(data=a.data, split="val")
    results = {
        "mAP50": float(m.box.map50),
        "mAP50-95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
    }
    Path("runs/metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
