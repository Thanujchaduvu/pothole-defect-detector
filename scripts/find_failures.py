"""Rank validation images by how badly the model does, to pick your 3 failure cases.

For each val image, compares predictions with ground-truth labels (IoU >= 0.5) and counts
missed potholes (FN) and false alarms (FP). Annotated worst cases are saved to failure_cases/.

Usage:
    python scripts/find_failures.py --images datasets/pothole/valid/images \
                                    --labels datasets/pothole/valid/labels --top 10
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(path, w, h):
    boxes = []
    if path.exists():
        for line in path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            _, xc, yc, bw, bh = map(float, parts[:5])
            boxes.append([(xc-bw/2)*w, (yc-bh/2)*h, (xc+bw/2)*w, (yc+bh/2)*h])
    return boxes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/best.pt")
    p.add_argument("--images", default="datasets/pothole/valid/images")
    p.add_argument("--labels", default="datasets/pothole/valid/labels")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--top", type=int, default=10)
    a = p.parse_args()

    model = YOLO(a.weights)
    rows = []
    for img_path in sorted(Path(a.images).glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gt = load_gt(Path(a.labels) / (img_path.stem + ".txt"), w, h)
        res = model.predict(img, conf=a.conf, verbose=False)[0]
        pred = [b.xyxy[0].tolist() for b in res.boxes]
        matched, tp = set(), 0
        for pb in pred:
            best, best_j = 0, -1
            for j, g in enumerate(gt):
                if j in matched:
                    continue
                v = iou(pb, g)
                if v > best:
                    best, best_j = v, j
            if best >= 0.5:
                matched.add(best_j)
                tp += 1
        fp, fn = len(pred) - tp, len(gt) - tp
        rows.append((fp + fn, fp, fn, img_path, res))

    rows.sort(key=lambda r: r[0], reverse=True)
    out = Path("failure_cases")
    out.mkdir(exist_ok=True)
    print(f"{'errors':>6} {'FP':>4} {'FN':>4}  image")
    for errs, fp, fn, path, res in rows[:a.top]:
        print(f"{errs:>6} {fp:>4} {fn:>4}  {path.name}")
        cv2.imwrite(str(out / f"fail_{path.stem}.jpg"), res.plot())
    print(f"Saved annotated worst cases to {out}/. Pick 3 and explain WHY in the README.")


if __name__ == "__main__":
    main()
