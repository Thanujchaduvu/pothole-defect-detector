"""Run the trained model on sample images and save annotated outputs to examples/.

Usage:
    python scripts/make_examples.py path/to/img1.jpg path/to/img2.jpg path/to/img3.jpg
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detector import PotholeDetector  # noqa: E402

det = PotholeDetector("models/best.pt")
Path("examples").mkdir(exist_ok=True)
for p in sys.argv[1:]:
    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    r = det.predict(img)
    out = Path("examples") / f"result_{Path(p).stem}_{r.severity}.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(r.annotated_rgb, cv2.COLOR_RGB2BGR))
    print(f"{out}: {len(r.detections)} detections, {r.coverage_pct:.1f}% coverage, {r.severity}")
