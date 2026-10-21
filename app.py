"""Gradio web app: upload a road photo -> annotated image, detection table, severity rating."""
import os
from pathlib import Path

import gradio as gr
import numpy as np

from src.detector import PotholeDetector

WEIGHTS = os.environ.get("WEIGHTS", "models/best.pt")
if not Path(WEIGHTS).exists():
    raise FileNotFoundError(
        f"{WEIGHTS} not found. Train first (python scripts/train.py) or place best.pt in models/."
    )
detector = PotholeDetector(WEIGHTS)

COLORS = {"NONE": "#2e7d32", "LOW": "#9e9d24", "MEDIUM": "#ef6c00",
          "HIGH": "#d84315", "CRITICAL": "#b71c1c"}


def detect(image: np.ndarray, conf: float, mode: str):
    if image is None:
        return None, [], "Upload an image first."
    r = detector.predict(image, conf=conf, mode="sum" if mode.startswith("Sum") else "union")
    rows = [[i + 1, d["class"], round(d["confidence"], 3), str(d["box"])]
            for i, d in enumerate(r.detections)]
    summary = (
        f"<div style='padding:14px;border-radius:10px;background:{COLORS[r.severity]};color:white'>"
        f"<div style='font-size:13px;opacity:.85'>Severity</div>"
        f"<div style='font-size:28px;font-weight:700'>{r.severity}</div>"
        f"<div>{len(r.detections)} defect(s) &middot; {r.coverage_pct:.2f}% of image covered</div></div>"
    )
    return r.annotated_rgb, rows, summary


with gr.Blocks(title="Pothole & Defect Detector") as demo:
    gr.Markdown(
        "# Pothole & Defect Detector\n"
        "Upload a road photo. The model draws boxes around potholes, scores each detection, "
        "and rates overall severity from the share of the image the defects cover.\n\n"
        "| Coverage | Severity |\n|---|---|\n| none | NONE |\n| 0-3% | LOW |\n"
        "| 3-10% | MEDIUM |\n| 10-20% | HIGH |\n| >20% | CRITICAL |"
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Road photo")
            conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
            mode = gr.Radio(["Union (overlap-safe)", "Sum (assignment formula)"],
                            value="Union (overlap-safe)", label="Coverage calculation")
            btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Annotated result")
            sev = gr.HTML()
    table = gr.Dataframe(headers=["#", "Class", "Confidence", "Box (x1,y1,x2,y2)"],
                         label="Detections")
    btn.click(detect, [inp, conf, mode], [out_img, table, sev])

    examples = sorted(str(p) for p in Path("examples").glob("input_*.jpg"))
    if examples:
        gr.Examples(examples=examples, inputs=inp)

if __name__ == "__main__":
    demo.launch()
