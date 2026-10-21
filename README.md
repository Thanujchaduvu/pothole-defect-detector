---
title: Pothole & Defect Detector
emoji: 🕳️
colorFrom: orange
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
---

# Pothole & Defect Detector

> **Status:** fill every `TODO` below with your own results before submitting. Do not leave placeholder numbers.

A YOLOv8 model fine-tuned on the Roboflow Universe pothole dataset detects potholes in road photographs, draws bounding boxes with confidence scores, and converts the total defect coverage into a severity rating (NONE / LOW / MEDIUM / HIGH / CRITICAL) that a municipal team could use to prioritise repairs. It is served through a Gradio web app deployed on Hugging Face Spaces.

**Live demo:** TODO `https://huggingface.co/spaces/<username>/<space-name>`

## Results

Evaluated on the dataset's validation/test split with `scripts/evaluate.py` (weights: `yolov8n.pt` fine-tuned, 50 epochs, imgsz 640).

| Metric | Value |
|---|---|
| mAP50 | TODO |
| mAP50-95 | TODO |
| Precision | TODO |
| Recall | TODO |

Training curves, confusion matrix and PR curve are generated in `runs/<name>/`. TODO: add 1-2 sentences on which metric is weakest and why (a typical pattern: mAP50 is much higher than mAP50-95 because box edges on irregular potholes are ambiguous).

## Example detections

| Input | Output |
|---|---|
| TODO `examples/result_1.jpg` | severity ... |
| TODO `examples/result_2.jpg` | severity ... |
| TODO `examples/result_3.jpg` | severity ... |

Generate them with `python scripts/make_examples.py img1.jpg img2.jpg img3.jpg`.

## Where the model fails

Use `python scripts/find_failures.py` to surface the worst validation images, then pick three and explain them honestly.

| Screenshot | What went wrong | Likely reason |
|---|---|---|
| TODO `failure_cases/fail_a.jpg` | | e.g. shadow / water puddle mistaken for pothole |
| TODO `failure_cases/fail_b.jpg` | | e.g. small distant pothole missed |
| TODO `failure_cases/fail_c.jpg` | | e.g. cluster of potholes merged into one box |

## Severity logic

```
coverage = area covered by detected boxes / total image area
```

| Coverage | Severity |
|---|---|
| No detection | NONE |
| 0-3% | LOW |
| 3-10% | MEDIUM |
| 10-20% | HIGH |
| >20% | CRITICAL |

The app offers two coverage modes. **Sum** is the formula from the brief (adds every box area). **Union** (default) paints all boxes onto a mask and counts filled pixels, so overlapping boxes are not double-counted. Logic lives in `src/severity.py` and is unit-tested.

## How to run

```bash
git clone https://github.com/<you>/pothole-defect-detector.git
cd pothole-defect-detector
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Get data (free API key from roboflow.com, never commit it)
export ROBOFLOW_API_KEY=your_key
python scripts/download_dataset.py

# 2. Train (GPU recommended; see colab_train.md for free Colab)
python scripts/train.py --epochs 50

# 3. Evaluate and find failure cases
python scripts/evaluate.py --split val
python scripts/find_failures.py

# 4. Launch the app
python app.py          # http://127.0.0.1:7860

# 5. Run tests
pytest
```

`models/best.pt` is created by training. Weights and datasets are git-ignored.

## Deploying to Hugging Face Spaces

1. Create a new Space, SDK **Gradio**.
2. Upload `app.py`, `src/`, `requirements.txt`, `README.md` (its YAML header configures the Space) and `models/best.pt` (use Git LFS or the web uploader for the weights).
3. The Space builds automatically and gives you the public URL; paste it above.

## Project structure

```
app.py                  Gradio UI
src/detector.py         YOLO wrapper (boxes, confidences, severity)
src/severity.py         coverage + severity logic
scripts/                download, train, evaluate, find_failures, make_examples
tests/test_severity.py  unit tests for severity bands and overlap handling
colab_train.md          free-GPU training recipe
```

## Limitations

TODO: edit to reflect what you actually observed. Likely candidates:

- **Dataset bias.** The recommended dataset is not specific to Indian roads; lighting, road texture, vehicles and signage differ, so performance on Indian street photos may be lower.
- **Coverage is a rough proxy.** Percentage of image area depends on camera height, angle, distance and zoom. A small pothole photographed up close can rate CRITICAL while a dangerous one far away rates LOW. It does not measure depth or real-world size.
- **Bounding boxes overestimate area** for irregular potholes (a box is larger than the actual hole), inflating coverage.
- **Single class.** The model does not distinguish potholes from cracks, patches or rutting.
- **Confusers.** Shadows, puddles, manhole covers and repaired patches can trigger false positives.
- **Small model.** `yolov8n` trades accuracy for speed; larger variants may improve recall on small defects.
- Not a safety-certified system; it supports, not replaces, manual inspection.
