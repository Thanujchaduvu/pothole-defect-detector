"""Download the pothole dataset from Roboflow Universe in YOLOv8 format.

Usage:
    export ROBOFLOW_API_KEY=your_key      # free key from roboflow.com -> Settings -> API
    python scripts/download_dataset.py
Optional env vars: RF_WORKSPACE, RF_PROJECT, RF_VERSION
"""
import os
import sys

from roboflow import Roboflow

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    sys.exit("Set ROBOFLOW_API_KEY first (never hard-code or commit it).")

workspace = os.environ.get("RF_WORKSPACE", "brad-dwyer")
project = os.environ.get("RF_PROJECT", "pothole-voxrl")
version = int(os.environ.get("RF_VERSION", "1"))

rf = Roboflow(api_key=api_key)
ds = rf.workspace(workspace).project(project).version(version).download(
    "yolov8", location="datasets/pothole"
)
print("Downloaded to:", ds.location)
print("Check datasets/pothole/data.yaml: train/val/test paths must resolve.")
