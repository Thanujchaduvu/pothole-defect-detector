
import os
import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path

from src.detector import PotholeDetector

st.set_page_config(
    page_title="AI Pothole Detection System",
    page_icon="🚧",
    layout="wide"
)

st.title("🚧 AI Pothole Detection System")
st.write("Upload a road image to detect potholes and assess road damage.")

WEIGHTS = "models/best.pt"


@st.cache_resource
def load_detector():
    if not Path(WEIGHTS).exists():
        raise FileNotFoundError(f"Model not found: {WEIGHTS}")
    return PotholeDetector(WEIGHTS)


uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    st.subheader("Uploaded Road Image")
    st.image(image_array, width="stretch")

    if st.button("Detect potholes", type="primary"):
        try:
            detector = load_detector()

            with st.spinner("Analyzing road image..."):
                # Use the actual predict() method from detector.py
                result = detector.predict(image_array)

            st.success("Detection completed successfully!")

            st.subheader("Detection Results")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    result.annotated_rgb,
                    caption="Detected potholes",
                    width="stretch"
                )

            with col2:
                st.metric(
                    "Potholes Detected",
                    len(result.detections)
                )
                st.metric(
                    "Affected Road Coverage",
                    f"{result.coverage_pct:.2f}%"
                )
                st.metric("Severity", result.severity)

            if result.detections:
                st.subheader("Individual Detections")

                rows = []
                for i, detection in enumerate(result.detections, start=1):
                    x1, y1, x2, y2 = detection["box"]
                    rows.append({
                        "ID": i,
                        "Class": detection["class"],
                        "Confidence": round(
                            detection["confidence"] * 100, 2
                        ),
                        "Bounding Box": f"({x1}, {y1}, {x2}, {y2})"
                    })

                st.dataframe(rows, use_container_width=True)
            else:
                st.info("No potholes detected in this image.")

        except Exception as e:
            st.error(f"Detection failed: {e}")

