"""
Brain-Tumor MRI Classifier - Streamlit Front-end
Model: brain_tumor.h5
Run: streamlit run streamlit_app.py
"""

import os
import numpy as np
import cv2
import h5py
import tensorflow as tf
import streamlit as st
from PIL import Image

# --------------------
# Constants & settings
# --------------------
MODEL_PATH = "brain_tumor.h5"
IMG_SIZE   = 224
CLASSES    = ["pituitary", "notumor", "meningioma", "glioma"]

# Make TF logs quieter in prod
tf.get_logger().setLevel("ERROR")

# --------------------
# Helpers
# --------------------
def _is_valid_hdf5(path: str) -> bool:
    try:
        with h5py.File(path, "r"):
            return True
    except OSError:
        return False

@st.cache_resource(show_spinner=True)
def load_model():
    if not os.path.isfile(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found.")
        st.stop()

    # sanity-check the .h5 so we fail gracefully if it's corrupt/LFS/HTML
    if not _is_valid_hdf5(MODEL_PATH):
        st.error(
            f"'{MODEL_PATH}' is not a valid HDF5 file (it may be corrupted, a Git LFS pointer, or an HTML download)."
        )
        st.stop()

    # compile=False avoids needing the original training-time objects
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

def preprocess(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to normalized NCHW batch the model expects (NHWC actually)."""
    # Ensure RGB (handles grayscale or RGBA inputs robustly)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Resize with good quality
    img = np.array(pil_image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Scale to [0,1]
    img = img.astype(np.float32) / 255.0

    # Add batch dimension -> (1, H, W, C)
    return np.expand_dims(img, axis=0)

# --------------------
# UI
# --------------------
st.set_page_config(page_title="🧠 Brain-Tumor MRI Classifier", layout="centered")
st.title("🧠 Brain-Tumor MRI Classification")
st.markdown("Upload an axial T1-weighted MRI slice and the model will classify it.")

uploaded = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded is not None:
    try:
        pil_image = Image.open(uploaded)
    except Exception as e:
        st.error(f"Could not open the image: {e}")
        st.stop()

    st.image(pil_image, caption="Uploaded MRI", use_column_width=True)

    if st.button("🔍 Classify"):
        with st.spinner("Analysing..."):
            x = preprocess(pil_image)
            # Model should output probabilities; if logits, add softmax here
            probs = model.predict(x)[0]
            # if the model outputs logits, uncomment next line:
            # probs = tf.nn.softmax(probs).numpy()

            label_idx = int(np.argmax(probs))
            label = CLASSES[label_idx]
            confidence = float(probs[label_idx])

        st.success("Done!")
        st.metric("Predicted class", label.upper())
        st.metric("Confidence", f"{confidence:.2%}")

        # Display probabilities as a bar chart
        st.bar_chart({c: float(p) for c, p in zip(CLASSES, probs)})
