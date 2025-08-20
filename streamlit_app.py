"""
Brain-Tumor MRI Classifier - Streamlit Front-end (Google Drive model)
Run: streamlit run streamlit_app.py
"""

import os
from pathlib import Path
import hashlib
import numpy as np
import cv2
import h5py
import gdown
import tensorflow as tf
import streamlit as st
from PIL import Image

#constant
GDRIVE_FILE_ID = "1hyFu6_sTE7lKJniBRTownckrTT8OrIIf"  
EXPECTED_SHA256 = None  
CACHE_DIR = Path("models")
MODEL_PATH = CACHE_DIR / "brain_tumor.h5"

IMG_SIZE = 224
CLASSES = ["pituitary", "notumor", "meningioma", "glioma"]

tf.get_logger().setLevel("ERROR")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def is_valid_hdf5(path: Path) -> bool:
    try:
        with h5py.File(path, "r"):
            return True
    except OSError:
        return False

@st.cache_resource(show_spinner=True)
def load_model() -> tf.keras.Model:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Download from Google Drive if missing/empty
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1024:
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        tmp_path = CACHE_DIR / (MODEL_PATH.name + ".part")
        gdown.download(url, str(tmp_path), quiet=False) 

        if not tmp_path.exists() or tmp_path.stat().st_size < 1024:
            st.error("Download from Google Drive failed or returned an empty file.")
            st.stop()

        tmp_path.replace(MODEL_PATH)

    # Sanity checks: not LFS/HTML, valid HDF5
    with open(MODEL_PATH, "rb") as f:
        head = f.read(256)
    if b"git-lfs.github.com/spec/v1" in head:
        st.error("Downloaded file is a Git LFS pointer, not the actual weights.")
        st.stop()
    if b"<html" in head.lower() or b"<!doctype html" in head.lower():
        st.error("Google Drive returned an HTML page (bad/unauthorized link). "
                 "Set sharing to 'Anyone with the link'.")
        st.stop()

    if not is_valid_hdf5(MODEL_PATH):
        st.error(f"'{MODEL_PATH.name}' is not a valid HDF5 file (corrupted/truncated).")
        st.stop()

    if EXPECTED_SHA256:
        digest = sha256(MODEL_PATH)
        if digest != EXPECTED_SHA256:
            st.error("Model checksum mismatch. Refusing to load.")
            st.stop()

    # Load the model (Keras 3 via TF 2.20 supports legacy .h5 with h5py)
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)

model = load_model()

def preprocess(pil_image: Image.Image) -> np.ndarray:
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    img = np.array(pil_image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# UI
st.set_page_config(page_title="🧠 Brain-Tumor MRI Classifier", layout="centered")
st.title("🧠 Brain-Tumor MRI Classification")
st.markdown("Upload an axial T1-weighted MRI slice and the model will classify it.")

uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tif", "tiff"])

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
            probs = model.predict(x)[0]
            # If model outputs logits, uncomment
            # probs = tf.nn.softmax(probs).numpy()

            label_idx = int(np.argmax(probs))
            label = CLASSES[label_idx]
            confidence = float(probs[label_idx])

        st.success("Done!")
        st.metric("Predicted class", label.upper())
        st.metric("Confidence", f"{confidence:.2%}")
        st.bar_chart({c: float(p) for c, p in zip(CLASSES, probs)})
