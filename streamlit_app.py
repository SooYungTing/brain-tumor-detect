"""
    Brain Tumor MRI Classifier - Streamlit Front-end
    Model: brain_tumor.h5
    $ streamlit run streamlit_app.py
"""

import os, cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# constants
MODEL_URL = st.secrets["MODEL_URL"]
MODEL_PATH = "brain_tumor.h5"
IMG_SIZE = 224
CLASSES = ["pituitary", "notumor", "meningioma", "glioma"]

@st.cache_resource(show_spinner=True)
def load_model():
    if not os.path.isfile(MODEL_PATH):
        with st.spinner("Downloading model …"):
            import requests, io
            r = requests.get(MODEL_URL, timeout=60)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# preprocess image
def preprocess(pil_image):
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# UI
st.set_page_config(page_title="🧠 Brain-Tumor MRI Classifier", layout="centered")
st.title("🧠 Brain-Tumor MRI Classification")
st.markdown("Upload an axial T1-weighted MRI slice and the model will classify it")

uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded is not None:
    pil_image = Image.open(uploaded)
    st.image(pil_image, caption="Uploaded MRI", use_column_width=True)
    if st.button("🔍 Classify"):
        with st.spinner("Analysing ..."):
            x = preprocess(pil_image)
            probs = model.predict(x)[0]
            label_idx = int(np.argmax(probs))
            label = CLASSES[label_idx]
            confidence = float(probs[label_idx])

        st.success("Done!!")
        st.metric("Predicted class", label.upper(), delta=None)
        st.metric("Confidence", f"{confidence:.2%}", delta=None)

        # bar chart
        st.subheader("Class probabilities")
        st.bar_chart({c: float(p) for c, p in zip(CLASSES, probs)})