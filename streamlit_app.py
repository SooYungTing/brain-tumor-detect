"""
    Brain-Tumor MRI Classifier - Streamlit Front-end
    Model: brain_tumor.h5
    $ streamlit run streamlit_app.py
"""

import os, cv2, numpy as np, tensorflow as tf, streamlit as st
from PIL import Image

# constants
MODEL_PATH = "brain_tumor.h5"
IMG_SIZE   = 224
CLASSES    = ["pituitary", "notumor", "meningioma", "glioma"]

# load model
@st.cache_resource(show_spinner=True)
def load_model():
    if not os.path.isfile(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found.")
        st.stop()
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
        with st.spinner("Analysing..."):
            x = preprocess(pil_image)
            probs = model.predict(x)[0]
            label_idx = int(np.argmax(probs))
            label = CLASSES[label_idx]
            confidence = float(probs[label_idx])

        st.success("Done!")
        st.metric("Predicted class", label.upper())
        st.metric("Confidence", f"{confidence:.2%}")
        st.bar_chart({c: float(p) for c, p in zip(CLASSES, probs)})