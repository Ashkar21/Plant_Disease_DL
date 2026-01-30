import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ================= CONFIG =================
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_plant_disease_model.h5")

model = load_model()

# ================= CLASS NAMES =================
class_names = ["Healthy", "Powdery", "Rust"]

IMG_SIZE = (224, 224)

# ================= IMAGE PREPROCESS =================
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ================= UI =================
st.title("🌿 Plant Disease Detection")
st.markdown("Upload a leaf image to detect **Healthy, Powdery, or Rust disease**.")

uploaded_file = st.file_uploader(
    "📤 Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict Disease"):
        with st.spinner("Analyzing image..."):
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            confidence = np.max(predictions)
            predicted_class = class_names[np.argmax(predictions)]

        st.success(f"✅ **Prediction:** {predicted_class}")
        st.info(f"🎯 **Confidence:** {confidence * 100:.2f}%")

        # Progress bar confidence
        st.progress(int(confidence * 100))

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "👨‍💻 Developed by **Ashkar R** | Deep Learning Plant Disease Detection"
)
