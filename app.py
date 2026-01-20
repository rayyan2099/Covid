import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Covid-19 X-Ray Detection", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("covid_model.keras")

model = load_model()

st.title("🩻 Covid-19 Detection from Chest X-Ray")
st.caption("This tool is for educational purposes only, not medical diagnosis.")

uploaded_file = st.file_uploader(
    "Upload a Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-Ray", use_container_width=True)

    # Preprocessing
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    st.write(f"**Model Confidence:** {confidence:.2f}")

    if confidence < 0.5:
        st.error("Prediction: COVID Positive")
    else:
        st.success("Prediction: Normal")
