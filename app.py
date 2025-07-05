import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Setup layout
st.set_page_config(page_title="Durian Maturity Detector", page_icon="🥥", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("xcep_test.h5")

model = load_model()
class_names = ["Mature", "Young"]  # Adjust based on your training

# Preprocess image
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # drop alpha
    return np.expand_dims(img_array, axis=0)

# Predict
def predict(image):
    input_tensor = preprocess_image(image)
    prediction = model.predict(input_tensor)
    result = class_names[np.argmax(prediction)]
    return result

# UI
st.markdown("<h1 style='text-align: center;'>🥥 Durian Maturity Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Take a photo or upload one to predict if your durian is ready! 🍈</p>", unsafe_allow_html=True)

# Create tabs for Camera and Upload
tab1, tab2 = st.tabs(["📸 Camera", "🖼️ Upload Image"])

image = None  # Initialize

# Camera tab
with tab1:
    image_data = st.camera_input("Capture a durian image")
    if image_data:
        image = Image.open(image_data)

# Upload tab
with tab2:
    uploaded_file = st.file_uploader("Upload a durian image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

# If an image was provided (camera or upload)
if image:
    st.image(image, caption="📷 Selected durian image", use_container_width=True)
    if st.button("🔍 Predict Maturity"):
        result = predict(image)
        st.success(f"✅ **Prediction:** {result}")
