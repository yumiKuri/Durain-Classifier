import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from datetime import datetime
import csv
import os

# Setup layout
st.set_page_config(page_title="Durian Maturity Detector", page_icon="🌳", layout="centered")

if "last_pred" not in st.session_state:
    # will store dict: {"result": str, "confidence": float}
    st.session_state["last_pred"] = None

if "correct_count" not in st.session_state:
    st.session_state["correct_count"] = 0

if "wrong_count" not in st.session_state:
    st.session_state["wrong_count"] = 0

#For save feedback
    
FEEDBACK_FILE = "feedback.csv"

def save_feedback_to_csv(predicted_label, confidence, was_correct):

    file_exists = os.path.exists(FEEDBACK_FILE)

    with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header if file is new
        if not file_exists:
            writer.writerow(["timestamp", "predicted_label", "confidence", "was_correct"])

        writer.writerow([
            datetime.now().isoformat(),
            str(predicted_label),
            float(confidence),
            bool(was_correct),
        ])

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
    prediction = model.predict(input_tensor)[0]  # Get the first (and only) prediction vector
    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index]     # Get the confidence of the predicted class
    result = class_names[predicted_index]
    return result, confidence

# UI
st.markdown("<h1 style='text-align: center;'>Durian Maturity Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Take a photo or upload one to predict if your durian is ready!</p>", unsafe_allow_html=True)

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
        result, confidence = predict(image)
        st.session_state["last_pred"] = (result, confidence)
        
        st.session_state["show_feedback"] = True

    #Add receiving feedback feature
        
if st.session_state["last_pred"] is not None:
    result, confidence = st.session_state["last_pred"]

    st.success(f"✅ **Prediction:** {result} ({confidence*100:.2f}%)")

    st.write("Was the prediction correct?")
    fb_col1, fb_col2 = st.columns(2)

    with fb_col1:
        if st.button("Yes"):
            st.session_state["correct_count"] += 1
            save_feedback_to_csv(result, confidence, True)

    with fb_col2:
        if st.button("No"):
            st.session_state["wrong_count"] += 1
            save_feedback_to_csv(result, confidence, False) 

    total = st.session_state["correct_count"] + st.session_state["wrong_count"]
    if total > 0:
        acc = st.session_state["correct_count"] / total * 100
        st.info(f"User-verified accuracy: **{acc:.1f}%** ({st.session_state['correct_count']}/{total})")
        

st.markdown("---")
st.subheader("Admin Download")

# Admin code is stored in secrets, never shown
admin_code = st.text_input("Enter admin code to access feedback:", type="password")

# Only compare against the secret; never print or reveal it
real_admin_code = st.secrets.get("ADMIN_CODE", "")

if admin_code:
    if admin_code == real_admin_code:
        st.success("Admin access granted.")

        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "rb") as f:
                st.download_button(
                    label="📥 Download feedback.csv",
                    data=f,
                    file_name="feedback.csv",
                    mime="text/csv",
                )
        else:
            st.info("No feedback file found yet.")
    else:
        st.error("Incorrect admin code.")