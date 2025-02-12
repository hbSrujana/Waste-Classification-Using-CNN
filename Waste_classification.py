import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Apply custom CSS for styling
st.markdown(
    """
    <style>
        body {
            position : fixed
            background-color: #f4f4f4;
            font-family: Arial, sans-serif;
        }
        .stApp {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
        }
        .title {
            color: #004D40;
            text-align: center;
            font-size: 32px;
            font-weight: bold;
        }
        .upload-box {
            border: 2px dashed #004D40;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            background-color: #e0f7fa;
            font-size: 18px;
            font-weight: bold;
            color: #004D40;
        }
        .result-box {
            background-color: #004D40;
            color: white;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            color: #7a7a7a;
            font-size: 18px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model.keras")

model = load_model()

# Define class labels
class_labels = ['Organic', 'Recyclable']

def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)

# Sidebar with "About" section
st.sidebar.title("📌 About")
st.sidebar.markdown(
    """
    This project classifies waste into **Organic** or **Recyclable** using a **Convolutional Neural Network (CNN)**.  
    """,
    unsafe_allow_html=True
)

# Title with improved visibility
st.markdown("<h1 class='title'>♻️ Waste Classification App</h1>", unsafe_allow_html=True)
st.write("Upload an image to classify it as **Organic** or **Recyclable** waste.")

# File uploader with improved visibility
st.markdown("<div class='upload-box'>📤 Choose an image (JPG, PNG, JPEG)</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Resize image for display
    st.image(image, caption="🖼️ Uploaded Image", width=250)  # Reduced size

    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Get model prediction
    prediction = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    # Display result with better contrast
    st.markdown(f"<div class='result-box'>🎯 Prediction: <b>{predicted_class}</b> <br> 📊 Confidence: {confidence:.2f}%</div>", unsafe_allow_html=True)

# Footer with developer credit
st.markdown("<div class='footer'>Developed with ❤️ by <b>Srujana</b>.</div>", unsafe_allow_html=True)
