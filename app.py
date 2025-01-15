import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import gdown
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

url = "https://drive.google.com/uc?export=download&id=1qL1GjQiLWwoEHAoXqDKn23V_wpS8vMtE"
output = "cats_and_dogs_vgg_model.h5"

# Download the .h5 file from Google Drive
gdown.download(url, output, quiet=False)

# Load the trained model
model = load_model(output)

# Define class labels
class_names = ["Cat", "Dog"]

# Streamlit app
st.title("🐾 Cats and Dogs Classification 🐾")
st.markdown("""
    <style>
    .main {background-color: #F8F8FF;}
    h1 {color: #ff6f61;}
    </style>
    """, unsafe_allow_html=True)

# Display example images and classifications
st.markdown("### 🖼️ Example Images:")
col1, col2 = st.columns(2)

with col1:
    # Show example cat image with prediction
    st.image("cat_example.jpg", caption="Cat Example", use_container_width=True)
    st.markdown("**Classified as: Cat**")

with col2:
    # Show example dog image with prediction
    st.image("dog_example.jpg", caption="Dog Example", use_container_width=True)
    st.markdown("**Classified as: Dog**")

st.markdown("### Upload an image to classify it as a **Cat** or **Dog**!")

st.sidebar.title("Navigation")
st.sidebar.write("Upload your image below:")

# Upload image
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# How It Works section
with st.expander("📖 How It Works", expanded=True):
    st.markdown("""
        1. Upload an image of a cat or dog.
        2. The model will preprocess the image and make a prediction.
        3. The result will display whether the image is classified as a **cat** or **dog**.
    """)

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("🔍 **Classifying...**")

    # Progress bar
    progress = st.progress(0)
    for percent_complete in range(100):
        progress.progress(percent_complete + 1)

    # Preprocess the image
    image = load_img(uploaded_file, target_size=(256, 256))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    prediction = model.predict(image_array)
    predicted_class = class_names[int(prediction[0] > 0.5)]
    st.success(f"🎉 The image is classified as: **{predicted_class}**")

    # Download result
    result = f"Image Classification Result: {predicted_class}"
    st.download_button("📥 Download Result", result, file_name="classification_result.txt")
