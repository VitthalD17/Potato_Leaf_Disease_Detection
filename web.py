import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2

# Load the pre-trained model
MODEL_PATH = 'plant_disease_model.h5'
model = load_model(MODEL_PATH)

# Define the labels
LABELS = ['Potato-Early_blight', 'Potato-Late_blight', 'Potato-Healthy']

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (256, 256))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Streamlit app
st.title("Potato Leaf Disease Prediction")
st.write("Upload an image of a potato leaf to predict the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Predict the class
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Display the result
    st.write(f"Predicted Class: {LABELS[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}%")

st.write("---")
st.write("Note: Ensure that the image is clear and focused for better accuracy.")
