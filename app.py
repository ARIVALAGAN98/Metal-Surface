import streamlit as st
import cv2
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io

new_model = tf.keras.models.load_model("my_model")

# Define class names
class_names = ['Crazing', 'Inclusion', 'Patches', 'Pitted Surface', 'Rolled-in Scale', 'Scratches']

# Function to convert image to grayscale if it is RGB
def convert_to_grayscale(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

# Function to detect defects
def detect_defects(img):
    st.text(img.shape)
    #img = convert_to_grayscale(img)
    new_img = img.astype('float32')/255
    if new_img is not None:
        # Reshape input to match model's input shape
        new_img = np.reshape(new_img, (1, new_img.shape[0], new_img.shape[1], -1))
        st.text(new_img.shape)
        # Make predictions
        prediction = new_model.predict(new_img)
        # Get the index of the class with the highest probability
        predicted_class_index = np.argmax(prediction)
        # Get the name of the predicted class
        predicted_class_name = class_names[predicted_class_index]
        return predicted_class_name
    else:
        return 'no image detected'

# Function to convert uploaded image to array and detect defects
def convert_image_to_array(image_bytes):
    # If image_bytes is already in bytes format, skip the bytearray() step
    if isinstance(image_bytes, bytes):
        img_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    else:
        with open(image_bytes, 'rb') as f:
            img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
    
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))
    return img

# Streamlit app
def app():
    st.title("Defect Detection App")
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["bmp","jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Show uploaded image
        img_bytes = uploaded_file.read()
        img_format = uploaded_file.type.split('/')[1]
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        # Detect defects
        if st.button("Detect Defects"):
            output = detect_defects(convert_image_to_array(img_bytes))
            st.write(output)
            st.success("Defects detected!")

app()
