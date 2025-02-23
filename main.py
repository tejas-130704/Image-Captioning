import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import preprocessing
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load tokenizer from JSON
with open("tokenizer.json", "r") as json_file:
    tokenizer_json = json_file.read()

tokenizer = tokenizer_from_json(tokenizer_json)

# Load the pre-trained model (you will implement captioning logic)
model = tf.keras.models.load_model("models/model.h5")

# Streamlit App Interface
st.title("Image Captioning App")
st.write("Upload an image and get its caption.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file,target_size=(224,224,3))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    
    if st.button("Generate Caption"):
    
        caption = preprocessing.predict_caption(model, img, tokenizer, max_length=34)
        st.write(caption)
