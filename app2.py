import streamlit as st
import cv2
from PIL import Image, ImageOps
import tensorflow as tf

import numpy as np

model = tf.keras.models.load_model("./Adam_32_32_32_32__best")
st.write("""
         # Rock-Paper-Scissor Hand Sign Prediction
         """
         )
st.write("This is a simple image classification web app to predict rock-paper-scissor hand sign")
file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])

def import_and_predict(image_data, model):
    
        size = (64,64)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(224, 224),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a paper!")
    elif np.argmax(prediction) == 1:
        st.write("It is a rock!")
    else:
        st.write("It is a scissor!")
    
    st.text("Probability (0: Paper, 1: Rock, 2: Scissor")
    st.write(prediction)