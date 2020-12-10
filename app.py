import streamlit as st
from PIL import Image, ImageOps
from img_classification import teachable_machine_classification


st.title("Pneumonia Classifier")
# st.header("")
st.header("Upload a chest X-ray image to be classified as Normal or Pneumonia")

weights_file = "./Adam_32_32_32_32__best"
# weights_file = "./checkpoints/testloadd.data-00000-of-00001"

uploaded_file = st.file_uploader("I said chest X-ray!! If you try something else I might blow up...", type=["jpg","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, weights_file)
    if label == 0:
        st.write("The image has pneumonia")
    else:
        st.write("The image is healthy")