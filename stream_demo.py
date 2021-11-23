import streamlit as st
from fastai.vision.widgets import *
import os
from fastbook import *


def return_list(x): return [x] #need to add this custom function from learner
inferer = load_learner('multimod.pkl')

st.title("Insect Inferer")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "heic"])
if uploaded_file is not None:
    image = PILImage.create(uploaded_file)
    st.image(image, caption='Uploaded Image.', width=256, use_column_width=False)
    st.write("Classifying...")
    st.write("")
    label, _, probs = inferer.predict(image)
    if probs.max().item() > 0.7:
        st.write(f"I am guessing this is a {label[0]} with a probability of: {probs.max().item() * 100:.04f}%")
    else:
        st.write("Excuse me I am not sure what this is")
    
    #label = predict(uploaded_file)
    #st.write('%s (%.2f%%)' % (label[1], label[2]*100))

