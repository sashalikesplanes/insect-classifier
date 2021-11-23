import streamlit as st
from fastbook import load_learner, PILImage


def return_list(x): return [x] #need to add this custom function from learner
inferer = load_learner('multimod.pkl')

st.title("Insect Inferer")
st.write("A simple web app where a user may upload the picture of an insect found in their house and it is classified into one of the top 7 common insects (spiders, mosquitos, fruit flies, bed bugs, cockroaches, moths, silverfish) or informs the user that it does not know the insect.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "heic"])
output = st.empty()
if uploaded_file is not None:
    image = PILImage.create(uploaded_file)
    output.text("Classifying...")
    st.image(image, caption='Uploaded Image.', width=256, use_column_width=False)
    st.write("")
    label, _, probs = inferer.predict(image)
    if probs.max().item() > 0.7:
        output.text("I am guessing this is a {label[0]} with a probability of: {probs.max().item() * 100:.04f}%")
    else:
        output.text("Excuse me I am not sure what this is")
    
    #label = predict(uploaded_file)
    #st.write('%s (%.2f%%)' % (label[1], label[2]*100))

