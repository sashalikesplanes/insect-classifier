import streamlit as st
from fastbook import load_learner, PILImage


# load the learner and add the function needed for the learner 
def return_list(x): return [x] #need to add this custom function from learner
inferer = load_learner('multimod.pkl')
# setup the title and description of web app
st.title("Insect Classifier")
st.write("A simple web app where a user may upload the picture of an insect found in their house and it is classified into one of the top 7 common insects (spiders, mosquitos, fruit flies, bed bugs, cockroaches, moths, silverfish) or informs the user that it does not know the insect.")

uploaded_file = st.file_uploader("Choose an image with an insect", type=["jpg", "jpeg", "heic", "png"])

#output = st.empty() # make an empty object to be used to write the text for the prediction

if uploaded_file is not None: # once an image is uploaded run inference
    image = PILImage.create(uploaded_file)
    with st.spinner("Classifying..."):
        st.image(image, caption='Your image', width=512, use_column_width=False)
        st.write("")
        label, _, probs = inferer.predict(image)
    if probs.max().item() > 0.7:
        st.success(f"I am guessing this is a {label[0]} with a probability of: {probs.max().item() * 100:.04f}%")
        st.balloons()
    else:
        st.success("Excuse me, I am not sure what this is")


