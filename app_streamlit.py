import streamlit as st
from fastai.vision.all import *
from PIL import Image

learn = load_learner("bear_classifier.pkl")

st.title("🐻 Bear Classifier AI")

uploaded_file = st.file_uploader(
    "Upload a bear image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    img = Image.open(uploaded_file)

    st.image(
        img,
        caption="Uploaded image"
    )

    pred, pred_idx, probs = learn.predict(img)

    st.success(f"Prediction: {pred}")

    st.write("### Prediction probabilities")

    for i, probability in enumerate(probs):
        label = learn.dls.vocab[i]
        
        st.write(
            f"{label}: {probability*100:.2f}%"
        )