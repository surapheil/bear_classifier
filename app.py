import gradio as gr
from fastai.vision.all import *

# Load the trained model
learn = load_learner("bear_classifier.pkl")

# Prediction function
def predict(img):
    pred, pred_idx, probs = learn.predict(img)
    return {
        learn.dls.vocab[i]: float(probs[i])
        for i in range(len(probs))
    }

# Create the interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="🐻 Bear Classifier",
    description="Upload a bear image and the AI will predict its type."
)

demo.launch()