import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model (must be in same folder)
model = tf.keras.models.load_model("apple_model_checkpoint.keras")

# Class labels (must match training order)
classes = ["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"]

def predict_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    return f"{classes[class_index]} ({confidence*100:.2f}%)"

gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(),
    title="🍎 Apple Disease Detection AI"
).launch()
