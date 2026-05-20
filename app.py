import os
import json
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image
import os

print("FILES IN DIRECTORY:")
print(os.listdir("."))
print("MODEL EXISTS:", os.path.exists("apple_model.keras"))
# =========================
# CONFIG
# =========================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

IMG_SIZE = (224, 224)

# =========================
# LOAD CLASS LABELS
# =========================
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Convert:
# {"black_rot":0,"healthy":1,"rust":2,"scab":3}
# into:
# {0:"black_rot",1:"healthy",2:"rust",3:"scab"}

idx_to_class = {v: k for k, v in class_indices.items()}

print("Loaded classes:", idx_to_class)

# =========================
# LOAD MODEL
# =========================
print("TensorFlow:", tf.__version__)
print("Keras:", tf.keras.__version__)

MODEL_PATH = "apple_model.keras"

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("✅ Model loaded successfully")

# =========================
# STARTUP TEST
# =========================
try:
    dummy = tf.random.normal((1, 224, 224, 3))
    result = model.predict(dummy, verbose=0)

    print("✅ Startup prediction successful")
    print("Prediction shape:", result.shape)

except Exception as e:
    print("❌ Startup prediction failed")
    print(e)
    raise e

# =========================
# PREDICTION FUNCTION
# =========================
def predict(image):

    if image is None:
        return {"No image uploaded": 1.0}

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image).astype("float32") / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    results = {
        idx_to_class[i]: float(predictions[i])
        for i in range(len(predictions))
    }

    return results

# =========================
# GRADIO INTERFACE
# =========================
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    title="Apple Leaf Disease Classifier",
    description="Upload an apple leaf image to classify the disease."
)

# =========================
# LAUNCH
# =========================
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
