import os
import tensorflow as tf

# =========================
# 1. FORCE STABLE EXECUTION MODE
# =========================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# IMPORTANT: ensure consistent Keras behavior
tf.keras.mixed_precision.set_global_policy("float32")

print("TensorFlow:", tf.__version__)
print("Keras:", tf.keras.__version__)

# =========================
# 2. LOAD MODEL SAFELY
# =========================
MODEL_PATH = "apple_model.keras"  # or full path on Render

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,  # VERY IMPORTANT FIX
        safe_mode=False  # avoids strict config validation
    )
    print("✅ Model loaded successfully")

except Exception as e:
    print("❌ Primary load failed:", e)

    # =========================
    # 3. FALLBACK LOADER (CRITICAL FOR KERAS 3 ISSUES)
    # =========================
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={}
        )
        print("✅ Model loaded with fallback method")

    except Exception as e2:
        print("❌ Fatal model load error:", e2)
        raise e2


# =========================
# 4. SIMPLE PREDICTION FUNCTION (SAFE)
# =========================
from tensorflow.keras.preprocessing import image
import numpy as np

IMG_SIZE = (224, 224)

def predict(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    preds = model.predict(x)
    return preds
