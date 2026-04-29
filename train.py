import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import json

# =========================
# Dataset Paths
# =========================
dataset_path = "/content/drive/MyDrive/apple_dataset"

train_dir = dataset_path + "/train"
val_dir   = dataset_path + "/validation"
test_dir  = dataset_path + "/test"

# =========================
# Data Generators
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Save class indices
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

# =========================
# Model
# =========================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# Train
# =========================
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# =========================
# Save Model
# =========================
model.save("apple_model.keras")

print("✅ Training Complete")
