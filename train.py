import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# =============================
# KONFIGURASI
# =============================
IMG_SIZE = (150, 150)
BATCH_SIZE = 8
EPOCHS = 3

DATASET_DIR = "dataset"
MODEL_PATH = "model/best_model.h5"

# =============================
# BUAT FOLDER DATASET
# =============================
os.makedirs("dataset/train/Depresi", exist_ok=True)
os.makedirs("dataset/train/Normal", exist_ok=True)
os.makedirs("model", exist_ok=True)

# =============================
# DATA GENERATOR
# =============================
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    "dataset/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# =============================
# MODEL CNN
# =============================
model = Sequential([
    Conv2D(16, (3,3), activation="relu", input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =============================
# TRAIN
# =============================
model.fit(train_data, epochs=EPOCHS)

# =============================
# SAVE MODEL
# =============================
model.save(MODEL_PATH)
print("✅ Model berhasil disimpan ke:", MODEL_PATH)
print("TRAIN FILE TERDETEKSI DAN JALAN")
