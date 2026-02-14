import numpy as np
import tensorflow as tf
from PIL import Image

PEST_MODEL_PATH = "models/pest_model/pest_model.h5"

# Load once
pest_model = tf.keras.models.load_model(PEST_MODEL_PATH)

# Adjust according to your dataset folder names
PEST_CLASSES = ["moth", "slug", "snail", "wasp"]

def preprocess_image(img, size=(224, 224)):
    img = img.resize(size)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predict_pest(image_file):
    img = Image.open(image_file).convert("RGB")
    img_array = preprocess_image(img)

    prediction = pest_model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return PEST_CLASSES[class_index], confidence