import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH = "models/soil_model/soil_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

# Soil classes (same order as training)
CLASS_NAMES = [
    "Black Soil",
    "Cinder Soil",
    "Laterite Soil",
    "Peat Soil",
    "Yellow Soil"
]

# Rule-based NPK mapping (simple & explainable)
SOIL_NPK_MAP = {
    "Black Soil":     {"N": 90, "P": 45, "K": 50},
    "Cinder Soil":    {"N": 40, "P": 30, "K": 35},
    "Laterite Soil":  {"N": 35, "P": 25, "K": 30},
    "Peat Soil":      {"N": 70, "P": 40, "K": 45},
    "Yellow Soil":    {"N": 50, "P": 35, "K": 40},
}

def predict_soil_and_npk(image):
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    index = np.argmax(prediction)
    confidence = prediction[0][index]

    soil_type = CLASS_NAMES[index]
    npk = SOIL_NPK_MAP[soil_type]

    return soil_type, confidence, npk