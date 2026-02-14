import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==============================
# Load Model (runs once)
# ==============================
model = load_model("models/weed_model_v2/weed_model.h5")

CLASS_NAMES = ["Crop", "Weed"]


# ==============================
# Image Preprocess
# ==============================
def preprocess_image(img_file):
    img = Image.open(img_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img


# ==============================
# Density Estimation
# ==============================
def estimate_density(img):
    arr = np.array(img)

    # detect green vegetation pixels
    green_pixels = np.sum(
        (arr[:, :, 1] > arr[:, :, 0]) &
        (arr[:, :, 1] > arr[:, :, 2])
    )

    total_pixels = arr.shape[0] * arr.shape[1]
    density = (green_pixels / total_pixels) * 100

    return round(density, 2)


# ==============================
# Recommendation Engine
# ==============================
def generate_recommendation(label, density):

    if label == "Crop":
        return "Healthy crop detected. No weed control needed."

    if density < 10:
        return "Low weed presence. Manual removal recommended."

    elif density < 30:
        return "Moderate weeds detected. Use mechanical weeding."

    elif density < 60:
        return "High weed density. Use targeted eco-friendly herbicide."

    else:
        return "Very high weed infestation. Immediate action required: combine manual + organic control."


# ==============================
# Main Prediction Function
# ==============================
def predict_weed(img_file):

    img_array, original_img = preprocess_image(img_file)

    prediction = model.predict(img_array)

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    label = CLASS_NAMES[class_index]

    density = estimate_density(original_img)

    recommendation = generate_recommendation(label, density)

    return label, confidence, density, recommendation