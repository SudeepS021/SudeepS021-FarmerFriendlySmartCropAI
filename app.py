import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from modules.pest_detection import predict_pest
from modules.weed_detection import predict_weed
from modules.fertilizer import recommend_fertilizer

# ------------------------------------------------
# Load Models
# ------------------------------------------------

@st.cache_resource
def load_models():

    models = {}

    with open("models/crop_model/crop_model.pkl", "rb") as f:
        models["crop"] = pickle.load(f)

    models["soil"] = load_model("models/soil_model/soil_model.h5")

    return models

models = load_models()

# ------------------------------------------------
# Soil Setup
# ------------------------------------------------

SOIL_CLASSES = ["Black Soil", "Red Soil", "Sandy Soil", "Yellow Soil"]

SOIL_NPK_MAP = {
    "Black Soil": (90, 40, 40),
    "Red Soil": (60, 30, 30),
    "Sandy Soil": (30, 20, 15),
    "Yellow Soil": (50, 25, 25),
}

# ------------------------------------------------
# UI
# ------------------------------------------------

st.set_page_config(page_title="Farmer Friendly Smart Crop AI")

st.title("🌾 Farmer Friendly Smart Crop AI")

option = st.sidebar.selectbox(
    "Select Module",
    [
        "Crop Recommendation (Manual)",
        "Crop Recommendation (Soil Image)",
        "Fertilizer Recommendation",
        "Pest Detection",
        "Weed Detection",
    ],
)

# =================================================
# Manual Crop
# =================================================

if option == "Crop Recommendation (Manual)":

    N = st.number_input("Nitrogen", 0, 200, 50)
    P = st.number_input("Phosphorus", 0, 200, 50)
    K = st.number_input("Potassium", 0, 200, 50)

    if st.button("Recommend Crop"):
        features = np.array([[N, P, K, 25, 70, 6.5, 200]])
        crop = models["crop"].predict(features)[0]
        st.success(f"🌾 Recommended Crop: {crop}")

# =================================================
# Soil Based Crop
# =================================================

elif option == "Crop Recommendation (Soil Image)":

    uploaded_file = st.file_uploader("Upload Soil Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:

        st.image(uploaded_file)

        img = image.load_img(uploaded_file, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        soil_pred = models["soil"].predict(img_array)
        soil_index = np.argmax(soil_pred)
        soil_type = SOIL_CLASSES[soil_index]

        st.success(f"Detected Soil: {soil_type}")

        N, P, K = SOIL_NPK_MAP[soil_type]

        features = np.array([[N, P, K, 25, 70, 6.5, 200]])
        crop = models["crop"].predict(features)[0]

        st.success(f"🌾 Recommended Crop: {crop}")

# =================================================
# Fertilizer
# =================================================
elif option == "Fertilizer Recommendation":

    crop_name = st.text_input("Enter Crop Name (rice, wheat, maize, cotton, sugarcane)")
    N = st.number_input("Nitrogen", 0, 200, 50)
    P = st.number_input("Phosphorus", 0, 200, 50)
    K = st.number_input("Potassium", 0, 200, 50)

    if st.button("Recommend Fertilizer"):
        result = recommend_fertilizer(N, P, K, crop_name)
        st.success(result)

        st.markdown("### 🛒 Want to buy this fertilizer?")
        search_query = result.replace(" ", "+")
        buy_link = f"https://www.amazon.in/s?k={search_query}"
        st.link_button("Buy Now", buy_link)
# =================================================
# Pest
# =================================================

elif option == "Pest Detection":

    uploaded_file = st.file_uploader("Upload Pest Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        st.image(uploaded_file)

        pest, confidence = predict_pest(uploaded_file)

        st.error(f"Detected Pest: {pest}")
        st.write(f"Confidence: {confidence*100:.2f}%")

# =================================================
# Weed
# =================================================

elif option == "Weed Detection":

    st.subheader("🌿 Upload Crop Field Image")

    uploaded_file = st.file_uploader(
        "Upload Image", 
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # ✅ IMPORTANT: 4 variables
        label, confidence, density, recommendation = predict_weed(uploaded_file)

        st.success(f"🌱 Detection Result: {label}")
        st.info(f"📊 Confidence: {confidence:.2f}%")
        st.write(f"🌿 Density Level: {density:.2f}%")

        st.warning(recommendation)