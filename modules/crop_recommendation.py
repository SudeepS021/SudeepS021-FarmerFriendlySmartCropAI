import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

MODEL_PATH = "models/crop_model/crop_model.pkl"

def train_crop_model():
    data_path = "data/crops/Crop_recommendation.csv"

    if not os.path.exists(data_path):
        return "Dataset not found!"

    data = pd.read_csv(data_path)

    X = data.drop("label", axis=1)
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    os.makedirs("models/crop_model", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return "Crop model trained successfully!"

def predict_crop(features):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    prediction = model.predict([features])
    return prediction[0]