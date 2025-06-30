import os
import requests

def download_model():
    url = "https://drive.google.com/uc?export=download&id=1LCBxOvygsdJSRZ9IXbTlX3DvAmd1F4Pd"
    model_path = "models/heart_disease_rf_optimized.pkl"

    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        print("Downloading model...")
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("Model downloaded successfully.")
