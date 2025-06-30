import os
import requests

def download_model():
    url = "https://drive.google.com/uc?export=download&id=1LCBxOvygsdJSRZ9IXbTlX3DvAmd1F4Pd"
    model_path = "models/heart_disease_rf_optimized.pkl"

    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        print("📥 Downloading model from Google Drive...")
        os.makedirs("models", exist_ok=True)
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)

        # Double-check size
        if os.path.getsize(model_path) < 1000000:
            raise ValueError("❌ Download failed or file corrupted.")
        else:
            print("✅ Model downloaded successfully.")
