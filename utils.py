import os
import gdown

def download_model():
    model_path = "models/heart_disease_rf_optimized.pkl"

    if not os.path.exists(model_path) or os.path.getsize(model_path) < 5000000:  # ~5MB minimum
        print("Downloading model from Google Drive...")
        os.makedirs("models", exist_ok=True)
        url = "https://drive.google.com/uc?id=1LCBxOvygsdJSRZ9IXbTlX3DvAmd1F4Pd"
        gdown.download(url, model_path, quiet=False)

        # Verify again
        if os.path.getsize(model_path) < 5000000:
            raise ValueError("❌ Download failed or model corrupted!")