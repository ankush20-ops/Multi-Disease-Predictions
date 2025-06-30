import os
import gdown

def download_model():
    model_path = "models/heart_disease_rf_optimized.pkl"

    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        url = "https://drive.google.com/uc?id=1LCBxOvygsdJSRZ9IXbTlX3DvAmd1F4Pd"
        output = model_path
        gdown.download(url, output, quiet=False)