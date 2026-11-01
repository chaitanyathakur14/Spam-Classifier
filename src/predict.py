"""Model loading and prediction helpers used by the Streamlit app and tests."""
import json
 
import joblib
 
from src.preprocess import clean_text
from src.train import METRICS_PATH, MODEL_PATH, train_all
 
 
def load_model():
    """Load the saved model; retrain automatically if it is missing or incompatible."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return train_all(log_to_mlflow=False)["model"]
 
 
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
 
 
def predict_message(model, text: str) -> dict:
    cleaned = clean_text(text)
    spam_prob = float(model.predict_proba([cleaned])[0][1])
    return {"label": "Spam" if spam_prob >= 0.5 else "Not Spam",
            "spam_probability": spam_prob, "cleaned": cleaned}
