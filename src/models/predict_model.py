import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models/model.joblib"

def load_model():
    return joblib.load(MODEL_PATH)

def predict(texts: list[str]) -> list:
    model = load_model()

    df = pd.DataFrame({
        "review_text": texts,
        "supplier_response": [None] * len(texts),
        "verified": [0] * len(texts)
    })

    # Dummy Features (werden später im Preprocessing verarbeitet)
    df["review_text_clean"] = df["review_text"]

    preds = model.predict(df)
    return preds


if __name__ == "__main__":
    sample = ["Great product, fast delivery!", "Terrible service, very disappointed."]
    predictions = predict(sample)
    print(predictions)