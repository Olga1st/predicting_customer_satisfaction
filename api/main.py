# api/main.py

from fastapi import FastAPI
import joblib
import pandas as pd

from api.schemas import ReviewRequest, PredictionResponse

app = FastAPI(title="Trustpilot NLP API")

# Modell laden (einmal beim Start)
MODEL_PATH = "models/model.joblib"
model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {"message": "Trustpilot NLP API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ReviewRequest):
    """
    Single prediction endpoint
    """
    df = pd.DataFrame([{
        "review_text": request.review_text,
        "supplier_response": request.supplier_response,
        "verified": request.verified,
        "review_text_clean": request.review_text
    }])

    pred = model.predict(df)[0]

    return PredictionResponse(prediction=int(pred))