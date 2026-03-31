# api/schemas.py

from pydantic import BaseModel
from typing import Optional


class ReviewRequest(BaseModel):
    review_text: str
    supplier_response: Optional[str] = None
    verified: int = 0


class PredictionResponse(BaseModel):
    prediction: int