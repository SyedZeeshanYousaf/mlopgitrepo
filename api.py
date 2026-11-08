import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
app = FastAPI(title="House Price API")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


class House(BaseModel):
    rooms: int
    size: float   # sqft


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(h: House):
    x = [[h.rooms, h.size]]
    return {"predicted_price_lakh": float(get_model().predict(x)[0])}