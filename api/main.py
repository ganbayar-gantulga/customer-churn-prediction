"""
FastAPI сервер — Churn таамаглал API
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Харилцагчийн үйлчилгээнээс гарах магадлалыг таамаглана",
    version="1.0.0",
)

# Загвар ачаалах
try:
    model = joblib.load("../models/best_model.pkl")
except FileNotFoundError:
    model = None


# ============================================================
# Schemas
# ============================================================

class CustomerInput(BaseModel):
    """Нэг харилцагчийн өгөгдөл."""
    tenure: int = Field(..., ge=0, le=72, description="Хэдэн сар үйлчлүүлсэн")
    monthly_charges: float = Field(..., ge=0, description="Сарын төлбөр")
    total_charges: float = Field(..., ge=0, description="Нийт төлбөр")
    contract: int = Field(..., ge=0, le=2, description="0=Сар, 1=1жил, 2=2жил")
    internet_service: int = Field(..., ge=0, le=2, description="0=Үгүй, 1=DSL, 2=Fiber")
    online_security: int = Field(..., ge=0, le=1, description="0=Үгүй, 1=Тийм")
    tech_support: int = Field(..., ge=0, le=1, description="0=Үгүй, 1=Тийм")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tenure": 5,
                    "monthly_charges": 89.5,
                    "total_charges": 447.5,
                    "contract": 0,
                    "internet_service": 2,
                    "online_security": 0,
                    "tech_support": 0,
                }
            ]
        }
    }


class PredictionOutput(BaseModel):
    """Таамаглалын үр дүн."""
    churn_probability: float
    will_churn: bool
    risk_level: str  # "Low", "Medium", "High"
    recommendation: str


# ============================================================
# Endpoints
# ============================================================

@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API", "status": "running"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_churn(customer: CustomerInput):
    """Нэг харилцагчийн churn магадлалыг таамаглана."""
    if model is None:
        raise HTTPException(status_code=503, detail="Загвар ачаалагдаагүй байна")

    # Input-ийг DataFrame болгох
    input_data = pd.DataFrame([customer.model_dump()])

    # Таамаглал
    probability = float(model.predict_proba(input_data)[0][1])
    will_churn = probability >= 0.5

    # Эрсдэлийн түвшин
    if probability < 0.3:
        risk_level = "Low"
        recommendation = "Харилцагч тогтвортой байна. Одоогийн үйлчилгээг үргэлжлүүлнэ."
    elif probability < 0.7:
        risk_level = "Medium"
        recommendation = "Анхаарал хандуулах шаардлагатай. Хөнгөлөлт эсвэл урамшуулал санал болгоно."
    else:
        risk_level = "High"
        recommendation = "Яаралтай арга хэмжээ авах! Хувийн менежер томилох, тусгай санал өгөх."

    return PredictionOutput(
        churn_probability=round(probability, 4),
        will_churn=will_churn,
        risk_level=risk_level,
        recommendation=recommendation,
    )


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerInput]):
    """Олон харилцагчийн churn магадлалыг нэг дор таамаглана."""
    if model is None:
        raise HTTPException(status_code=503, detail="Загвар ачаалагдаагүй байна")

    results = []
    for customer in customers:
        result = predict_churn(customer)
        results.append(result)

    return {
        "total": len(results),
        "high_risk_count": sum(1 for r in results if r.risk_level == "High"),
        "predictions": results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
