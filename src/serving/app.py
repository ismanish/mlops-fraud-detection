"""FastAPI application for fraud prediction serving."""

import time
from contextlib import asynccontextmanager

import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import load_artifacts, predict
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    logger.info("Model loaded — ready to serve")
    yield


app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    description="Real-time credit card fraud detection",
    lifespan=lifespan,
)


class TransactionRequest(BaseModel):
    V1: float = Field(..., description="PCA component V1")
    V2: float = Field(..., description="PCA component V2")
    V3: float = Field(..., description="PCA component V3")
    V4: float = Field(..., description="PCA component V4")
    V5: float = Field(..., description="PCA component V5")
    V6: float = Field(..., description="PCA component V6")
    V7: float = Field(..., description="PCA component V7")
    V8: float = Field(..., description="PCA component V8")
    V9: float = Field(..., description="PCA component V9")
    V10: float = Field(..., description="PCA component V10")
    V11: float = Field(..., description="PCA component V11")
    V12: float = Field(..., description="PCA component V12")
    V13: float = Field(..., description="PCA component V13")
    V14: float = Field(..., description="PCA component V14")
    V15: float = Field(..., description="PCA component V15")
    V16: float = Field(..., description="PCA component V16")
    V17: float = Field(..., description="PCA component V17")
    V18: float = Field(..., description="PCA component V18")
    V19: float = Field(..., description="PCA component V19")
    V20: float = Field(..., description="PCA component V20")
    V21: float = Field(..., description="PCA component V21")
    V22: float = Field(..., description="PCA component V22")
    V23: float = Field(..., description="PCA component V23")
    V24: float = Field(..., description="PCA component V24")
    V25: float = Field(..., description="PCA component V25")
    V26: float = Field(..., description="PCA component V26")
    V27: float = Field(..., description="PCA component V27")
    V28: float = Field(..., description="PCA component V28")
    Amount: float = Field(..., ge=0, description="Transaction amount")


class PredictionResponse(BaseModel):
    prediction: int
    fraud_probability: float
    label: str
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "healthy", "model": "loaded"}


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionRequest):
    start = time.time()
    try:
        result = predict(transaction.model_dump())
        latency = (time.time() - start) * 1000
        result["latency_ms"] = round(latency, 2)

        _publish_metrics(result)

        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _publish_metrics(result: dict):
    try:
        cloudwatch = boto3.client("cloudwatch", region_name="us-east-1")
        cloudwatch.put_metric_data(
            Namespace="MLOps/FraudDetection",
            MetricData=[
                {
                    "MetricName": "PredictionLatency",
                    "Value": result["latency_ms"],
                    "Unit": "Milliseconds",
                },
                {
                    "MetricName": "FraudPredicted",
                    "Value": result["prediction"],
                    "Unit": "Count",
                },
                {
                    "MetricName": "FraudProbability",
                    "Value": result["fraud_probability"],
                    "Unit": "None",
                },
            ],
        )
    except Exception as e:
        logger.warning(f"Failed to publish CloudWatch metrics: {e}")
