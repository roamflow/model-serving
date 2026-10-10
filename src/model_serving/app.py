"""FastAPI app: /health (liveness), /ready (readiness), /predict, /metrics."""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from model_serving.model import load_model

logger = logging.getLogger("model_serving")

N_FEATURES = 4  # iris: sepal/petal length and width

# ---- Prometheus Metrics ----
REQUEST_COUNT = Counter(
    "model_serving_requests_total",
    "Total HTTP requests received",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "model_serving_request_duration_seconds",
    "HTTP request end-to-end latency in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5],
)

INFERENCE_LATENCY = Histogram(
    "model_serving_inference_duration_seconds",
    "Raw ML model predict() compute latency in seconds",
    ["model_name"],
    buckets=[0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)

PREDICTIONS_COUNT = Counter(
    "model_serving_predictions_total",
    "Total model predictions segmented by predicted class (Label Shift / Output Drift)",
    ["model_name", "predicted_class"],
)

BATCH_SIZE = Histogram(
    "model_serving_batch_size",
    "Batch size of prediction requests",
    ["endpoint"],
    buckets=[1, 2, 5, 10, 20, 50, 100, 500],
)

MODEL_INFO = Gauge(
    "model_serving_model_info",
    "Metadata about the currently loaded model",
    ["model_name", "model_uri", "app_version"],
)


class PredictRequest(BaseModel):
    # Pydantic rejects bad shapes/types with HTTP 422 before our code runs.
    instances: list[list[float]] = Field(
        min_length=1,
        max_length=1000,
        description="Batch of feature vectors, each with exactly 4 floats.",
        examples=[[[5.1, 3.5, 1.4, 0.2]]],
    )


class PredictResponse(BaseModel):
    predictions: list[int]


APP_VERSION = os.getenv("APP_VERSION", "dev")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup, not per request: loading can take seconds and
    # would blow the latency budget on every call.
    logger.info("Loading model...")
    app.state.model = load_model()
    logger.info("Model loaded.")

    model_name = os.getenv("MODEL_NAME", "iris-classifier")
    model_uri = os.getenv("MODEL_URI", "fallback-in-process")
    MODEL_INFO.labels(
        model_name=model_name,
        model_uri=model_uri,
        app_version=APP_VERSION,
    ).set(1)

    yield
    app.state.model = None


app = FastAPI(title="model-serving", lifespan=lifespan)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    endpoint = request.url.path
    if endpoint == "/metrics":
        return await call_next(request)

    start_time = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        latency = time.perf_counter() - start_time
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: the process is up. Must not depend on the model or other
    services, otherwise a slow dependency makes Kubernetes restart the pod."""
    return {"status": "ok", "version": APP_VERSION}


@app.get("/ready")
def ready(request: Request) -> dict[str, str]:
    """Readiness: only receive traffic once the model is loaded."""
    if getattr(request.app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    for row in body.instances:
        if len(row) != N_FEATURES:
            raise HTTPException(
                status_code=422,
                detail=f"each instance must have {N_FEATURES} features, got {len(row)}",
            )

    model_name = os.getenv("MODEL_NAME", "iris-classifier")
    t0 = time.perf_counter()
    predictions = model.predict(body.instances)
    inf_duration = time.perf_counter() - t0

    INFERENCE_LATENCY.labels(model_name=model_name).observe(inf_duration)
    BATCH_SIZE.labels(endpoint="/predict").observe(len(body.instances))
    for p in predictions:
        PREDICTIONS_COUNT.labels(
            model_name=model_name,
            predicted_class=str(int(p)),
        ).inc()

    return PredictResponse(predictions=[int(p) for p in predictions])
