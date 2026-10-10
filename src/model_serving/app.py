"""FastAPI app: /health (liveness), /ready (readiness), /predict."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from model_serving.model import load_model

logger = logging.getLogger("model_serving")

N_FEATURES = 4  # iris: sepal/petal length and width


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup, not per request: loading can take seconds and
    # would blow the latency budget on every call.
    logger.info("Loading model...")
    app.state.model = load_model()
    logger.info("Model loaded.")
    yield
    app.state.model = None


app = FastAPI(title="model-serving", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: the process is up. Must not depend on the model or other
    services, otherwise a slow dependency makes Kubernetes restart the pod."""
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> dict[str, str]:
    """Readiness: only receive traffic once the model is loaded."""
    if getattr(request.app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


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

    # Plain `def` (not async): FastAPI runs it in a threadpool, so the
    # CPU-bound predict() does not block the event loop.
    predictions = model.predict(body.instances)
    return PredictResponse(predictions=[int(p) for p in predictions])
