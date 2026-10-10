"""Model loading.

Two sources, chosen by the MODEL_URI environment variable:

* ``MODEL_URI`` set  -> load a scikit-learn model from the MLflow registry,
  e.g. ``models:/iris-classifier/Production`` (wired up in Step 3).
* ``MODEL_URI`` unset -> train a tiny iris classifier in-process, so the
  service runs anywhere without needing a tracking server.
"""

import os

from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def _train_fallback_model() -> BaseEstimator:
    x, y = load_iris(return_X_y=True)
    return LogisticRegression(max_iter=200).fit(x, y)


def load_model() -> BaseEstimator:
    model_uri = os.getenv("MODEL_URI")
    if not model_uri:
        return _train_fallback_model()

    # Imported lazily: mlflow is heavy (seconds of import time, 100s of MB),
    # so it is an optional extra: `uv sync --extra mlflow`.
    try:
        import mlflow.sklearn
    except ImportError as exc:
        raise RuntimeError(
            "MODEL_URI is set but mlflow is not installed; "
            "install it with the 'mlflow' extra (uv sync --extra mlflow)"
        ) from exc

    return mlflow.sklearn.load_model(model_uri)
