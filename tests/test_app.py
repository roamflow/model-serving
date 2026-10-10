import pytest
from fastapi.testclient import TestClient

from model_serving.app import app


@pytest.fixture
def client():
    # `with` triggers the lifespan: the model is loaded before the first
    # request and released on exit. Without `with`, startup never runs.
    with TestClient(app) as c:
        yield c


def test_health_is_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_ready_once_model_loaded(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready"}


def test_predict_returns_one_label_per_instance(client):
    r = client.post(
        "/predict",
        json={"instances": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]},
    )
    assert r.status_code == 200
    # Well-separated iris samples: setosa (0) and virginica (2).
    assert r.json() == {"predictions": [0, 2]}


def test_predict_wrong_feature_count_is_422(client):
    r = client.post("/predict", json={"instances": [[1.0, 2.0, 3.0]]})
    assert r.status_code == 422
    assert "4 features" in r.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing field
        {"instances": []},  # below min_length
        {"instances": "x"},  # wrong type
        {"instances": [["a", "b", "c", "d"]]},  # non-numeric
    ],
)
def test_predict_invalid_payload_is_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_not_ready_before_startup():
    # No `with`: lifespan has not run, so there is no model.
    c = TestClient(app)
    app.state.model = None
    assert c.get("/health").status_code == 200  # liveness unaffected
    assert c.get("/ready").status_code == 503
    assert c.post("/predict", json={"instances": [[1, 2, 3, 4]]}).status_code == 503
