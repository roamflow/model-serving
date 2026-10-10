# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.12

# Pinned uv binary, copied in below. Multi-arch (amd64 + arm64).
FROM ghcr.io/astral-sh/uv:0.11.32 AS uv

# ---- builder: resolve and install dependencies into /app/.venv ----
FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Set to "true" to bake in the optional mlflow extra (needed to load models
# from the MLflow registry, MODEL_URI=models:/...).
ARG WITH_MLFLOW=false

# Layer 1: dependencies only. Re-runs only when pyproject.toml / uv.lock
# change, so code edits do not reinstall every package.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    EXTRA=""; [ "$WITH_MLFLOW" = "true" ] && EXTRA="--extra mlflow"; \
    uv sync --locked --no-dev --no-install-project $EXTRA

# Layer 2: the project itself, installed (not editable) into the venv.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    EXTRA=""; [ "$WITH_MLFLOW" = "true" ] && EXTRA="--extra mlflow"; \
    uv sync --locked --no-dev --no-editable $EXTRA

# ---- runtime: no uv, no compilers, no source tree ----
FROM python:${PYTHON_VERSION}-slim AS runtime

# Non-root, fixed UID so Kubernetes securityContext (runAsNonRoot) can verify it.
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home app

COPY --from=builder --chown=app:app /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000

USER 10001:10001
EXPOSE 8000

# Informational for plain Docker; Kubernetes ignores HEALTHCHECK and uses probes.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=2)"]

CMD ["model-serving"]
