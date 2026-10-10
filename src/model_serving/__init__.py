import os

import uvicorn


def main() -> None:
    """Console entrypoint: `uv run model-serving`."""
    uvicorn.run(
        "model_serving.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
