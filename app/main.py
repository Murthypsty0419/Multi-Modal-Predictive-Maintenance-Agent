"""
Entry point — `python -m app` or `uvicorn app.main:app`.
"""

import logging

import uvicorn

from app.api.routes import app  # noqa: F401
from app.config import settings

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
)

if __name__ == "__main__":
    uvicorn.run(
        "app.api.routes:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
