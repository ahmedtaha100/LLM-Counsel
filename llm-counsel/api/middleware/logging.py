import logging
import time
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("llm_counsel")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = (time.perf_counter() - start) * 1000

        logger.info(
            "%s %s %d %.2fms",
            request.method,
            request.url.path,
            response.status_code,
            duration,
        )

        response.headers["X-Response-Time-Ms"] = f"{duration:.2f}"
        return response
