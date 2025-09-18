# app/core/middleware.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
import time

# Setup logger
logger = logging.getLogger("middleware")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def add_middlewares(app: FastAPI):
    # === CORS Middleware ===
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Change in prod (restrict to frontend domain)
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # === Logging Middleware ===
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = round((time.time() - start_time) * 1000, 2)

        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} - {process_time}ms"
        )

        return response
