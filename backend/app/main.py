# app/main.py

import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from sqlalchemy.exc import SQLAlchemyError
from app.api.v1 import api_router

from app.config.settings import settings
#from app.api import api_router
from app.core.middleware import add_middlewares
from app.core.exceptions import (
    http_exception_handler,
    validation_exception_handler,
    unhandled_exception_handler,
)

from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.db.base import Base
from app.db.session import engine

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === DB Initialization ===
def create_tables():
    try:
        #Base.metadata.create_all(bind=engine)
        logger.info(" Database tables created.")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")


# === Background Scheduler ===
#scheduler = BackgroundScheduler()


# === Lifespan Context ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        with engine.connect() as conn:
            logger.info(" Database connected successfully.")
    except SQLAlchemyError as e:
        logger.error(f" Startup DB error: {e}")
        raise RuntimeError(f"Startup failed: {e}")

    create_tables()
    #schedule(scheduler)
    yield  # <-- App runs here
    #shutdown(scheduler)
    logger.info("Application shutdown complete.")


# === FastAPI App Setup ===
app = FastAPI(
    title="Agent Marketplace",
    description="Modular Agentic AI Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Middlewares & Exception Handlers ===
add_middlewares(app)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# === Routers ===
app.include_router(api_router, prefix="/api/v1")

# === Utility Routes ===
@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/")
def root():
    return {"message": "Welcome to the Agent Marketplace FastAPI App!"}
