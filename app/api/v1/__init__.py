from fastapi import APIRouter
from app.api.v1 import auth
# from app.core import file_upload_download
api_router = APIRouter()

api_router.include_router(auth.router, tags=["authentication"])