from fastapi import APIRouter
from app.api.v1 import chat
# from app.core import file_upload_download
api_router = APIRouter()

api_router.include_router(chat.router, tags=["chat"])