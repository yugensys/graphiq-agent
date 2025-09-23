from fastapi import APIRouter, HTTPException
import httpx
from pydantic import BaseModel
import os
from app.schemas.chat import Message
router = APIRouter(prefix="/chat",tags=["chat"])

RASA_SERVER_URL = os.getenv("RASA_SERVER_URL", "http://rasa:5005")
print(f"Using RASA_SERVER_URL: {RASA_SERVER_URL}")  # Debug log


@router.post("/chat")
async def chat(message: Message):
    """
    Send a message to the Rasa server and get the bot's response.
    """
    try:
        print(f"Sending message to Rasa: {message.message}")  
        async with httpx.AsyncClient(timeout=30.0) as client:  
            # Send the message to Rasa
            response = await client.post(
                f"{RASA_SERVER_URL}/webhooks/rest/webhook",
                json={
                    "sender": message.sender,
                    "message": message.message
                }
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Error communicating with Rasa server"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
