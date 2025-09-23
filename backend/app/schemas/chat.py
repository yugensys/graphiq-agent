from pydantic import BaseModel

class Message(BaseModel):
    sender: str = "user"
    message: str