from sqlalchemy import Column, Integer, String, Enum, Date, ForeignKey,Boolean,Enum as SAEnum
from sqlalchemy.orm import relationship
from app.db.base import Base
from app.models.mixins import TimestampMixin
from sqlalchemy import Text, Float
class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    hf_endpoint = Column(String(255), nullable=False)
    pricing = Column(Float, nullable=True)