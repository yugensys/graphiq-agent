from sqlalchemy import Column, Integer, String, Enum, Date, ForeignKey,Boolean,Enum as SAEnum
from sqlalchemy.orm import relationship
from app.db.base import Base
from app.models.mixins import TimestampMixin

class User(Base, TimestampMixin):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, nullable=False)
    password_hash = Column(String)
    full_name = Column(String)
    company_name = Column(String)
    role_id = Column(Integer, ForeignKey('role.id'))
    industry_id = Column(Integer, ForeignKey('industry.id'))
    country_id = Column(Integer, ForeignKey('country.id'))
    profile_photo_url = Column(String)
    job_title = Column(String) 
    account_id = Column(Integer, ForeignKey('account_tier.id'))
    email_verified = Column(Boolean, default=False)
    
    created_by = Column(Integer)
    updated_by = Column(Integer)
    
    # Relationships
    role = relationship("Role", back_populates="users")
    industry = relationship("Industry", back_populates="users")
    country = relationship("Country", back_populates="users")
    account_tier = relationship("AccountTier", back_populates="users")
    social_identities = relationship("SocialIdentity", back_populates="user")
    #task_executions = relationship("TaskExecution", back_populates="user")
    #user_favorites = relationship("UserFavorite", back_populates="user")
    #agent_requests = relationship("AgentRequest", back_populates="user")
    #agent_reviews = relationship("AgentReview", back_populates="user")
    #payments = relationship("Payment", back_populates="user")