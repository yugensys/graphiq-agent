from sqlalchemy import Column, Boolean

class StatusMixin:
    is_active = Column(Boolean, default=True)