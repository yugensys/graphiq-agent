from typing import Any, Dict, Optional, Union, List
from sqlalchemy.orm import Session
from app.core.security import get_password_hash, verify_password
from app.models.user import User
from app.models.industry import Industry
from app.models.role import Role
from app.models.account_tier import AccountTier
from app.schemas.user import UserCreate
import logging
from fastapi import HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from typing import List
class CRUDUser:
    def get_by_email(self, db: Session, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email).first()

    def create_user(self, db: Session, user_data: dict) -> User:
        user = User(**user_data)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user


class CRUDIndustry:
    def get_by_name(self, db: Session, name: str) -> Optional[Industry]:
        return db.query(Industry).filter(Industry.name.ilike(name)).first()

    def create(self, db: Session, name: str) -> Industry:
        industry = Industry(name=name)
        db.add(industry)
        db.commit()
        db.refresh(industry)
        return industry


class CRUDRole:
    def get_by_name(self, db: Session, name: str) -> Optional[Role]:
        return db.query(Role).filter(Role.role_type.ilike(name)).first()


class CRUDAccountTier:
    def get_default_free_tier(self, db: Session) -> Optional[AccountTier]:
        return db.query(AccountTier).filter(AccountTier.account_type == "free").first()


user_crud = CRUDUser()
industry_crud = CRUDIndustry()
role_crud = CRUDRole()
tier_crud = CRUDAccountTier()
