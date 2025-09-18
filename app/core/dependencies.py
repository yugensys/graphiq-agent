# app/core/dependencies.py

from typing import Generator
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer
from app.db.session import SessionLocal
from app.models.user import User
from app.services.auth_service import auth_service
from app.services.email_service import EmailService


# === OAuth2 Token Flow ===
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# === Core DB Dependency ===
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# === Authentication & Role Access ===
def get_current_active_user(
    current_user: User = Depends(auth_service.get_current_user),
    db: Session = Depends(get_db)
) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def get_admin_user(
    current_user: User = Depends(auth_service.get_current_user),
    db: Session = Depends(get_db)
) -> User:
    """
    Ensure the current user is an admin.
    """
    if not current_user.role or current_user.role.role_type.upper() != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this resource"
        )
    return current_user


def require_role(expected_role: str):
    """
    Dynamic dependency to check role type (e.g., ADMIN, DEVELOPER, etc.)
    Usage: Depends(require_role("DEVELOPER"))
    """
    def checker(current_user: User = Depends(auth_service.get_current_user)) -> User:
        if not current_user.role or current_user.role.role_type.upper() != expected_role.upper():
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return current_user
    return checker


# === Service Dependencies (Singleton-style) ===

def get_email_service() -> EmailService:
    return EmailService()


