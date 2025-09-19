# app/config/settings.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import ClassVar, Optional


class Settings(BaseSettings):
    # === Project ===
    PROJECT_NAME: str = "Agent Marketplace"
    DEBUG: bool = True

    # === Database ===
    DB_USER: str = Field(..., env="DB_USER")
    DB_PASSWORD: str = Field(..., env="DB_PASSWORD")
    DB_NAME: str = Field(..., env="DB_NAME")
    DB_PORT: int = Field(5432, env="DB_PORT")
    INSTANCE_CONNECTION_NAME: Optional[str] = Field(None, env="INSTANCE_CONNECTION_NAME")

    # === JWT ===
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ALGORITHM: str = Field("HS256", env="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(1440, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # === Email (SMTP) ===
    MAIL_USERNAME: str = Field(..., env="MAIL_USERNAME")
    MAIL_PASSWORD: str = Field(..., env="MAIL_PASSWORD")
    MAIL_FROM: str = Field(..., env="MAIL_FROM")
    MAIL_PORT: int = Field(587, env="MAIL_PORT")
    MAIL_SERVER: str = Field("smtp.gmail.com", env="MAIL_SERVER")
    MAIL_FROM_NAME: str = Field("Agent Marketplace", env="MAIL_FROM_NAME")
    COMPANY_NAME: str = Field("Your Company", env="COMPANY_NAME")
    SUPPORT_EMAIL: Optional[str] = Field(None, env="SUPPORT_EMAIL")
    FRONTEND_URL: Optional[str] = Field(None, env="FRONTEND_URL")
    MAIL_SSL_TLS: bool = False
    MAIL_STARTTLS: bool = True

    # === Razorpay / Payments ===
    RAZORPAY_KEY_ID: Optional[str] = Field(None, env="RAZORPAY_KEY_ID")
    RAZORPAY_KEY_SECRET: Optional[str] = Field(None, env="RAZORPAY_KEY_SECRET")
    RAZORPAY_WEBHOOK_SECRET: Optional[str] = Field(None, env="RAZORPAY_WEBHOOK_SECRET")

    # === SQLAlchemy ===
    SQLALCHEMY_ECHO: ClassVar[bool] = True

    # === Computed ===
    @property
    def DATABASE_URL(self) -> str:
        if self.INSTANCE_CONNECTION_NAME:
            return (
                f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@/"
                f"{self.DB_NAME}?host=/cloudsql/{self.INSTANCE_CONNECTION_NAME}"
            )
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@postgres:{self.DB_PORT}/{self.DB_NAME}"
        )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Singleton settings instance
settings = Settings()
