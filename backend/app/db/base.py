from typing import Any
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy_continuum import make_versioned, versioning_manager

# Enable versioning before models are defined
make_versioned(user_cls=None)

class Base(DeclarativeBase):
    id: Any
    __name__: str

    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()

# Bind versioning to this custom Base
versioning_manager.options['base_classes'] = (Base,)

# Import all models for Alembic to detect them
# from app.models.user import User
# from app.models.account_tier import AccountTier
# from app.models.country import Country
# from app.models.industry import Industry
# from app.models.role import Role
# from app.models.social_identity import SocialIdentity

# FINALIZE all mappings (this is CRUCIAL)
from sqlalchemy.orm import configure_mappers
configure_mappers()
