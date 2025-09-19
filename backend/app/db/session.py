from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy_continuum import make_versioned
from app.config.settings import settings

# ✅ Enable versioning first
#make_versioned(user_cls=None)

# ✅ Declare base BEFORE engine
#Base = declarative_base()

# ✅ Create engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=32,
    max_overflow=64,
    echo=settings.SQLALCHEMY_ECHO
)

# ✅ Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ✅ Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
