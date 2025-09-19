import logging
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from sqlalchemy.orm import configure_mappers  # ✅ Add this

configure_mappers()
# from google.cloud import logging as cloud_logging

# # Initialize Google Cloud Logging client
# cloud_logging_client = cloud_logging.Client()
# cloud_logging_client.setup_logging()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Alembic Config object
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import application modules
from app.models import *
from app.config.settings import settings
from app.db.base import Base

target_metadata = Base.metadata
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    logger.info(f"Running offline migrations with DB URL: {url}")
    context.configure(
        url=url, target_metadata=target_metadata, literal_binds=True, dialect_opts={"paramstyle": "named"}
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    logger.info("Starting online migrations...")
    
    try:
        connectable = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )
        logger.info("Database engine created successfully")

        with connectable.connect() as connection:
            logger.info("Database connection established")
            context.configure(connection=connection, target_metadata=target_metadata)

            with context.begin_transaction():
                logger.info("Running migrations now...")
                context.run_migrations()
                logger.info("Migrations completed successfully")

    except Exception as e:
        logger.error(f"Error running migrations: {e}", exc_info=True)
        raise


logger.info("Checking Alembic execution mode...")
if context.is_offline_mode():
    logger.info("Running offline migrations")
    run_migrations_offline()
else:
    logger.info("Running online migrations")
    run_migrations_online()