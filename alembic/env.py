import sys
from pathlib import Path
from logging.config import fileConfig
from sqlalchemy import pool
from alembic import context

# Add your project directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

# THIS IS THE KEY CHANGE: Import your WORKING engine and Base
from backend.app.models.database import engine, Base

# IMPORT YOUR MODELS TO REGISTER THEM WITH Base.metadata
from backend.app.models.database import Personnel, DetectionLog, PersonnelImage, Room

# this is the Alembic Config object, which provides access to the .ini file
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set your metadata
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = str(engine.url)
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    with engine.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()