import sys
from pathlib import Path
from logging.config import fileConfig
from sqlalchemy import pool
from alembic import context

# Add your project directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

# THIS IS THE KEY CHANGE: Import your WORKING engine and Base
from backend.app.models.database import engine, Base

# this is the Alembic Config object, which provides access to the .ini file
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set your metadata
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    # Use the URL from your working engine for offline mode
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
    # --- THIS IS THE CRITICAL CHANGE ---
    # Use the working engine you already have from database.py
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
    
    
    
    
    
    
    
# import sys
# from pathlib import Path
# from logging.config import fileConfig
# from sqlalchemy import engine_from_config
# from sqlalchemy import pool
# from alembic import context

# # Add your project directory to Python path
# sys.path.append(str(Path(__file__).parent.parent))

# # Import your Base and all models - using the working import
# from backend.app.models.database import Base, DetectionLog

# # this is the Alembic Config object
# config = context.config

# # Interpret the config file for Python logging
# if config.config_file_name is not None:
#     fileConfig(config.config_file_name)

# target_metadata = Base.metadata

# def run_migrations_offline() -> None:
#     """Run migrations in 'offline' mode."""
#     url = config.get_main_option("sqlalchemy.url")
#     context.configure(
#         url=url,
#         target_metadata=target_metadata,
#         literal_binds=True,
#         dialect_opts={"paramstyle": "named"},
#     )

#     with context.begin_transaction():
#         context.run_migrations()

# def run_migrations_online() -> None:
#     """Run migrations in 'online' mode."""
#     connectable = engine_from_config(
#         config.get_section(config.config_ini_section, {}),
#         prefix="sqlalchemy.",
#         poolclass=pool.NullPool,
#     )

#     with connectable.connect() as connection:
#         context.configure(
#             connection=connection, target_metadata=target_metadata
#         )

#         with context.begin_transaction():
#             context.run_migrations()

# if context.is_offline_mode():
#     run_migrations_offline()
# else:
#     run_migrations_online()

    
# import sys
# from pathlib import Path
# from logging.config import fileConfig
# from sqlalchemy import engine_from_config
# from sqlalchemy import pool
# from alembic import context

# # Add your project directory to Python path
# sys.path.append(str(Path(__file__).parent.parent))
# # Import your Base and all models
# from backend.app.models.database import Base, DetectionLog, DATABASE_URL

# # this is the Alembic Config object, which provides
# # access to the values within the .ini file in use.
# config = context.config

# # Interpret the config file for Python logging.
# # This line sets up loggers basically.
# if config.config_file_name is not None:
#     fileConfig(config.config_file_name)

# # add your model's MetaData object here
# # for 'autogenerate' support
# # from myapp import mymodel
# # target_metadata = mymodel.Base.metadata
# # config.set_main_option("sqlalchemy.url", DATABASE_URL)

# target_metadata = Base.metadata

# # other values from the config, defined by the needs of env.py,
# # can be acquired:
# # my_important_option = config.get_main_option("my_important_option")
# # ... etc.


# def run_migrations_offline() -> None:
#     """Run migrations in 'offline' mode.

#     This configures the context with just a URL
#     and not an Engine, though an Engine is acceptable
#     here as well.  By skipping the Engine creation
#     we don't even need a DBAPI to be available.

#     Calls to context.execute() here emit the given string to the
#     script output.

#     """
#     url = config.get_main_option("sqlalchemy.url")
#     context.configure(
#         url=url,
#         target_metadata=target_metadata,
#         literal_binds=True,
#         dialect_opts={"paramstyle": "named"},
#     )

#     with context.begin_transaction():
#         context.run_migrations()


# def run_migrations_online() -> None:
#     """Run migrations in 'online' mode.

#     In this scenario we need to create an Engine
#     and associate a connection with the context.

#     """
#     connectable = engine_from_config(
#         config.get_section(config.config_ini_section, {}),
#         prefix="sqlalchemy.",
#         poolclass=pool.NullPool,
#     )

#     with connectable.connect() as connection:
#         context.configure(
#             connection=connection, target_metadata=target_metadata
#         )

#         with context.begin_transaction():
#             context.run_migrations()


# if context.is_offline_mode():
#     run_migrations_offline()
# else:
#     run_migrations_online()
