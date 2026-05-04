import urllib
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Index, Boolean, ForeignKey, NVARCHAR, Unicode, Table

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
from datetime import datetime, timedelta
import os
import cv2
import numpy as np
from pathlib import Path
import uuid
from ..video_utils import save_video_with_ffmpeg, save_fallback_opencv
import asyncio
from typing import Optional
# ==================== CONFIGURATION ====================
FACE_STORAGE_DIR = Path("saved_media")
FACE_STORAGE_DIR.mkdir(exist_ok=True, parents=True)

# Use localhost - this ALWAYS works on the same machine
# params = urllib.parse.quote_plus(
#     "DRIVER={ODBC Driver 17 for SQL Server};"
#     "SERVER=localhost\\sql,14330;"
#     "DATABASE=AI_DB;"
#     "UID=sa;"
#     "PWD=Asd@12345;"
#     "TrustServerCertificate=yes;"
#     "Connection Timeout=30;"
# )

# DATABASE_URL = f"mssql+pyodbc:///?odbc_connect={params}"

# server database
# params = urllib.parse.quote_plus(
#     "DRIVER={ODBC Driver 17 for SQL Server};"
#     "SERVER=192.168.110.13,14330;"  # Remote server IP with port
#     "DATABASE=HR_DB_20;"             # 
#     "UID=sa;"
#     "PWD=Asd@12345;"
#     "TrustServerCertificate=yes;"     # Keep this for self-signed certs
#     "Encrypt=yes;"                    # Added encryption
#     "Connection Timeout=30;"
# )

# DATABASE_URL = f"mssql+pyodbc:///?odbc_connect={params}"


# Create engine
# engine = create_engine(
#     DATABASE_URL,
#     echo=False,
#     pool_pre_ping=True,
#     pool_size=5,
#     max_overflow=10
# )


import urllib
from sqlalchemy import create_engine
import os

# SQLite configuration - creates file in current directory
# You can change the path as needed
SQLITE_DB_PATH = "ai_database.db"  # Database file name
DATABASE_URL = f"sqlite:///{SQLITE_DB_PATH}"

# For absolute path (e.g., in user directory):
# SQLITE_DB_PATH = os.path.join(os.path.expanduser("~"), "AI_DB", "ai_database.db")
# os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
# DATABASE_URL = f"sqlite:///{SQLITE_DB_PATH}"

# For in-memory database (data lost when program ends):
# DATABASE_URL = "sqlite:///:memory:"

# Create engine with SQLite-specific settings
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True to see SQL queries
    connect_args={"check_same_thread": False}  # Needed for multi-threading
)





# Base class for models
Base = declarative_base()


# ==================== ASSOCIATION TABLE (Many-to-Many) ====================
# This table links Personnel and Rooms
personnel_room_association = Table(
    'personnel_room_association',
    Base.metadata,
    Column('personnel_id', Integer, ForeignKey('Personnel.id'), primary_key=True),
    Column('room_id', Integer, ForeignKey('Rooms.id'), primary_key=True),
    Column('assigned_at', DateTime, default=datetime.now),
    Column('assigned_by', String(100), nullable=True)  # Who granted access
)


class Personnel(Base):
    __tablename__ = "Personnel"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fname = Column(NVARCHAR(100), nullable=False)  # Explicit NVARCHAR for Persian text
    lname = Column(NVARCHAR(100), nullable=False)  # Explicit NVARCHAR for Persian text
    # national_code = Column(String(20), unique=True, nullable=False, index=True)  # Keep as String for numbers
    national_code = Column(String(20), nullable=False, index=True)  # Keep as String for numbers
    staff = Column(Boolean, default=False)
    department = Column(NVARCHAR(200), nullable=True)  # Explicit NVARCHAR for Persian text
    # created_at = Column(DateTime, server_default=func.now()) # sql server
    created_at = Column(DateTime, default=datetime.now)
    images = relationship(
        "PersonnelImage", 
        back_populates="personnel",
        cascade="all, delete-orphan"  # This enables cascade delete
    )
    rooms = relationship("Room", secondary=personnel_room_association, back_populates="personnel")

    @property
    def primary_image(self) -> Optional[str]:
        """Get the primary image URL"""
        primary = next((img for img in self.images if img.is_primary), None)
        if primary:
            return primary.image_url
        # Return first image if exists
        return self.images[0].image_url if self.images else None
    
    
class Room(Base):
    __tablename__ = "Rooms"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    room_number = Column(String(50), nullable=False, unique=True, index=True)  # e.g., "101", "A-202"
    room_name = Column(NVARCHAR(200), nullable=True)  # e.g., "Conference Room", "Office A"
    room_type = Column(NVARCHAR(100), nullable=True)  # e.g., "office", "conference", "lab"
    capacity = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=True)
    description = Column(NVARCHAR(500), nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    personnel = relationship("Personnel", secondary=personnel_room_association, back_populates="rooms")
    # access_logs = relationship("RoomAccessLog", back_populates="room", cascade="all, delete-orphan")



# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class DetectionLog(Base):
    __tablename__ = "DetectionLogs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ref_img_id = Column(Integer, nullable=True )
    person = Column(String(200), nullable=False)
    area = Column(String(200), nullable=True)
    confidence = Column(Float, nullable=True)
    detection_time = Column(DateTime, default=datetime.now)
    face_image_path = Column(String(512), nullable=True)
    video_path = Column(String(512), nullable=True)
    camera_id = Column(Integer, nullable=True)
    # created_at = Column(DateTime, server_default=func.now())
    created_at = Column(DateTime, default=datetime.now)
    room_id = Column(Integer, ForeignKey('Rooms.id'), nullable=True)
    access_granted = Column(Boolean, nullable=False, default=True)
    room = relationship("Room")

            

class PersonnelImage(Base):
    __tablename__ = "PersonnelImages"  # or whatever your table name is
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_url = Column(Unicode(500), nullable=False)  # Changed to Unicode for paths that might have Persian chars
    personnel_id = Column(Integer, ForeignKey('Personnel.id'), nullable=False)
    # uploaded_at = Column(DateTime, server_default=func.now())
    is_primary = Column(Boolean, default=False)
    uploaded_at = Column(DateTime, default=datetime.now)
    
    personnel = relationship("Personnel", back_populates="images")
