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


# class RoomAccessLog(Base):
#     __tablename__ = "RoomAccessLogs"
    
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     personnel_id = Column(Integer, ForeignKey('Personnel.id'), nullable=False)
#     room_id = Column(Integer, ForeignKey('Rooms.id'), nullable=False)
    
#     access_time = Column(DateTime, default=datetime.now, index=True)
#     access_granted = Column(Boolean, nullable=False)
#     denial_reason = Column(String(50), nullable=True)  # "no_access" or "low_confidence"
#     face_confidence = Column(Float, nullable=True)  # For audit trail
    
#     # Relationships
#     personnel = relationship("Personnel")
#     room = relationship("Room")


# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class DetectionLog(Base):
    __tablename__ = "DetectionLogs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ref_img_id = Column(Integer, nullable=True )
    person = Column(String(200), nullable=False)
    area = Column(String(200), nullable=True)
    confidence = Column(Float, nullable=True)
    detection_time = Column(DateTime, server_default=func.now())
    face_image_path = Column(String(512), nullable=True)
    video_path = Column(String(512), nullable=True)
    camera_id = Column(Integer, nullable=True)
    # created_at = Column(DateTime, server_default=func.now())
    created_at = Column(DateTime, default=datetime.now)
    room_id = Column(Integer, ForeignKey('Rooms.id'), nullable=True)
    access_granted = Column(Boolean, nullable=True)  # ADD THIS FIELD

    
    room = relationship("Room")

<<<<<<< Updated upstream
# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ==================== HELPER FUNCTIONS ====================
def get_today_subdirectory(file_type) -> Path:
    """Create and return today's date-based subdirectory"""
    today = datetime.now().strftime("%Y-%m-%d")
    daily_dir = FACE_STORAGE_DIR / today / file_type
    daily_dir.mkdir(exist_ok=True, parents=True)
    return daily_dir

# ==================== MAIN SAVE FUNCTION ====================


def save_detection(
    person: str, 
    area:str, 
    confidence: float, 
    face_image: np.ndarray = None, 
    ref_img_id :int=None,
    camera_id: int = None, 
    frames_to_save: list = None,  
    face_to_save: list = None,
    save_video: bool=False
    ):
    """Save a detection with optional face image, with 60-second cooldown per person.
    If new detection has higher confidence, replace the old one.
    Returns: log_id if saved, None if skipped due to lower confidence or error"""
    
    db = SessionLocal()
    face_path = None
    log_id = None
    video_path = None
    
    try:
        # Look for recent detection of the same person (within 60 seconds)
        sixty_seconds_ago = datetime.now() - timedelta(seconds=0)

        recent_detection = db.query(DetectionLog).filter(
            DetectionLog.person == person,
            DetectionLog.area == area,
            DetectionLog.detection_time >= sixty_seconds_ago
        ).order_by(DetectionLog.detection_time.desc()).first()
        
        if recent_detection  :

            if recent_detection.confidence >= confidence:
                # Skip because existing detection is stronger
                time_diff = (datetime.now() - recent_detection.detection_time).total_seconds()
                print(f"⏱️  Skipping detection for '{person}' - last detection {time_diff:.1f}s ago with higher confidence ({recent_detection.confidence:.2f})")
                return None
            else:
                # Delete old detection
                old_face_path = recent_detection.face_image_path
                old_video_path = recent_detection.video_path

                db.delete(recent_detection)
                db.flush()
                
                # Delete old files
                if old_face_path and os.path.exists(old_face_path):
                    try:
                        os.remove(old_face_path)
                        print(f"🗑️ Deleted old face image: {old_face_path}")
                    except Exception as e:
                        print(f"⚠️ Could not delete old face image: {e}")
                
                if old_video_path and os.path.exists(old_video_path):
                    try:
                        os.remove(old_video_path)
                        print(f"🗑️ Deleted old video: {old_video_path}")
                    except Exception as e:
                        print(f"⚠️ Could not delete old video: {e}")
                
                print(f"♻️  Replacing old detection for '{person}' with higher confidence {confidence:.2f}")

        # Save video if frames provided
        if save_video and  frames_to_save and len(frames_to_save) > 0:
            video_dir = get_today_subdirectory('video')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            safe_person = "".join(c if c.isalnum() or c in " _-" else "_" for c in person)[:30]
            video_filename = f"{safe_person}_{timestamp}_{unique_id}.mp4"
            video_path = str(video_dir / video_filename)
            
            print(f"🎬 Saving video with {len(frames_to_save)} frames...")
            
            # Try FFmpeg first (best quality and web compatibility)
            
            success = save_video_with_ffmpeg(
                frames=frames_to_save,
                output_path=video_path,
                fps=15,
                quality="medium",
                for_web=True
            )
            
            # Fallback to OpenCV if FFmpeg fails
            if not success:
                print("⚠️ FFmpeg failed, trying OpenCV fallback...")
                success = save_fallback_opencv(frames_to_save, video_path, fps=15)
            
            if not success:
                print("❌ Failed to save video with both methods")
                video_path = None
            else:
                print(f"✅ Video saved successfully: {video_path}")
        if  face_to_save and len(face_to_save) > 0:
            video_dir = get_today_subdirectory('unknown_faces')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            safe_person = "".join(c if c.isalnum() or c in " _-" else "_" for c in person)[:30]
            video_filename = f"{safe_person}_{timestamp}_{unique_id}.mp4"
            video_path = str(video_dir / video_filename)
            
            print(f"🎬 Saving video with {len(face_to_save)} frames...")
            
            # Try FFmpeg first (best quality and web compatibility)
            
            success = save_video_with_ffmpeg(
                frames=face_to_save,
                output_path=video_path,
                fps=15,
                quality="medium",
                for_web=True
            )
            
            # Fallback to OpenCV if FFmpeg fails
            if not success:
                print("⚠️ FFmpeg failed, trying OpenCV fallback...")
                success = save_fallback_opencv(face_to_save, video_path, fps=15)
            
            if not success:
                print("❌ Failed to save video with both methods")
                video_path = None
            else:
                print(f"✅ Video saved successfully: {video_path}")

        detection = DetectionLog(
            person=person,
            confidence=float(confidence),
            detection_time=datetime.now(),
            face_image_path=None,
            camera_id=camera_id,
            video_path=video_path,
            ref_img_id=ref_img_id, 
            area= area
        )

        db.add(detection)
        db.flush()
        log_id = detection.id
        
        # Save face image if provided
        # if face_image is not None and face_image.size > 0:
            # try:
        daily_dir = get_today_subdirectory('image')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe_person = "".join(c if c.isalnum() or c in " _-" else "_" for c in person)[:50]
        unique_id = str(uuid.uuid4())[:6]
        confidence_str = f"{confidence:.2f}".replace(".", "p")
        filename = f"{safe_person}_{confidence_str}_{timestamp}_{unique_id}.jpg"
        face_path = str(daily_dir / filename)
        
        # Save with high quality
        cv2.imwrite(face_path, face_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        detection.face_image_path = face_path
        print(f"✅ Face image saved: {face_path}")
            # except Exception as img_error:
            #     print(f"⚠️ Image save error: {img_error}")
            #     import traceback
            #     traceback.print_exc()
            #     face_path = None
        
        db.commit()
        print(f"✅ Saved detection: {person} ({confidence:.2f}) with ID: {log_id}")
        return log_id
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up orphaned files
        if face_path and os.path.exists(face_path):
            try:
                os.remove(face_path)
                print(f"🗑️ Cleaned up orphaned face: {face_path}")
            except:
                pass
        
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"🗑️ Cleaned up orphaned video: {video_path}")
            except:
                pass
        
        db.rollback()
        return None
    finally:
        db.close()


# ==================== WRAPPER FUNCTION ====================
def save_detection_with_face(
        person: str, 
        area: str, 
        confidence: float, 
        face_image: np.ndarray = None,
        camera_id: int = None, 
        ref_img_id:int=None,
        frames_to_save:list=None,
        face_to_save:list=None,
        save_video:bool=False
        ):
    """
    Helper function to save detection with face image
    Returns the saved log ID or None if failed/skipped
    """
    try:
        log_id = save_detection(
            person=person,
            area=area,
            confidence=confidence, 
            face_image=face_image, 
            ref_img_id=ref_img_id, 
            camera_id=camera_id, 
            frames_to_save=frames_to_save,
            face_to_save=face_to_save,
            save_video=save_video,
            
            )
        return log_id
    except Exception as e:
        print(f"❌ Failed to save detection for {person}: {e}")
        return None

# ==================== DEPENDENCY ====================
def get_db():
    """FastAPI dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==================== QUERY FUNCTIONS ====================
def get_recent_detections(limit: int = 50, db: Session = None):
    """Get recent detection logs"""
    if db is None:
        db = SessionLocal()
        close_db = True
    else:
        close_db = False
    
    try:
        logs = db.query(DetectionLog)\
            .order_by(DetectionLog.detection_time.desc())\
            .limit(limit)\
            .all()
        return logs
    finally:
        if close_db:
            db.close()

def get_detection_by_id(log_id: int, db: Session = None):
    """Get detection log by ID"""
    if db is None:
        db = SessionLocal()
        close_db = True
    else:
        close_db = False
    
    try:
        log = db.query(DetectionLog)\
            .filter(DetectionLog.id == log_id)\
            .first()
        return log
    finally:
        if close_db:
            db.close()
=======
>>>>>>> Stashed changes
            

class PersonnelImage(Base):
    __tablename__ = "PersonnelImages"  # or whatever your table name is
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_url = Column(Unicode(500), nullable=False)  # Changed to Unicode for paths that might have Persian chars
    personnel_id = Column(Integer, ForeignKey('Personnel.id'), nullable=False)
    # uploaded_at = Column(DateTime, server_default=func.now())
    is_primary = Column(Boolean, default=False)
    uploaded_at = Column(DateTime, default=datetime.now)
    
    personnel = relationship("Personnel", back_populates="images")
