import urllib
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Index, Boolean, ForeignKey

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
from datetime import datetime, timedelta
import os
import cv2
import numpy as np
from pathlib import Path
import uuid
 
# ==================== CONFIGURATION ====================
FACE_STORAGE_DIR = Path("saved_media")
FACE_STORAGE_DIR.mkdir(exist_ok=True, parents=True)

# Use localhost - this ALWAYS works on the same machine
params = urllib.parse.quote_plus(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost\\sql,14330;"
    "DATABASE=AI_DB;"
    "UID=sa;"
    "PWD=Asd@12345;"
    "TrustServerCertificate=yes;"
    "Connection Timeout=30;"
)

DATABASE_URL = f"mssql+pyodbc:///?odbc_connect={params}"


# server database
# params = urllib.parse.quote_plus(
#     "DRIVER={ODBC Driver 17 for SQL Server};"
#     "SERVER=192.168.110.13,14330;"  # Remote server IP with port
#     "DATABASE=HR_DB_20;"             # Database name changed to HR_DB_20
#     "UID=sa;"
#     "PWD=Asd@12345;"
#     "TrustServerCertificate=yes;"     # Keep this for self-signed certs
#     "Encrypt=yes;"                    # Added encryption
#     "Connection Timeout=30;"
# )

# DATABASE_URL = f"mssql+pyodbc:///?odbc_connect={params}"




# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10
)

# Base class for models
Base = declarative_base()

# ==================== DATABASE MODEL ====================
class Personnel(Base):
    __tablename__ = "Personnel"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fname = Column(String(100), nullable=False)
    lname = Column(String(100), nullable=False)
    national_code = Column(String(20), unique=True, nullable=False, index=True)
    staff = Column(Boolean, default=False)
    department = Column(String(200), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    
    # Optional: Add relationship to DetectionLogs if you want to link them
    detections = relationship("DetectionLog", back_populates="personnel")
    
    __table_args__ = (
        Index('idx_national_code', 'national_code'),
        Index('idx_name', 'fname', 'lname'),
    )


class DetectionLog(Base):
    __tablename__ = "DetectionLogs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    person = Column(String(200), nullable=False)
    confidence = Column(Float, nullable=True)
    detection_time = Column(DateTime, server_default=func.now())
    face_image_path = Column(String(512), nullable=True)
    video_path = Column(String(512), nullable=True)
    camera_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    personnel_id = Column(Integer, ForeignKey('Personnel.id'), nullable=True)
    personnel = relationship("Personnel", back_populates="detections")


    __table_args__ = (
        Index('idx_detection_time', 'detection_time'),
        Index('idx_person', 'person'),
    )

# Create tables
# Base.metadata.create_all(bind=engine)

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
def save_detection(person: str, confidence: float, face_image: np.ndarray = None, camera_id: int = None, frames_to_save:list=None):
    """Save a detection with optional face image, with 60-second cooldown per person.
    If new detection has higher confidence, replace the old one.
    Returns: log_id if saved, None if skipped due to lower confidence or error"""
    
    db = SessionLocal()
    face_path = None
    log_id = None
    
    try:
        # Look for recent detection of the same person (within 60 seconds)
        sixty_seconds_ago = datetime.now() - timedelta(seconds=60)
        
        recent_detection = db.query(DetectionLog).filter(
            DetectionLog.person == person,
            DetectionLog.detection_time >= sixty_seconds_ago
        ).order_by(DetectionLog.detection_time.desc()).first()
        
        if recent_detection:
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
                db.flush()  # commit delete in memory
                if old_face_path and os.path.exists(old_face_path):
                    try:
                        os.remove(old_face_path)
                        print(f"🗑️ Deleted old face image: {old_face_path}")
                    except:
                        pass
                print(f"♻️  Replacing old detection for '{person}' with higher confidence {confidence:.2f}")
                if old_video_path and os.path.exists(old_video_path):
                    try:
                        os.remove(old_video_path)
                        print(f"🗑️ Deleted old face video: {old_video_path}")
                    except:
                        pass
                print(f"♻️  Replacing old detection for '{person}' with higher confidence {confidence:.2f}")
        

        video_path = None
        if frames_to_save:
                                            
            video_dir = get_today_subdirectory('video') 
            video_filename = f"{person}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}.mp4"
            video_path = str(video_dir / video_filename)

  
            height, width, _ = frames_to_save[0].shape
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height))

            for f in frames_to_save:
                out.write(f)
            out.release()
            print(f"🎬 Saved video clip: {video_path}")



        # Save new detection
        detection = DetectionLog(
            person=person,
            confidence=float(confidence),
            detection_time=datetime.now(),
            face_image_path=None,  # will update if face image is saved
            camera_id=camera_id,
            video_path=video_path,
        )

  

        db.add(detection)
        db.flush()  # assign ID without committing
        log_id = detection.id
        
        # Save face image if provided
        if face_image is not None and face_image.size > 0:
            try:
                daily_dir = get_today_subdirectory('image')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                safe_person = "".join(c if c.isalnum() or c in " _-" else "_" for c in person)[:50]
                unique_id = str(uuid.uuid4())[:6]
                confidence_str = f"{confidence:.2f}".replace(".", "p")
                filename = f"{safe_person}_{confidence_str}_{timestamp}_{unique_id}.jpg"
                face_path = str(daily_dir / filename)
                
                cv2.imwrite(face_path, face_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                detection.face_image_path = face_path
                print(f"✅ Face image saved: {face_path}")
            except Exception as img_error:
                print(f"⚠️ Image save error: {img_error}")
                import traceback
                traceback.print_exc()
                face_path = None
        
        db.commit()
        print(f"✅ Saved detection: {person} ({confidence:.2f}) with ID: {log_id}")
        return log_id
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        import traceback
        traceback.print_exc()
        if face_path and os.path.exists(face_path):
            try:
                os.remove(face_path)
                print(f"🗑️ Cleaned up orphaned file: {face_path}")
            except:
                pass
        db.rollback()
        return None
    finally:
        db.close()


# ==================== WRAPPER FUNCTION ====================
def save_detection_with_face(
        person: str, 
        confidence: float, 
        face_image: np.ndarray = None,
        camera_id: int = None, 
        frames_to_save:list=None
        ):
    """
    Helper function to save detection with face image
    Returns the saved log ID or None if failed/skipped
    """
    try:
        log_id = save_detection(person=person, confidence=confidence, face_image=face_image, camera_id=camera_id, frames_to_save=frames_to_save)
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
            
            
          
          