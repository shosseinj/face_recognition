import urllib
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Index, Boolean, ForeignKey, NVARCHAR, Unicode

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
from .database import DetectionLog, Personnel, PersonnelImage, Room, personnel_room_association
# from sqlalchemy.orm import sessionmaker
from .database import engine
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
    area: str, 
    confidence: float, 
    face_image: np.ndarray = None, 
    ref_img_id: int = None,
    camera_id: int = None, 
    frames_to_save: list = None,  
    face_to_save: list = None,
    save_video: bool = False,
    room_id: int = None,  # NEW parameter
    db: Session = None  # NEW parameter for optional session
):
    """Save a detection with optional face image, with 60-second cooldown per person.
    If new detection has higher confidence, replace the old one.
    Automatically determines if person has access to the room.
    Returns: log_id if saved, None if skipped due to lower confidence or error"""
    
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    face_path = None
    log_id = None
    video_path = None
    access_granted = None
    denial_reason = None
    
    try:
        # Look for recent detection of the same person (within 60 seconds)
        sixty_seconds_ago = datetime.now() - timedelta(seconds=60)

        recent_detection = db.query(DetectionLog).filter(
            DetectionLog.person == person,
            DetectionLog.area == area,
            DetectionLog.detection_time >= sixty_seconds_ago
        ).order_by(DetectionLog.detection_time.desc()).first()
        
        if recent_detection and not('Unknown' in person):
            if recent_detection.confidence >= confidence:
                # Skip because existing detection is stronger
                time_diff = (datetime.now() - recent_detection.detection_time).total_seconds()
                print(f"⏱️  Skipping detection for '{person}' - last detection {time_diff:.1f}s ago with higher confidence ({recent_detection.confidence:.2f})")
                if close_db:
                    db.close()
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

        # ==================== CHECK ROOM ACCESS ====================
        # Determine if the person has access to the room (if room_id is provided)
        if room_id and not('Unknown' in person):
            from .database import Personnel, personnel_room_association
            
            # Try to find personnel by first and last name
            # Split person string (assumes format: "FirstName LastName")
            name_parts = person.split(' ', 1)
            if len(name_parts) >= 2:
                first_name = name_parts[0]
                last_name = name_parts[1]
                
                # Find personnel with matching name
                personnel = db.query(Personnel).filter(
                    Personnel.fname == first_name,
                    Personnel.lname == last_name
                ).first()
                
                if personnel:
                    # Check if personnel has access to this room
                    access = db.query(personnel_room_association).filter(
                        personnel_room_association.c.personnel_id == personnel.id,
                        personnel_room_association.c.room_id == room_id
                    ).first()
                    
                    if access:
                        access_granted = True
                        denial_reason = None
                        print(f"✅ {person} has access to room {room_id}")
                    else:
                        access_granted = False
                        denial_reason = "no_access"
                        print(f"❌ {person} does NOT have access to room {room_id}")
                else:
                    access_granted = False
                    denial_reason = "personnel_not_found"
                    print(f"⚠️ Personnel not found for name: {person}")
            else:
                access_granted = False
                denial_reason = "invalid_name_format"
                print(f"⚠️ Invalid name format: {person}")
        elif 'Unknown' in person:
            access_granted = False
            denial_reason = "unknown_person"
            print(f"⚠️ Unknown person detected - access denied")
        else:
            # No room specified, just logging detection
            access_granted = None
            denial_reason = None

        # Save video if frames provided
        if save_video and frames_to_save and len(frames_to_save) > 0:
            video_dir = get_today_subdirectory('video')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            safe_person = "".join(c if c.isalnum() or c in " _-" else "_" for c in person)[:30]
            video_filename = f"{safe_person}_{timestamp}_{unique_id}.mp4"
            video_path = str(video_dir / video_filename)
            
            print(f"🎬 Saving video with {len(frames_to_save)} frames...")
            
            success = save_video_with_ffmpeg(
                frames=frames_to_save,
                output_path=video_path,
                fps=20,
                quality="medium",
                for_web=True
            )
            
            if not success:
                print("⚠️ FFmpeg failed, trying OpenCV fallback...")
                success = save_fallback_opencv(frames_to_save, video_path, fps=20)
            
            if not success:
                print("❌ Failed to save video with both methods")
                video_path = None
            else:
                print(f"✅ Video saved successfully: {video_path}")
                
        if face_to_save and len(face_to_save) > 0:
            video_dir = get_today_subdirectory('unknown_faces')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            safe_person = "".join(c if c.isalnum() or c in " _-" else "_" for c in person)[:30]
            video_filename = f"{safe_person}_{timestamp}_{unique_id}.mp4"
            video_path = str(video_dir / video_filename)
            
            print(f"🎬 Saving video with {len(face_to_save)} frames...")
            
            success = save_video_with_ffmpeg(
                frames=face_to_save,
                output_path=video_path,
                fps=20,
                quality="veryslow",
                for_web=False
            )
            
            if not success:
                print("⚠️ FFmpeg failed, trying OpenCV fallback...")
                success = save_fallback_opencv(face_to_save, video_path, fps=20)
            
            if not success:
                print("❌ Failed to save video with both methods")
                video_path = None
            else:
                print(f"✅ Video saved successfully: {video_path}")

        # Create detection log with access information
        detection = DetectionLog(
            person=person,
            confidence=float(confidence),
            detection_time=datetime.now(),
            face_image_path=None,
            camera_id=camera_id,
            video_path=video_path,
            ref_img_id=ref_img_id, 
            area=area,
            room_id=room_id,
            access_granted=access_granted,  # NEW
            denial_reason=denial_reason  # NEW
        )

        db.add(detection)
        db.flush()
        log_id = detection.id
        
        # Save face image if provided
        if face_image is not None and face_image.size > 0:
            daily_dir = get_today_subdirectory('image')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe_person = "".join(c if c.isalnum() or c in " _-" else "_" for c in person)[:50]
            unique_id = str(uuid.uuid4())[:6]
            confidence_str = f"{confidence:.2f}".replace(".", "p")
            filename = f"{safe_person}_{confidence_str}_{timestamp}_{unique_id}.jpg"
            face_path = str(daily_dir / filename)
            
            cv2.imwrite(face_path, face_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            detection.face_image_path = face_path
            print(f"✅ Face image saved: {face_path}")
        
        db.commit()
        
        # Print access summary
        if room_id:
            status = "GRANTED" if access_granted else "DENIED"
            print(f"🔐 Access {status} for {person} to room {room_id} - Reason: {denial_reason or 'N/A'}")
        
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
        if close_db:
            db.close()



# ==================== WRAPPER FUNCTION ====================
def save_detection_with_face(
    person: str, 
    area: str, 
    confidence: float, 
    face_image: np.ndarray = None,
    camera_id: int = None, 
    ref_img_id: int = None,
    frames_to_save: list = None,
    face_to_save: list = None,
    save_video: bool = False,
    room_id: int = None  # NEW parameter
):
    """
    Helper function to save detection with face image and room access check
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
            room_id=room_id  # Pass room_id
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




def add_personnel_image(personnel_id: int, image_url: str, description: str = None, is_primary: bool = False, db: Session = None):
    """
    Add an image to a personnel record
    Returns the created PersonnelImage object or None if failed
    """
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    try:
        # Check if personnel exists
        personnel = db.query(Personnel).filter(Personnel.id == personnel_id).first()
        if not personnel:
            print(f"❌ Personnel with ID {personnel_id} not found")
            return None
        
        # If this is set as primary, unset any existing primary images
        if is_primary:
            db.query(PersonnelImage).filter(
                PersonnelImage.personnel_id == personnel_id,
                PersonnelImage.is_primary == True
            ).update({PersonnelImage.is_primary: False})
        
        # Create new image record
        personnel_image = PersonnelImage(
            image_url=image_url,
            personnel_id=personnel_id,
            description=description,
            is_primary=is_primary
        )
        
        db.add(personnel_image)
        db.commit()
        db.refresh(personnel_image)
        
        print(f"✅ Added image for personnel ID {personnel_id}: {image_url}")
        return personnel_image
        
    except Exception as e:
        print(f"❌ Error adding personnel image: {e}")
        db.rollback()
        return None
    finally:
        if close_db:
            db.close()


def get_personnel_images(personnel_id: int, db: Session = None):
    """Get all images for a specific personnel"""
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    try:
        images = db.query(PersonnelImage).filter(
            PersonnelImage.personnel_id == personnel_id
        ).order_by(
            PersonnelImage.is_primary.desc(),
            PersonnelImage.uploaded_at.desc()
        ).all()
        
        return images
    finally:
        if close_db:
            db.close()

def delete_personnel_image(image_id: int, db: Session = None):
    """Delete a personnel image"""
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    try:
        image = db.query(PersonnelImage).filter(PersonnelImage.id == image_id).first()
        if image:
            db.delete(image)
            db.commit()
            print(f"✅ Deleted personnel image ID {image_id}")
            return True
        else:
            print(f"❌ Personnel image ID {image_id} not found")
            return False
    except Exception as e:
        print(f"❌ Error deleting personnel image: {e}")
        db.rollback()
        return False
    finally:
        if close_db:
            db.close()
         
# ==================== ROOM MANAGEMENT FUNCTIONS ====================
def create_room(
    room_number: str,
    room_name: str = None,
    room_type: str = None,
    capacity: int = None,
    description: str = None,
    db: Session = None
):
    """Create a new room"""
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    try:
        # Check if room already exists
        existing_room = db.query(Room).filter(Room.room_number == room_number).first()
        if existing_room:
            print(f"❌ Room {room_number} already exists")
            return None
        
        room = Room(
            room_number=room_number,
            room_name=room_name,
            room_type=room_type,
            capacity=capacity,
            is_active=True,
            description=description
        )
        
        db.add(room)
        db.commit()
        db.refresh(room)
        print(f"✅ Created room: {room_number} (ID: {room.id})")
        return room
        
    except Exception as e:
        print(f"❌ Error creating room: {e}")
        db.rollback()
        return None
    finally:
        if close_db:
            db.close()


def get_room(room_id: int = None, room_number: str = None, db: Session = None):
    """Get room by ID or room number"""
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    try:
        if room_id:
            room = db.query(Room).filter(Room.id == room_id, Room.is_active == True).first()
        elif room_number:
            room = db.query(Room).filter(Room.room_number == room_number, Room.is_active == True).first()
        else:
            print("❌ Please provide either room_id or room_number")
            return None
        
        return room
    finally:
        if close_db:
            db.close()


def get_all_rooms(active_only: bool = True, db: Session = None):
    """Get all rooms"""
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    try:
        query = db.query(Room)
        if active_only:
            query = query.filter(Room.is_active == True)
        
        return query.order_by(Room.room_number).all()
    finally:
        if close_db:
            db.close()


def update_room(room_id: int, db: Session = None, **kwargs):
    """Update room information"""
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    try:
        room = db.query(Room).filter(Room.id == room_id).first()
        if not room:
            print(f"❌ Room with ID {room_id} not found")
            return None
        
        for key, value in kwargs.items():
            if hasattr(room, key) and key not in ['id', 'created_at']:
                setattr(room, key, value)
        
        db.commit()
        db.refresh(room)
        print(f"✅ Updated room: {room.room_number}")
        return room
        
    except Exception as e:
        print(f"❌ Error updating room: {e}")
        db.rollback()
        return None
    finally:
        if close_db:
            db.close()


def delete_room(room_id: int, hard_delete: bool = False, db: Session = None):
    """Delete or deactivate a room"""
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    try:
        room = db.query(Room).filter(Room.id == room_id).first()
        if not room:
            print(f"❌ Room with ID {room_id} not found")
            return False
        
        if hard_delete:
            db.delete(room)
            print(f"✅ Permanently deleted room: {room.room_number}")
        else:
            room.is_active = False
            print(f"✅ Deactivated room: {room.room_number}")
        
        db.commit()
        return True
        
    except Exception as e:
        print(f"❌ Error deleting room: {e}")
        db.rollback()
        return False
    finally:
        if close_db:
            db.close()


# ==================== PERSONNEL-ROOM ACCESS FUNCTIONS ====================
def grant_room_access(
    personnel_id: int, 
    room_id: int, 
    assigned_by: str = None,
    db: Session = None
):
    """Grant a personnel access to a room"""
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    try:
        # Check if personnel exists
        personnel = db.query(Personnel).filter(Personnel.id == personnel_id).first()
        if not personnel:
            print(f"❌ Personnel with ID {personnel_id} not found")
            return False
        
        # Check if room exists and is active
        room = db.query(Room).filter(Room.id == room_id, Room.is_active == True).first()
        if not room:
            print(f"❌ Room with ID {room_id} not found or inactive")
            return False
        
        # Check if access already exists
        existing = db.query(personnel_room_association).filter(
            personnel_room_association.c.personnel_id == personnel_id,
            personnel_room_association.c.room_id == room_id
        ).first()
        
        if existing:
            print(f"⚠️ Personnel {personnel.fname} {personnel.lname} already has access to {room.room_number}")
            return False
        
        # Grant access
        stmt = personnel_room_association.insert().values(
            personnel_id=personnel_id,
            room_id=room_id,
            assigned_at=datetime.now(),
            assigned_by=assigned_by
        )
        db.execute(stmt)
        db.commit()
        
        print(f"✅ Granted {personnel.fname} {personnel.lname} access to {room.room_number}")
        return True
        
    except Exception as e:
        print(f"❌ Error granting access: {e}")
        db.rollback()
        return False
    finally:
        if close_db:
            db.close()


def revoke_room_access(personnel_id: int, room_id: int, db: Session = None):
    """Revoke a personnel's access to a room"""
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    try:
        # Get names for logging
        personnel = db.query(Personnel).filter(Personnel.id == personnel_id).first()
        room = db.query(Room).filter(Room.id == room_id).first()
        
        result = db.execute(
            personnel_room_association.delete().where(
                personnel_room_association.c.personnel_id == personnel_id,
                personnel_room_association.c.room_id == room_id
            )
        )
        db.commit()
        
        if result.rowcount > 0:
            print(f"✅ Revoked {personnel.fname} {personnel.lname}'s access to {room.room_number}")
            return True
        else:
            print(f"⚠️ No access found to revoke")
            return False
            
    except Exception as e:
        print(f"❌ Error revoking access: {e}")
        db.rollback()
        return False
    finally:
        if close_db:
            db.close()


def get_personnel_rooms(personnel_id: int, db: Session = None):
    """Get all rooms a personnel has access to"""
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    try:
        personnel = db.query(Personnel).filter(Personnel.id == personnel_id).first()
        if not personnel:
            print(f"❌ Personnel with ID {personnel_id} not found")
            return []
        
        return personnel.rooms  # Using the relationship
    finally:
        if close_db:
            db.close()


def get_room_personnel(room_id: int, db: Session = None):
    """Get all personnel who have access to a room"""
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    try:
        room = db.query(Room).filter(Room.id == room_id).first()
        if not room:
            print(f"❌ Room with ID {room_id} not found")
            return []
        
        return room.personnel  # Using the relationship
    finally:
        if close_db:
            db.close()


def check_room_access(personnel_id: int, room_id: int, db: Session = None):
    """Check if a personnel has access to a room"""
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    try:
        result = db.query(personnel_room_association).filter(
            personnel_room_association.c.personnel_id == personnel_id,
            personnel_room_association.c.room_id == room_id
        ).first()
        
        return result is not None
    finally:
        if close_db:
            db.close()


# ==================== ROOM ACCESS LOG FUNCTIONS ====================
# In db_functions.py - Fix your log_room_access function

# def log_room_access(
#     personnel_id: int,
#     room_id: int,
#     access_granted: bool,
#     face_confidence: float = None,  # Make sure this parameter exists
#     denial_reason: str = None,
#     camera_id: int = None,
#     db: Session = None
# ):
#     """Log an access attempt to a room"""
#     close_db = False
#     if db is None:
#         db = SessionLocal()
#         close_db = True
    
#     try:
#         # First, verify personnel exists
#         personnel = db.query(Personnel).filter(Personnel.id == personnel_id).first()
#         if not personnel:
#             print(f"❌ Personnel {personnel_id} not found")
#             return None
        
#         # Verify room exists
#         room = db.query(Room).filter(Room.id == room_id).first()
#         if not room:
#             print(f"❌ Room {room_id} not found")
#             return None
        
#         # Create access log
#         access_log = RoomAccessLog(
#             personnel_id=personnel_id,
#             room_id=room_id,
#             access_time=datetime.now(),
#             access_granted=access_granted,
#             face_confidence=face_confidence,  # Now this works
#             denial_reason=denial_reason if not access_granted else None
#         )
        
#         db.add(access_log)
#         db.commit()
#         db.refresh(access_log)
        
#         print(f"✅ Access logged: {access_granted} for personnel {personnel_id}")
#         return access_log
        
#     except Exception as e:
#         print(f"❌ Error in log_room_access: {e}")
#         import traceback
#         traceback.print_exc()
#         db.rollback()
#         return None
#     finally:
#         if close_db:
#             db.close()
# def get_access_logs(
#     room_id: int = None,
#     personnel_id: int = None,
#     start_date: datetime = None,
#     end_date: datetime = None,
#     access_granted: bool = None,
#     limit: int = 100,
#     db: Session = None
# ):
#     """Query access logs with filters"""
#     close_db = False
#     if db is None:
#         db = SessionLocal()
#         close_db = True
    
#     try:
#         query = db.query(RoomAccessLog)
        
#         if room_id:
#             query = query.filter(RoomAccessLog.room_id == room_id)
#         if personnel_id:
#             query = query.filter(RoomAccessLog.personnel_id == personnel_id)
#         if start_date:
#             query = query.filter(RoomAccessLog.access_time >= start_date)
#         if end_date:
#             query = query.filter(RoomAccessLog.access_time <= end_date)
#         if access_granted is not None:
#             query = query.filter(RoomAccessLog.access_granted == access_granted)
        
#         return query.order_by(RoomAccessLog.access_time.desc()).limit(limit).all()
#     finally:
#         if close_db:
#             db.close()


# def get_access_statistics(
#     room_id: int = None, 
#     personnel_id: int = None,
#     days: int = 30, 
#     db: Session = None
# ):
#     """Get access statistics"""
#     close_db = False
#     if db is None:
#         db = SessionLocal()
#         close_db = True
    
#     try:
#         start_date = datetime.now() - timedelta(days=days)
        
#         query = db.query(RoomAccessLog).filter(RoomAccessLog.access_time >= start_date)
        
#         if room_id:
#             query = query.filter(RoomAccessLog.room_id == room_id)
#         if personnel_id:
#             query = query.filter(RoomAccessLog.personnel_id == personnel_id)
        
#         total_attempts = query.count()
#         granted = query.filter(RoomAccessLog.access_granted == True).count()
#         denied = total_attempts - granted
        
#         # Denial reasons breakdown
#         denial_stats = db.query(
#             RoomAccessLog.denial_reason,
#             func.count(RoomAccessLog.id)
#         ).filter(
#             RoomAccessLog.access_time >= start_date,
#             RoomAccessLog.access_granted == False
#         )
        
#         if room_id:
#             denial_stats = denial_stats.filter(RoomAccessLog.room_id == room_id)
#         if personnel_id:
#             denial_stats = denial_stats.filter(RoomAccessLog.personnel_id == personnel_id)
        
#         denial_stats = denial_stats.group_by(RoomAccessLog.denial_reason).all()
        
#         return {
#             "period_days": days,
#             "total_attempts": total_attempts,
#             "granted": granted,
#             "denied": denied,
#             "grant_rate": granted / total_attempts if total_attempts > 0 else 0,
#             "denial_reasons": {reason or "unknown": count for reason, count in denial_stats}
#         }
#     finally:
#         if close_db:
#             db.close()