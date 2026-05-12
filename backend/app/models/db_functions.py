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
from typing import Optional
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
import base64

import asyncio
import json
# from .db_functions import sync_polygons_to_json



def sync_polygons_to_json(db: Session, json_path: str = "polygon_points.json") -> bool:
    """
    Gather all room polygons from database and save to JSON file.
    Replaces the existing file with current database data.
    
    Args:
        db: Database session
        json_path: Path to the JSON file (default: polygon_points.json)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Query all active rooms that have polygons
        rooms = db.query(Room).filter(
            Room.is_active == True,
            Room.polygon.isnot(None)
        ).all()
        
        # Extract polygons
        polygons = []
        for room in rooms:
            if room.polygon and len(room.polygon) >= 3:
                polygons.append(room.polygon)
        
        # Write to JSON file (overwrite)
        with open(json_path, 'w') as f:
            json.dump(polygons, f, indent=2)
        
        print(f"✅ Synced {len(polygons)} polygons to {json_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error syncing polygons to JSON: {e}")
        return False
    
    




# ==================== HELPER FUNCTIONS ====================
def get_today_subdirectory(file_type) -> Path:
    """Create and return today's date-based subdirectory"""
    today = datetime.now().strftime("%Y-%m-%d")
    daily_dir = FACE_STORAGE_DIR / today / file_type
    daily_dir.mkdir(exist_ok=True, parents=True)
    return daily_dir

# ==================== MAIN SAVE FUNCTION ====================

async def save_detection(
    person: str, 
    area: str, 
    confidence: float, 
    face_image: np.ndarray = None, 
    ref_img_id: int = None,
    camera_id: int = None, 
    frames_to_save: list = None,  
    face_to_save: list = None,
    save_video: bool = False,
    room_id: int = None,
    db: Session = None
):
    """Save a detection with optional face image, with 60-second cooldown per person."""
    
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    face_path = None
    log_id = None
    video_path = None
    access_granted = None
    
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
        # Check for unknown person first
        # Handle access check based on room type
        if 'Unknown' in person:
            access_granted = False
            print(f"⚠️ Unknown person (national code: {person}) - access denied")
            
        elif not room_id:
            access_granted = None
            print(f"⚠️ No room ID provided")
            
        else:
            room = db.query(Room).filter(Room.id == room_id).first()
            
            if not room:
                access_granted = False
                print(f"⚠️ Room {room_id} not found")
                
            elif room.room_type == 'public':
                access_granted = True
                print(f"✅ PUBLIC ROOM: National code {person} granted access to room {room_id}")
                
            elif room.room_type == 'illegal':
                access_granted = False
                print(f"❌ ILLEGAL ROOM: National code {person} denied access to room {room_id}")
                
            elif room.room_type == 'custom':
                personnel = db.query(Personnel).filter(
                    Personnel.national_code == person
                ).first()
                
                if not personnel:
                    access_granted = False
                    print(f"⚠️ Personnel not found with national code: {person}")
                else:
                    access_exists = db.query(personnel_room_association).filter(
                        personnel_room_association.c.personnel_id == personnel.id,
                        personnel_room_association.c.room_id == room_id
                    ).first()
                    
                    access_granted = bool(access_exists)
                    if access_exists:
                        print(f"✅ CUSTOM ROOM: {personnel.fname} {personnel.lname} (national code: {person}) has access to room {room_id}")
                    else:
                        print(f"❌ CUSTOM ROOM: {personnel.fname} {personnel.lname} (national code: {person}) does NOT have access to room {room_id}")
            else:
                access_granted = False
                print(f"⚠️ Unknown room type '{room.room_type}' for room {room_id}")




        # Save video if frames provided - NOW ASYNC
        if save_video and frames_to_save and len(frames_to_save) > 0:
            video_dir = get_today_subdirectory('video')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            safe_person = "".join(c if c.isalnum() or c in " _-" else "_" for c in person)[:30]
            video_filename = f"{safe_person}_{timestamp}_{unique_id}.mp4"
            video_path = str(video_dir / video_filename)
            
            print(f"🎬 Saving video with {len(frames_to_save)} frames...")
            
            # AWAIT the async function
            success = await save_video_with_ffmpeg(
                frames=frames_to_save,
                output_path=video_path,
                fps=20,
                quality="veryslow",
                for_web=True
            )
            
            if not success:
                print("⚠️ FFmpeg failed, trying OpenCV fallback...")
                # Use asyncio.to_thread for OpenCV fallback (CPU-bound)
                success = await asyncio.to_thread(save_fallback_opencv, frames_to_save, video_path, 20)
            
            if not success:
                print("❌ Failed to save video with both methods")
                video_path = None
            else:
                print(f"✅ Video saved successfully: {video_path}")
        
        # Save face video if provided
        if face_to_save and len(face_to_save) > 0:
            video_dir = get_today_subdirectory('unknown_faces')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            safe_person = "".join(c if c.isalnum() or c in " _-" else "_" for c in person)[:30]
            video_filename = f"{safe_person}_{timestamp}_{unique_id}.mp4"
            video_path = str(video_dir / video_filename)
            
            print(f"🎬 Saving video with {len(face_to_save)} frames...")
            
            # AWAIT the async function
            success = await save_video_with_ffmpeg(
                frames=face_to_save,
                output_path=video_path,
                fps=20,
                quality="veryslow",
                for_web=True
            )
            
            if not success:
                print("⚠️ FFmpeg failed, trying OpenCV fallback...")
                success = await asyncio.to_thread(save_fallback_opencv, face_to_save, video_path, 20)
            
            if not success:
                print("❌ Failed to save video with both methods")
                video_path = None
            else:
                print(f"✅ Video saved successfully: {video_path}")

        # Create detection log
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
            access_granted=access_granted
        )

        db.add(detection)
        db.flush()
        log_id = detection.id
        
        # Save face image if provided (use asyncio.to_thread for CPU-bound operation)
        if face_image is not None and face_image.size > 0:
            daily_dir = get_today_subdirectory('image')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe_person = "".join(c if c.isalnum() or c in " _-" else "_" for c in person)[:50]
            unique_id = str(uuid.uuid4())[:6]
            confidence_str = f"{confidence:.2f}".replace(".", "p")
            filename = f"{safe_person}_{confidence_str}_{timestamp}_{unique_id}.jpg"
            face_path = str(daily_dir / filename)
            
            # Save image in thread pool
            def save_face_image():
                cv2.imwrite(face_path, face_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            await asyncio.to_thread(save_face_image)
            detection.face_image_path = face_path
            print(f"✅ Face image saved: {face_path}")
        
        db.commit()
        
        if room_id:
            status = "GRANTED" if access_granted else "DENIED"
            print(f"🔐 Access {status} for {person} to room {room_id} ")
        
        print(f"✅ Saved detection: {person} ({confidence:.2f}) with ID: {log_id}")
        return log_id
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        import traceback
        traceback.print_exc()
        
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
async def save_detection_with_face(
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
        # ADD AWAIT HERE
        log_id = await save_detection(
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
    # capacity: int = None,
    description: str = None,
    polygon: Optional[list] = None,  # ✅ Add this
    camera_id: Optional[int] = None,  # ✅ Add this
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
            # capacity=capacity,
            is_active=True,
            description=description,
            polygon=polygon,  # ✅ Add this line
            camera_id=camera_id  # ✅ Add this line
        )
        
        db.add(room)
        db.commit()
        db.refresh(room)
        print(f"✅ Created room: {room_number} (ID: {room.id})")

        sync_polygons_to_json(db)

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



def update_room(
    room_id: int,
    db: Session = None,
    room_number: str = None,
    room_name: str = None,
    room_type: str = None,
    is_active: bool = None,
    description: str = None,
    polygon: Optional[list] = None,  # ✅ Add this
    camera_id: Optional[int] = None,  # ✅ Add this
    **kwargs
):
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
        
        # Update fields if provided
        if room_number is not None:
            room.room_number = room_number
        if room_name is not None:
            room.room_name = room_name
        if room_type is not None:
            room.room_type = room_type
        if is_active is not None:
            room.is_active = is_active
        if description is not None:
            room.description = description
        if polygon is not None:  # ✅ Add this
            room.polygon = polygon
        if camera_id is not None:  # ✅ Add this
            room.camera_id = camera_id
        
        db.commit()
        db.refresh(room)
        print(f"✅ Updated room ID: {room_id}")
        sync_polygons_to_json(db)
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
        sync_polygons_to_json(db)
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



def get_primary_image_url(personnel_id: int, db: Session = None) -> Optional[str]:
    """Get the primary image URL for a personnel"""
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    try:
        primary_image = db.query(PersonnelImage).filter(
            PersonnelImage.personnel_id == personnel_id,
            PersonnelImage.is_primary == True
        ).first()
        
        if primary_image:
            return primary_image.image_url
        
        # If no primary image, return the first image
        first_image = db.query(PersonnelImage).filter(
            PersonnelImage.personnel_id == personnel_id
        ).first()
        
        return first_image.image_url if first_image else None
        
    finally:
        if close_db:
            db.close()



def image_to_base64(image_path: str) -> Optional[str]:
    """Convert image file to base64 string"""
    try:
        if not os.path.exists(image_path):
            return None
        
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Detect image type for data URL prefix
        file_extension = os.path.splitext(image_path)[1].lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp'
        }.get(file_extension, 'image/jpeg')
        
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None
    



# app/utils/polygon_sync.py

