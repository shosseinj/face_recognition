from fastapi import APIRouter, Depends, Request, Query, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, date, time, timedelta
from sqlalchemy import func
from pydantic import BaseModel
from fastapi.responses import FileResponse, StreamingResponse
import os
from .routers import personnel, detections
# from .routers.personnel import delete_faces_from_vector_database

from .models.database import DetectionLog
from .models.db_functions import get_db
from .models.schemas import (
    DetectionLogCreate, DetectionLogResponse, 
    PersonnelImageCreate, PersonnelImageResponse, PersonnelWithImages
)
import uuid  # Add this import

from .routers import personnel
from fastapi import APIRouter, Depends, Request, Query, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, date, time, timedelta
from sqlalchemy import func
from pydantic import BaseModel
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
import os
import shutil
from pathlib import Path
import uuid
from .routers import personnel, detections, rooms
from .models.database import DetectionLog, Personnel as PersonnelDB, PersonnelImage, FACE_STORAGE_DIR, Room
from .models.db_functions import get_db
from .models.schemas import (
    DetectionLogCreate, DetectionLogResponse, 
    PersonnelImageCreate, PersonnelImageResponse, PersonnelWithImages, PersonnelFromLogsRequest
)
from .routers import personnel
#increamental images
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form
from sqlalchemy.orm import Session
from typing import  Optional
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
import cv2
from backend.app.utils import convert_image_to_base64, get_face_image_url, get_video_url






router = APIRouter()
# Include the routers from your modules
router.include_router(personnel.router)  # personnel router already has prefix="/personnel"
router.include_router(detections.router)  # detections router already has prefix="/detections"
router.include_router(rooms.router) 

# Keep only helper functions if needed elsewhere
# def get_face_image_url(request: Request, detection_id: int) -> Optional[str]:
#     """Generate URL for face image"""
#     if not detection_id:
#         return None
#     base_url = str(request.base_url).rstrip('/')
#     return f"{base_url}/api/v1/detections/{detection_id}/face"

# def get_video_url(request: Request, detection_id: int) -> Optional[str]:
#     """Generate URL for video clip"""
#     if not detection_id:
#         return None
#     base_url = str(request.base_url).rstrip('/')
#     return f"{base_url}/api/v1/detections/{detection_id}/video"









# @router.get("/detections/{detection_id}/video")
# async def get_detection_video(
#     detection_id: int, 
#     db: Session = Depends(get_db)
# ):
#     """Serve video file - FORCE DOWNLOAD THEN PLAY LOCALLY"""
    
#     detection = db.query(DetectionLog).filter(DetectionLog.id == detection_id).first()
#     if not detection:
#         raise HTTPException(status_code=404, detail="Detection not found")
    
#     if not detection.video_path:
#         raise HTTPException(status_code=404, detail="No video available")
    
#     if not os.path.exists(detection.video_path):
#         raise HTTPException(status_code=404, detail="Video file not found")
    
#     filename = os.path.basename(detection.video_path)
    
#     # Force download instead of playback
#     return FileResponse(
#         path=detection.video_path,
#         media_type="video/mp4",
#         filename=filename,
#         headers={
#             "Content-Disposition": f"attachment; filename=\"{filename}\"",
#             "Accept-Ranges": "none",
#         }
#     )





# @router.get("/debug/check-video/{detection_id}")
# async def check_video_quality(detection_id: int, db: Session = Depends(get_db)):
#     """Check if video is properly formatted for web"""
#     detection = db.query(DetectionLog).filter(DetectionLog.id == detection_id).first()
#     if not detection or not detection.video_path:
#         return {"error": "Video not found"}
    
#     video_path = detection.video_path
#     results = {
#         "path": video_path,
#         "exists": os.path.exists(video_path),
#     }
    
#     if results["exists"]:
#         # Get file size
#         results["file_size"] = os.path.getsize(video_path)
        
#         # Check with OpenCV
#         cap = cv2.VideoCapture(video_path)
#         if cap.isOpened():
#             results["readable"] = True
#             results["frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             results["fps"] = cap.get(cv2.CAP_PROP_FPS)
#             results["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             results["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             results["codec"] = int(cap.get(cv2.CAP_PROP_FOURCC))
#             cap.release()
            
#             # Check if web-optimized (moov atom at start)
#             try:
#                 with open(video_path, 'rb') as f:
#                     header = f.read(100)
#                     results["has_moov_at_start"] = b'moov' in header[:50]
#             except:
#                 pass
#         else:
#             results["readable"] = False
    
#     return results

   
@router.get("/home")
def get_home_data(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Single endpoint for home page - returns all necessary data using existing schemas
    """
    from sqlalchemy import func
    
    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = now.replace(hour=23, minute=59, second=59, microsecond=999999)
    month_start = now - timedelta(days=30)
    
    # ==================== PERSONNEL SUMMARY ====================
    # Total counts
    total_personnel = db.query(func.count(PersonnelDB.id)).scalar() or 0
    total_staff = db.query(func.count(PersonnelDB.id)).filter(PersonnelDB.staff == True).scalar() or 0
    
    # Latest 5 personnel added
    latest_personnel = db.query(PersonnelDB).order_by(
        PersonnelDB.created_at.desc()
    ).limit(5).all()
    
    # Create personnel list with primary images
    latest_personnel_list = []
    for p in latest_personnel:
        primary_image = db.query(PersonnelImage).filter(
            PersonnelImage.personnel_id == p.id,
            PersonnelImage.is_primary == True
        ).first()
        
        p_data = {
            "id": p.id,
            "fname": p.fname,
            "lname": p.lname,
            "national_code": p.national_code,
            "staff": p.staff,
            "department": p.department,
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "primary_image": convert_image_to_base64(primary_image.image_url) if primary_image else None
        }
        latest_personnel_list.append(p_data)
    
    # ==================== DETECTION LOGS SUMMARY ====================
    # Today's detection count
    today_detections = db.query(func.count(DetectionLog.id)).filter(
        DetectionLog.detection_time.between(today_start, today_end)
    ).scalar() or 0
    
    # Unique personnel detected today
    unique_today = db.query(func.count(func.distinct(DetectionLog.person))).filter(
        DetectionLog.detection_time.between(today_start, today_end)
    ).scalar() or 0
    
    # This month's detection count
    month_detections = db.query(func.count(DetectionLog.id)).filter(
        DetectionLog.detection_time >= month_start
    ).scalar() or 0
    
    # Latest 10 detection logs
    latest_logs = db.query(DetectionLog).order_by(
        DetectionLog.detection_time.desc()
    ).limit(10).all()
    
    # Get personnel info for logs
    personnel_codes = set(log.person for log in latest_logs if log.person)
    personnel_lookup = {}
    if personnel_codes:
        personnel_records = db.query(PersonnelDB).filter(
            PersonnelDB.national_code.in_(personnel_codes)
        ).all()
        personnel_lookup = {p.national_code: p for p in personnel_records}
    
    # Format recent logs
    recent_logs_list = []
    for log in latest_logs:
        personnel = personnel_lookup.get(log.person)
        log_data = {
            "id": log.id,
            "person": log.person,
            "confidence": float(log.confidence) if log.confidence else None,
            "detection_time": log.detection_time.isoformat() if log.detection_time else None,
            "face_image_url": get_face_image_url(request, log.id) if log.face_image_path else None,
            "video_url": get_video_url(request, log.id) if log.video_path else None,
            "fname": personnel.fname if personnel else None,
            "lname": personnel.lname if personnel else None,
            "full_name": f"{personnel.fname} {personnel.lname}".strip() if personnel else None,
            "access_granted": log.access_granted if hasattr(log, 'access_granted') else None
        }
        recent_logs_list.append(log_data)
    
    # ==================== ROOMS SUMMARY ====================
    total_rooms = db.query(func.count(Room.id)).scalar() or 0
    active_rooms = db.query(func.count(Room.id)).filter(Room.is_active == True).scalar() or 0
    
    # Latest 5 rooms
    latest_rooms = db.query(Room).order_by(Room.created_at.desc()).limit(5).all()
    latest_rooms_list = [
        {
            "id": room.id,
            "room_number": room.room_number,
            "room_name": room.room_name,
            "room_type": room.room_type,
            "capacity": room.capacity,
            "is_active": room.is_active,
            "created_at": room.created_at.isoformat() if room.created_at else None
        }
        for room in latest_rooms
    ]
    
    # ==================== COMBINED RESPONSE ====================
    return {
        "timestamp": now.isoformat(),
        "personnel": {
            "total": total_personnel,
            "staff": total_staff,
            "non_staff": total_personnel - total_staff,
            "latest": latest_personnel_list
        },
        "detections": {
            "today": today_detections,
            "this_month": month_detections,
            "unique_personnel_today": unique_today,
            "recent_logs": recent_logs_list
        },
        "rooms": {
            "total": total_rooms,
            "active": active_rooms,
            "latest": latest_rooms_list
        }
    }








@router.get("/extract-frames/{detection_id}/")
async def extract_frames_from_detection_video(
    detection_id: int,
    frame_interval: int = Query(30, description="Extract every Nth frame"),
    max_frames: Optional[int] = Query(None, description="Maximum number of frames to extract"),
    db: Session = Depends(get_db)
):
    """
    Extract frames from video and return them in memory (without saving to disk)
    """
    
    # Get detection log from database
    detection = db.query(DetectionLog).filter(DetectionLog.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    if not detection.video_path:
        raise HTTPException(status_code=404, detail="No video available for this detection")
    
    video_path = Path(detection.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found at: {video_path}")
    
    try:
        # Open video using OpenCV
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        frame_count = 0
        extracted_frames = []
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame based on interval
            if frame_count % frame_interval == 0:
                # Encode frame to JPEG in memory
                success, encoded_image = cv2.imencode('.jpg', frame)
                if success:
                    # Convert to base64 for JSON response
                    import base64
                    frame_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
                    
                    extracted_frames.append({
                        "frame_number": frame_count,
                        "frame_index": extracted_count,
                        "image_base64": frame_base64,
                        "image_size_kb": len(encoded_image.tobytes()) / 1024
                    })
                    extracted_count += 1
                
                # Stop if max frames reached
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        
        return {
            "success": True,
            "detection_id": detection_id,
            "video_path": str(video_path),
            "total_frames_in_video": frame_count,
            "frames_extracted": extracted_count,
            "frame_interval": frame_interval,
            "frames": extracted_frames
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting frames: {str(e)}")




