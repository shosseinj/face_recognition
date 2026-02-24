from fastapi import APIRouter, Depends, Request, Query, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, date, time, timedelta
from sqlalchemy import func
from pydantic import BaseModel
from fastapi.responses import FileResponse, StreamingResponse
import os
from .routers import personnel, detections
from .models.database import DetectionLog, get_db
from .models.schemas import DetectionLogCreate, DetectionLogResponse
from .routers import personnel




router = APIRouter()
# Include the routers from your modules
router.include_router(personnel.router)  # personnel router already has prefix="/personnel"
router.include_router(detections.router)  # detections router already has prefix="/detections"


# Keep only helper functions if needed elsewhere
def get_face_image_url(request: Request, detection_id: int) -> Optional[str]:
    """Generate URL for face image"""
    if not detection_id:
        return None
    base_url = str(request.base_url).rstrip('/')
    return f"{base_url}/api/v1/detections/{detection_id}/face"

def get_video_url(request: Request, detection_id: int) -> Optional[str]:
    """Generate URL for video clip"""
    if not detection_id:
        return None
    base_url = str(request.base_url).rstrip('/')
    return f"{base_url}/api/v1/detections/{detection_id}/video"



# Existing endpoints
@router.post("/log", response_model=DetectionLogResponse)
def log_detection(
    request: Request,
    person_data: DetectionLogCreate, 
    db: Session = Depends(get_db)
):
    """Log a face detection"""
    db_log = DetectionLog(
        person=person_data.person,
        confidence=person_data.confidence,
        face_image_path=person_data.face_image_path if hasattr(person_data, 'face_image_path') else None,
        detection_time=datetime.now()
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    
    return DetectionLogResponse(
        id=db_log.id,
        person=db_log.person,
        confidence=float(db_log.confidence) if db_log.confidence else None,
        detection_time=db_log.detection_time,
        face_image_url=get_face_image_url(request, db_log.id) if db_log.face_image_path else None,
        video_url=get_video_url(request, db_log.id) if db_log.video_path else None
    )


@router.get("/logs", response_model=List[DetectionLogResponse])
def get_logs(request: Request, db: Session = Depends(get_db)):
    """Get all detection logs with face image and video URLs"""
    logs = db.query(DetectionLog).order_by(DetectionLog.detection_time.desc()).all()
    
    return [
        DetectionLogResponse(
            id=log.id,
            person=log.person,
            confidence=float(log.confidence) if log.confidence else None,
            detection_time=log.detection_time,
            face_image_url=get_face_image_url(request, log.id) if log.face_image_path else None,
            video_url=get_video_url(request, log.id) if log.video_path else None
        )
        for log in logs
    ]


@router.get("/detections", response_model=List[DetectionLogResponse])
async def get_detections(
    request: Request,
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    """Get list of detections with face and video URLs"""
    detections = db.query(DetectionLog)\
        .order_by(DetectionLog.detection_time.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
    
    return [
        DetectionLogResponse(
            id=detection.id,
            person=detection.person,
            confidence=float(detection.confidence) if detection.confidence else None,
            detection_time=detection.detection_time,
            face_image_url=get_face_image_url(request, detection.id) if detection.face_image_path else None,
            video_url=get_video_url(request, detection.id) if detection.video_path else None
        )
        for detection in detections
    ]

@router.get("/detections/{detection_id}/face")
async def get_detection_face(
    detection_id: int, 
    db: Session = Depends(get_db)
):
    """Serve face image for a detection"""
    detection = db.query(DetectionLog).filter(DetectionLog.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    if not detection.face_image_path:
        raise HTTPException(status_code=404, detail="No face image available for this detection")
    
    if not os.path.exists(detection.face_image_path):
        raise HTTPException(status_code=404, detail="Face image file not found on disk")
    
    return FileResponse(
        detection.face_image_path,
        media_type="image/jpeg",
        filename=os.path.basename(detection.face_image_path)
    )


@router.get("/detections/{detection_id}/video")
async def get_detection_video(
    detection_id: int, 
    request: Request,
    db: Session = Depends(get_db)
):
    """Serve video file for a detection with range request support"""
    detection = db.query(DetectionLog).filter(DetectionLog.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    if not detection.video_path:
        raise HTTPException(status_code=404, detail="No video available for this detection")
    
    if not os.path.exists(detection.video_path):
        raise HTTPException(status_code=404, detail="Video file not found on disk")
    
    # Get file size
    file_size = os.path.getsize(detection.video_path)
    
    # Parse Range header
    range_header = request.headers.get("range")
    
    if range_header:
        # Handle range request
        byte_range = range_header.replace("bytes=", "").split("-")
        start = int(byte_range[0])
        end = int(byte_range[1]) if byte_range[1] else file_size - 1
        
        # Ensure end is within bounds
        end = min(end, file_size - 1)
        content_length = end - start + 1
        
        # Read the specified range
        def read_range():
            with open(detection.video_path, "rb") as video_file:
                video_file.seek(start)
                yield video_file.read(content_length)
        
        # Return partial content
        return StreamingResponse(
            read_range(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(content_length),
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-cache",
            }
        )
    else:
        # Return full file
        def read_file():
            with open(detection.video_path, "rb") as video_file:
                while chunk := video_file.read(8192):  # 8KB chunks
                    yield chunk
        
        return StreamingResponse(
            read_file(),
            media_type="video/mp4",
            headers={
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-cache",
            }
        )