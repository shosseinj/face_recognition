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
from .models.database import DetectionLog, Personnel, PersonnelImage, FACE_STORAGE_DIR
from .models.db_functions import get_db
from .models.schemas import (
    DetectionLogCreate, DetectionLogResponse, 
    PersonnelImageCreate, PersonnelImageResponse, PersonnelWithImages, PersonnelFromLogsRequest
)
from .routers import personnel
#increamental images
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
# from Detection.TensorRT.infer import vectorDatabase
from Face_ai.main import FaceEmbedding, DeletePointVD, FaceEmbeddingWithoutDetection, FaceCropping
import cv2
import numpy as np


from io import BytesIO
import zipfile
import tempfile




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

   








@router.get("/detections/{detection_id}/extract-frames")
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




