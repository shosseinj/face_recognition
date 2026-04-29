from fastapi import APIRouter, Depends, Request, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
from fastapi.responses import FileResponse, StreamingResponse
import os
from ..models.database import DetectionLog, get_db



router = APIRouter(prefix="/detections", tags=["detections"])

def get_face_image_url(request: Request, detection_id: int) -> str | None:
    """Generate URL for face image"""
    if not detection_id:
        return None
    base_url = str(request.base_url).rstrip('/')
    return f"{base_url}/api/v1/detections/{detection_id}/face"

def get_video_url(request: Request, detection_id: int) -> str | None:
    """Generate URL for video clip"""
    if not detection_id:
        return None
    base_url = str(request.base_url).rstrip('/')
    return f"{base_url}/api/v1/detections/{detection_id}/video"



@router.get("/{detection_id}/video")
async def get_detection_video(
    detection_id: int, 
    request: Request,
    db: Session = Depends(get_db)
):
    """Serve video file with range support"""
    detection = db.query(DetectionLog).filter(DetectionLog.id == detection_id).first()
    if not detection or not detection.video_path:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if not os.path.exists(detection.video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    file_size = os.path.getsize(detection.video_path)
    range_header = request.headers.get("range")
    
    # Get filename for Content-Disposition
    filename = os.path.basename(detection.video_path)
    
    # Log for debugging
    print(f"Serving video: {detection.video_path}, size: {file_size}")
    print(f"Range header: {range_header}")
    
    # Common headers for video streaming
    common_headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "Content-Type": "video/mp4",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    }
    
    if range_header:
        # Handle range request for video streaming
        byte_range = range_header.replace("bytes=", "").split("-")
        start = int(byte_range[0])
        end = int(byte_range[1]) if byte_range[1] else file_size - 1
        end = min(end, file_size - 1)
        content_length = end - start + 1
        
        def read_range():
            with open(detection.video_path, "rb") as video_file:
                video_file.seek(start)
                yield video_file.read(min(8192, content_length))
        
        headers = {
            **common_headers,
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(content_length),
            "Content-Disposition": f"inline; filename=\"{filename}\"",
        }
        
        return StreamingResponse(
            read_range(),
            status_code=206,
            headers=headers
        )
    else:
        def read_file():
            with open(detection.video_path, "rb") as video_file:
                while chunk := video_file.read(8192):
                    yield chunk
        
        headers = {
            **common_headers,
            "Content-Length": str(file_size),
            "Content-Disposition": f"inline; filename=\"{filename}\"",
        }
        
        return StreamingResponse(
            read_file(),
            status_code=200,
            headers=headers
        )