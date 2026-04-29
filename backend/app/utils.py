from fastapi import Request
from typing import Optional

def get_face_image_url(request: Request, detection_id: int) -> Optional[str]:
    """Generate URL for face image"""
    if not detection_id:
        return None
    base_url = str(request.base_url).rstrip('/')
    return f"{base_url}/api/v1/logs/{detection_id}/face"

def get_video_url(request: Request, detection_id: int) -> Optional[str]:
    """Generate URL for video clip"""
    if not detection_id:
        return None
    base_url = str(request.base_url).rstrip('/')
    return f"{base_url}/api/v1/logs/{detection_id}/video"