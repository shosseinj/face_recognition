from fastapi import Request
from typing import Optional

import base64
import requests
from pathlib import Path
from typing import Union

def convert_image_to_base64(image_source: Union[str, Path]) -> str:
    """
    Convert an image from a file path or URL to a base64 encoded string.
    
    Args:
        image_source: Local file path or remote URL to the image
        
    Returns:
        Base64 encoded string of the image
    """
    # Check if it's a URL
    if str(image_source).startswith(('http://', 'https://')):
        # Handle remote URL
        response = requests.get(image_source)
        response.raise_for_status()  # Raise error for bad status codes
        image_data = response.content
    else:
        # Handle local file path
        with open(image_source, 'rb') as img_file:
            image_data = img_file.read()
    
    # Encode to base64 and return as string
    return base64.b64encode(image_data).decode('utf-8')

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