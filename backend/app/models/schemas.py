from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from pydantic import Field
from ..validators import validate_iran_national_code, normalize_national_code
import re


# ==================== ROOM SCHEMAS ====================
class RoomBase(BaseModel):
    room_number: str
    room_name: Optional[str] = None
    room_type: Optional[str] = None
    capacity: Optional[int] = None
    description: Optional[str] = None

class RoomCreate(RoomBase):
    pass

class RoomUpdate(BaseModel):
    room_number: Optional[str] = None
    room_name: Optional[str] = None
    room_type: Optional[str] = None
    capacity: Optional[int] = None
    is_active: Optional[bool] = None
    description: Optional[str] = None

class RoomResponse(RoomBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# ==================== DETECTION LOG SCHEMAS ====================
class DetectionLogCreate(BaseModel):
    person: str
    confidence: Optional[float] = None
    ref_img_id: Optional[int] = None
    face_image_path: Optional[str] = None

class LogSummaryResponse(BaseModel):
    total_detections: int
    unique_personnel: int
    average_confidence: float
    first_detection: Optional[datetime]
    last_detection: Optional[datetime]
    date_range_days: Optional[int] = None    

class DetectionLogResponse(BaseModel):
    id: int
    person: str
    confidence: Optional[float] = None
    detection_time: datetime
    face_image_url: Optional[str] = None
    video_url: Optional[str] = None
    ref_img_id: Optional[int] = None
    fname: Optional[str] = None
    lname: Optional[str] = None
    full_name: Optional[str] = "None"
    room_id: Optional[int] = None
    access_granted: Optional[bool] = None
    denial_reason: Optional[str] = None
    
    class Config:
        from_attributes = True


# ==================== PERSONNEL SCHEMAS ====================
class PersonnelBase(BaseModel):
    fname: str
    lname: str
    national_code: str
    staff: bool = False
    department: Optional[str] = None

class PersonnelCreate(BaseModel):
    fname: str
    lname: str
    national_code: str
    staff: Optional[bool] = False
    department: Optional[str] = None

class PersonnelUpdate(BaseModel):
    fname: Optional[str] = None
    lname: Optional[str] = None
    national_code: Optional[str] = None
    staff: Optional[bool] = None
    department: Optional[str] = None

# Main Personnel schema for responses (includes rooms)
class Personnel(BaseModel):
    id: int
    fname: str
    lname: str
    national_code: str
    staff: bool
    department: Optional[str] = None
    created_at: datetime
    rooms: List[RoomResponse] = []  # Add rooms field
    
    class Config:
        from_attributes = True


# ==================== PERSONNEL IMAGE SCHEMAS ====================
class PersonnelImageBase(BaseModel):
    image_url: str
    personnel_id: int

class PersonnelImageCreate(PersonnelImageBase):
    pass

class PersonnelImageResponse(PersonnelImageBase):
    id: int
    uploaded_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# ==================== PERSONNEL WITH IMAGES SCHEMA ====================
class PersonnelWithImages(BaseModel):
    id: int
    fname: str
    lname: str
    national_code: str
    staff: Optional[bool] = False
    department: Optional[str] = None
    created_at: Optional[datetime] = None
    rooms: List[RoomResponse] = []  # Include rooms
    images: List[PersonnelImageResponse] = []
    
    class Config:
        from_attributes = True


# ==================== OTHER SCHEMAS ====================
class PersonnelFromLogsRequest(PersonnelCreate):
    log_ids: List[int] = Field(..., description="List of detection log IDs")