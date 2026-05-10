from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from pydantic import Field
from ..validators import validate_iran_national_code, normalize_national_code
import re
from enum import Enum
from pydantic import BaseModel, Field, validator

# ==================== ROOM SCHEMAS ====================

class RoomTypeEnum(str, Enum):
    """Room type options"""
    ILLEGAL = "illegal"
    PUBLIC = "public"
    CUSTOM = "custom"
    
    @classmethod
    def get_values(cls):
        return [item.value for item in cls]
    



class RoomBase(BaseModel):
    room_number: str
    room_name: Optional[str] = None
    room_type: Optional[RoomTypeEnum] = None 
    description: Optional[str] = None
    polygon: Optional[list] = None  # ✅ Add this - will store [[x1,y1], [x2,y2], ...]
    camera_id: Optional[int] = None  # ✅ Add this

    @validator('room_type', pre=True)
    def validate_room_type(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            # Case-insensitive validation with Persian error message
            v_lower = v.lower()
            if v_lower not in [e.value for e in RoomTypeEnum]:
                raise ValueError(f'نوع ناحیه باید یکی از موارد زیر باشد: illegal, public, custom')
            return RoomTypeEnum(v_lower)
        return v
    

class RoomCreate(RoomBase):
    pass

class RoomUpdate(BaseModel):
    room_number: Optional[str] = None
    room_name: Optional[str] = None
    room_type: Optional[RoomTypeEnum] = None 
    is_active: Optional[bool] = None
    description: Optional[str] = None
    polygon: Optional[list] = None  # ✅ Add this
    camera_id: Optional[int] = None  # ✅ Add this


class RoomResponse(RoomBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# ==================== DETECTION LOG SCHEMAS ====================
# class DetectionLogCreate(BaseModel):
#     person: str
#     confidence: Optional[float] = None
#     ref_img_id: Optional[int] = None
#     face_image_path: Optional[str] = None


class DetectionLogCreate(BaseModel):
    person: str
    confidence: Optional[float] = None
    ref_img_id: Optional[int] = None
    face_image_path: Optional[str] = None
    
    # Additional fields you might want to add:
    detection_time: Optional[datetime] = None  # Allow custom time, defaults to now()
    room_id: Optional[int] = None
    access_granted: Optional[bool] = None
    video_url: Optional[str] = None
    camera_id: Optional[int] = None  # If you have multiple cameras
    
    class Config:
        from_attributes = True

        
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
    primary_image: Optional[str]
    last_seen : Optional[datetime]
    primary_image_base64: Optional[str] = None  # ADD THIS LINE
    class Config:
        from_attributes = True


# ==================== PERSONNEL IMAGE SCHEMAS ====================
class PersonnelImageBase(BaseModel):
    image_url: str
    personnel_id: int
    is_primary: Optional[bool] = False  # ADD THIS

class PersonnelImageCreate(PersonnelImageBase):
    pass

class PersonnelImageResponse(BaseModel):
    id: int
    image_base64: str  # Changed from image_url to image_base64
    personnel_id: int
    is_primary: Optional[bool] = False
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
    primary_image: Optional[str] = None  # ADD THIS FIELD
    
    class Config:
        from_attributes = True


# ==================== OTHER SCHEMAS ====================
class PersonnelFromLogsRequest(PersonnelCreate):
    log_ids: List[int] = Field(..., description="List of detection log IDs")