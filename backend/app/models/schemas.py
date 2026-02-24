
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class DetectionLogCreate(BaseModel):
    person: str
    confidence: Optional[float] = None
    face_image_path: Optional[str] = None  # Add this if not present

class DetectionLogResponse(BaseModel):
    id: int
    person: str
    confidence: Optional[float] = None
    detection_time: datetime
    face_image_url: Optional[str] = None
    video_url : Optional[str] = None
    
    class Config:
        from_attributes = True
        



class PersonnelBase(BaseModel):
    fname: str
    lname: str
    national_code: str
    staff: bool = False
    department: Optional[str] = None

class PersonnelCreate(PersonnelBase):
    pass

class PersonnelUpdate(BaseModel):
    fname: Optional[str] = None
    lname: Optional[str] = None
    national_code: Optional[str] = None
    staff: Optional[bool] = None
    department: Optional[str] = None

class Personnel(PersonnelBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

        