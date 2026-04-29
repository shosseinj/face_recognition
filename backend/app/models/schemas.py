
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
from ..validators import validate_iran_national_code, normalize_national_code
import re



class DetectionLogCreate(BaseModel):
    person: str
    confidence: Optional[float] = None
    ref_img_id: Optional[int] = None
    face_image_path: Optional[str] = None  # Add this if not present

# class DetectionLogResponse(BaseModel):
#     id: int
#     person: str
#     confidence: Optional[float] = None
#     detection_time: datetime
#     face_image_url: Optional[str] = None
#     video_url : Optional[str] = None
    
#     class Config:
#         from_attributes = True
        
class DetectionLogResponse(BaseModel):
    id: int
    person: str
    confidence: Optional[float] = None
    detection_time: datetime
    face_image_url: Optional[str] = None
    video_url: Optional[str] = None
    ref_img_id: Optional[int] = None
    fname: Optional[str] = None  # Add this
    lname: Optional[str] = None  # Add this
    full_name : Optional[str] = "None"  # Add this
    class Config:
        from_attributes = True



class PersonnelBase(BaseModel):
    fname: str
    lname: str
    national_code: str
    staff: bool = False
    department: Optional[str] = None

# class PersonnelCreate(PersonnelBase):
#     pass

class PersonnelCreate(BaseModel):
    fname: str  # Pydantic handles Unicode automatically
    lname: str
    national_code: str
    staff: Optional[bool] = False
    department: Optional[str] = None
    
    class Config:
        json_encoders = {
            # Custom encoders if needed
        }

class Personnel(BaseModel):
    id: int
    fname: str
    lname: str
    national_code: str
    staff: bool
    department: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True



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

class PersonnelFromLogsRequest(PersonnelCreate):
    log_ids: List[int] = Field(..., description="List of detection log IDs")
    

class PersonnelWithImages(BaseModel):
    id: int
    fname: str
    lname: str
    national_code: str
    staff: Optional[bool] = False
    department: Optional[str] = None
    created_at: Optional[datetime] = None
    images: List[PersonnelImageResponse] = []
    
    class Config:
        from_attributes = True



class PersonnelImageResponse(PersonnelImageBase):
    id: int
    uploaded_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True