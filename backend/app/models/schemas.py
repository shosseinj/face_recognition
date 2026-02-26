from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, List
from ..validators import validate_iran_national_code, normalize_national_code
import re

class DetectionLogCreate(BaseModel):
    person: str
    confidence: Optional[float] = None
    face_image_path: Optional[str] = None

class DetectionLogResponse(BaseModel):
    id: int
    person: str
    confidence: Optional[float] = None
    detection_time: datetime
    face_image_url: Optional[str] = None
    video_url: Optional[str] = None
    fname: Optional[str] = None
    lname: Optional[str] = None
    full_name: Optional[str] = "None"
    
    class Config:
        from_attributes = True








class PersonnelBase(BaseModel):
    fname: str = Field(..., description="First name in Persian")
    lname: str = Field(..., description="Last name in Persian")
    national_code: str = Field(..., description="Iranian national code (must be exactly 10 digits)")
    staff: bool = False
    department: Optional[str] = Field(None, description="Department name in Persian")
    
    @field_validator('fname', 'lname', 'department')
    @classmethod
    def validate_persian_text(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        # Allow Persian/Arabic characters, spaces, and common punctuation
        if not re.match(r'^[\u0600-\u06FF\s\-\.\']+$', v):
            raise ValueError('فقط حروف فارسی وارد کنید!')
        return v.strip()
    
    @field_validator('national_code')
    @classmethod
    def validate_national_code(cls, v: str) -> str:
        # First normalize (remove non-digits)
        normalized = normalize_national_code(v)
        
        # Check if normalization returned None (not exactly 10 digits)
        if normalized is None:
            raise ValueError('کد ملی باید دقیقا 10 رقم باشد!')
        
        # Validate the checksum
        if not validate_iran_national_code(normalized):
            raise ValueError('Invalid Iranian national code')
        
        return normalized

class PersonnelCreate(PersonnelBase):
    """Schema for creating a new personnel"""
    pass


class PersonnelUpdate(BaseModel):
    """Schema for updating an existing personnel (all fields optional)"""
    fname: Optional[str] = Field(None, description="First name in Persian")
    lname: Optional[str] = Field(None, description="Last name in Persian")
    national_code: Optional[str] = Field(None, description="Iranian national code (must be exactly 10 digits)")
    staff: Optional[bool] = None
    department: Optional[str] = Field(None, description="Department name in Persian")
    
    @field_validator('fname', 'lname', 'department')
    @classmethod
    def validate_persian_text(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not re.match(r'^[\u0600-\u06FF\s\-\.\']+$', v):
            raise ValueError('Field must contain only Persian characters')
        return v.strip()
    
    @field_validator('national_code')
    @classmethod
    def validate_national_code(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        # First normalize (remove non-digits)
        normalized = normalize_national_code(v)
        
        # Check if normalization returned None (not exactly 10 digits)
        if normalized is None:
            raise ValueError('کد ملی باید دقیقا 10 رقم باشد!')
        
        # Validate the checksum
        if not validate_iran_national_code(normalized):
            raise ValueError('Invalid Iranian national code')
        
        return normalized
    
    # REMOVE any unique validation from here - handle it in the endpoint

class Personnel(BaseModel):
    """Schema for personnel response (includes database fields)"""
    id: int
    fname: str
    lname: str
    national_code: str
    staff: bool
    department: Optional[str] = None
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

class PersonnelWithImages(BaseModel):
    """Schema for personnel with their associated images"""
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