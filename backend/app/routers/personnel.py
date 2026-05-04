from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    UploadFile,
    Query,
    Request,
    File,
    Response
)
import cv2
import shutil  # Add this line at the top with other imports
from sqlalchemy.orm import joinedload

from ..validators import normalize_national_code, validate_iran_national_code
from fastapi import APIRouter, Depends, Request, Query, HTTPException, UploadFile, File, Form
import uuid  # Add this import
import os
from sqlalchemy.orm import Session
from typing import List
# from ..schemas import Personnel 
from ..models.db_functions import (
    get_db,
    get_personnel_rooms,
    get_personnel_images,
    add_personnel_image,
    delete_personnel_image
)
from sqlalchemy import text

from ..models.schemas import Personnel as PersonnelSchema, PersonnelWithImages, PersonnelCreate, PersonnelUpdate, PersonnelImageResponse, PersonnelImageResponse, RoomResponse
router = APIRouter(prefix="/personnel", tags=["personnel"])
from ..models.database import PersonnelImage, Room as RoomDB
from pathlib import Path
import tempfile
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from io import BytesIO
import re
from ..models.database import DetectionLog
from ..models.schemas import DetectionLogResponse
from datetime import datetime
from typing import Optional
# from .detections import get_face_image_url, get_video_url
from ..utils import get_face_image_url, get_video_url
# from Face_ai.main import DeletePointVD, FaceEmbeddingWithoutDetection,  FaceEmbeddingCropping
from ..models.schemas import (
    DetectionLogCreate, DetectionLogResponse, 
    PersonnelImageCreate, PersonnelImageResponse, PersonnelWithImages, PersonnelFromLogsRequest
)
from ..models.database import DetectionLog, Personnel as PersonnelDB, PersonnelImage, FACE_STORAGE_DIR
from ..models.db_functions import get_db
import numpy as np
import zipfile

from Face_ai.main import ModelManager
model_mgr = ModelManager()


@router.get("/import-template")
async def download_import_template():
    """
    Download an Excel template for personnel import with Persian headers
    """
    import io
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    
    # Create workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "ورود اطلاعات پرسنل"
    
    # Persian headers only - no asterisks
    headers = [
        "نام",
        "نام خانوادگی", 
        "کد ملی",
        "کارمند",
        "دپارتمان"
    ]
    
    # Style for headers
    header_font = Font(bold=True, size=12, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_alignment = Alignment(horizontal='center', vertical='center')
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    for col, header_name in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col)
        cell.value = header_name
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment
        cell.border = border
    
    # Add example row with zero-padded national code (as text to preserve zeros)
    example_data = [
        "محمد",
        "احمدی",
        "0012345678",  # Zero-padded example - stored as text
        "بله",
        "فناوری اطلاعات"
    ]
    
    example_font = Font(size=11)
    example_alignment = Alignment(horizontal='left', vertical='center')
    example_fill = PatternFill(start_color="F0F0F0", end_color="F0F0F0", fill_type="solid")
    
    for col, value in enumerate(example_data, 1):
        cell = ws.cell(row=2, column=col, value=value)
        cell.font = example_font
        cell.alignment = example_alignment
        cell.border = border
        cell.fill = example_fill
        
        # For national code column, explicitly set as text format to preserve leading zeros
        if col == 3:  # کد ملی column
            cell.number_format = '@'  # Text format
    
    # Add instruction section
    instruction_row = 4
    ws.cell(row=instruction_row, column=1, value="راهنمای تکمیل:").font = Font(bold=True, size=12)
    
    instructions = [
        "• نام: اجباری - فقط حروف فارسی",
        "• نام خانوادگی: اجباری - فقط حروف فارسی",
        "• کد ملی: اجباری - دقیقا 10 رقم (مثال: 0012345678) - صفرهای اول را وارد کنید",
        "• کارمند: اختیاری - می‌توانید از 'بله/خیر'، 'true/false'، '1/0' استفاده کنید",
        "• دپارتمان: اختیاری - فقط حروف فارسی"
    ]
    
    instruction_font = Font(size=11)
    for i, instruction in enumerate(instructions, instruction_row + 1):
        cell = ws.cell(row=i, column=1, value=instruction)
        cell.font = instruction_font
    
    # Adjust column widths
    column_widths = [20, 20, 18, 12, 25]
    for col, width in enumerate(column_widths, 1):
        ws.column_dimensions[chr(64 + col)].width = width
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
        wb.save(tmp.name)
        tmp_path = tmp.name
    
    return FileResponse(
        path=tmp_path,
        filename="قالب_ورود_پرسنل.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )



@router.post("/import-excel")
async def import_personnel_excel(
    file: UploadFile = File(...),
    update_existing: bool = Query(False, description="به‌روزرسانی پرسنل موجود در صورت وجود کد ملی"),
    skip_invalid_rows: bool = Query(True, description="رد شدن از سطرهای نامعتبر و ادامه پردازش"),
    db: Session = Depends(get_db)
):
    """
    وارد کردن پرسنل از فایل اکسل
    
    فایل اکسل باید دارای ستون‌های: نام، نام خانوادگی، کد ملی، کارمند، دپارتمان
    """
    
    # Validate file type
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="فقط فایل‌های اکسل (.xlsx, .xls) پذیرفته می‌شوند")
    
    try:
        # Read Excel file - IMPORTANT: read all as string to preserve leading zeros
        contents = await file.read()
        
        # Use openpyxl directly to have more control over data types
        from openpyxl import load_workbook
        wb = load_workbook(filename=BytesIO(contents), data_only=True)
        ws = wb.active
        
        # Get headers (first row)
        headers = []
        for cell in ws[1]:
            header = cell.value
            if header:
                # Clean header (remove asterisk if present)
                header = header.replace(' *', '').strip()
                headers.append(header)
        
        # Map Persian headers to English column names
        column_mapping = {
            'نام': 'fname',
            'نام خانوادگی': 'lname', 
            'کد ملی': 'national_code',
            'کارمند': 'staff',
            'دپارتمان': 'department'
        }
        
        # Check required columns
        required_persian = ['نام', 'نام خانوادگی', 'کد ملی']
        missing_columns = [col for col in required_persian if col not in headers]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"ستون‌های اجباری وجود ندارند: {', '.join(missing_columns)}"
            )
        
        # Get column indices
        col_indices = {}
        for i, header in enumerate(headers, 1):
            if header in column_mapping:
                col_indices[column_mapping[header]] = i
        
        # Track results
        successful_rows = []
        failed_rows = []
        skipped_rows = []
        
        # Process data rows (starting from row 2)
        for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
            row_num = row_idx
            
            try:
                # Extract values using column indices
                fname = str(row[col_indices['fname']-1]).strip() if row[col_indices['fname']-1] else ''
                lname = str(row[col_indices['lname']-1]).strip() if row[col_indices['lname']-1] else ''
                
                # Handle national code - CRITICAL: preserve leading zeros
                raw_national_code = row[col_indices['national_code']-1]
                if raw_national_code is None:
                    raw_national_code = ''
                
                # Convert to string and handle different types
                if isinstance(raw_national_code, (int, float)):
                    # Convert to integer first to remove decimal, then to string
                    raw_national_code = str(int(float(raw_national_code)))
                    # Pad with leading zeros to make it 10 digits
                    if len(raw_national_code) < 10:
                        raw_national_code = raw_national_code.zfill(10)
                else:
                    raw_national_code = str(raw_national_code).strip()
                    # Remove any decimal point if present (e.g., "0012345678.0")
                    if '.' in raw_national_code:
                        raw_national_code = raw_national_code.split('.')[0]
                
                # Handle staff
                staff_value = row[col_indices['staff']-1] if 'staff' in col_indices else None
                if staff_value is None or staff_value == '':
                    staff = False
                else:
                    if isinstance(staff_value, bool):
                        staff = staff_value
                    elif isinstance(staff_value, (int, float)):
                        staff = bool(staff_value)
                    elif isinstance(staff_value, str):
                        staff_str = staff_value.strip().lower()
                        staff = staff_str in ['true', 'yes', '1', 'بله', 'y']
                    else:
                        staff = False
                
                # Handle department
                department = None
                if 'department' in col_indices:
                    dept_val = row[col_indices['department']-1]
                    if dept_val and str(dept_val).strip() and str(dept_val).lower() != 'nan':
                        department = str(dept_val).strip()
                
                # Validate required fields
                if not fname:
                    raise ValueError("نام اجباری است")
                if not lname:
                    raise ValueError("نام خانوادگی اجباری است")
                if not raw_national_code:
                    raise ValueError("کد ملی اجباری است")
                
                # Validate national code format (should be exactly 10 digits)
                if not re.match(r'^\d{10}$', raw_national_code):
                    raise ValueError(f"کد ملی باید دقیقا 10 رقم باشد (مقدار وارد شده: {raw_national_code})")
                
                # Validate the checksum
                if not validate_iran_national_code(raw_national_code):
                    raise ValueError("کد ملی نامعتبر است")
                
                # Validate Persian text
                if not re.match(r'^[\u0600-\u06FF\s\-\.\']+$', fname):
                    raise ValueError("نام باید فقط شامل حروف فارسی باشد")
                if not re.match(r'^[\u0600-\u06FF\s\-\.\']+$', lname):
                    raise ValueError("نام خانوادگی باید فقط شامل حروف فارسی باشد")
                if department and not re.match(r'^[\u0600-\u06FF\s\-\.\']+$', department):
                    raise ValueError("دپارتمان باید فقط شامل حروف فارسی باشد")
                
                # Check if personnel exists
                existing = db.query(Personnel).filter(
                    Personnel.national_code == raw_national_code
                ).first()
                
                if existing:
                    if update_existing:
                        # Update existing record
                        existing.fname = fname
                        existing.lname = lname
                        existing.staff = staff
                        existing.department = department
                        
                        db.flush()
                        successful_rows.append(row_num)
                    else:
                        # Skip existing record
                        skipped_rows.append({
                            "row": row_num,
                            "reason": "کد ملی تکراری است"
                        })
                else:
                    # Create new personnel
                    personnel = Personnel(
                        fname=fname,
                        lname=lname,
                        national_code=raw_national_code,  # Keep the zero-padded format
                        staff=staff,
                        department=department
                    )
                    db.add(personnel)
                    db.flush()
                    
                    successful_rows.append(row_num)
                
            except Exception as e:
                # Collect failed row info - only row number and error
                failed_rows.append({
                    "row": row_num,
                    "error": str(e)
                })
                continue
        
        # Commit all successful changes
        try:
            db.commit()
        except IntegrityError as e:
            db.rollback()
            raise HTTPException(
                status_code=400,
                detail=f"خطای پایگاه داده: {str(e)}"
            )
        
        return {
            "success": True,
            "filename": file.filename,
            "message": f"{len(successful_rows)} پرسنل وارد شدند، {len(failed_rows)} ناموفق، {len(skipped_rows)} رد شدند",
            "summary": {
                "total_rows": len(successful_rows) + len(failed_rows) + len(skipped_rows),
                "successful": len(successful_rows),
                "failed": len(failed_rows),
                "skipped": len(skipped_rows)
            },
            "successful_rows": successful_rows,
            "failed_rows": failed_rows,
            "skipped_rows": skipped_rows
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در پردازش فایل: {str(e)}")




@router.post("/", response_model=PersonnelSchema, status_code=status.HTTP_201_CREATED)
def create_personnel(personnel: PersonnelCreate, db: Session = Depends(get_db)):
    # Check if national code already exists
    existing = db.query(PersonnelDB).filter(PersonnelDB.national_code == personnel.national_code).first()
    if existing:
        raise HTTPException(status_code=400, detail="کد ملی موجود است!")
    
    # Create new personnel (rooms will be added separately if needed)
    db_personnel = PersonnelDB(
        fname=personnel.fname,
        lname=personnel.lname,
        national_code=personnel.national_code,
        staff=personnel.staff,
        department=personnel.department
    )
    
    db.add(db_personnel)
    db.commit()
    db.refresh(db_personnel)
    return db_personnel

    
@router.get("/", response_model=List[PersonnelWithImages])
def get_all_personnel(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    personnel_list = db.query(PersonnelDB).offset(skip).limit(limit).all()
    
    # Batch load primary images for all personnel
    personnel_ids = [p.id for p in personnel_list]
    primary_images = db.query(PersonnelImage).filter(
        PersonnelImage.personnel_id.in_(personnel_ids),
        PersonnelImage.is_primary == True
    ).all()
    
    # Create mapping
    primary_image_map = {img.personnel_id: img.image_url for img in primary_images}
    
    # Build responses
    responses = []
    for personnel in personnel_list:
        response = PersonnelWithImages.model_validate(personnel)
        response.primary_image = primary_image_map.get(personnel.id)
        responses.append(response)
    
    return responses



@router.get("/summary")  # Make sure this comes BEFORE /{personnel_id}
def get_logs_summary(
    from_date: Optional[datetime] = Query(None),
    to_date: Optional[datetime] = Query(None),
    db: Session = Depends(get_db)
):
    """Get overall summary for all detection logs"""
    from sqlalchemy import func
    
    query = db.query(DetectionLog)  # ✅ This is fine, no text() needed
    
    if from_date:
        query = query.filter(DetectionLog.detection_time >= from_date)
    if to_date:
        query = query.filter(DetectionLog.detection_time <= to_date)
    
    stats = query.with_entities(
        func.count(DetectionLog.id).label('total'),
        func.avg(DetectionLog.confidence).label('avg_conf'),
        func.min(DetectionLog.detection_time).label('first'),
        func.max(DetectionLog.detection_time).label('last')
    ).first()
    
    unique_personnel = query.with_entities(DetectionLog.person).distinct().count()
    
    return {
        "total_detections": stats.total or 0,
        "unique_personnel": unique_personnel,
        "average_confidence": round(stats.avg_conf, 2) if stats.avg_conf else 0,
        "first_detection": stats.first.isoformat() if stats.first else None,
        "last_detection": stats.last.isoformat() if stats.last else None
    }


@router.get("/{personnel_id}", response_model=PersonnelSchema)
def read_personnel_by_id(
    personnel_id: int,
    include_rooms: bool = Query(True),
    db: Session = Depends(get_db)
):
    """Get personnel by ID with their rooms"""
    query = db.query(PersonnelDB)
    
    if include_rooms:
        query = query.options(joinedload(PersonnelDB.rooms))
    
    db_personnel = query.filter(PersonnelDB.id == personnel_id).first()
    
    if db_personnel is None:
        raise HTTPException(status_code=404, detail="Personnel not found")
    return db_personnel



# @router.get("/national-code/{national_code}", response_model=PersonnelSchema)
# def read_personnel_by_national(national_code: str, db: Session = Depends(get_db)):
#     db_personnel = db.query(Personnel).filter(Personnel.national_code == national_code).first()
#     if db_personnel is None:
#         raise HTTPException(status_code=404, detail="Personnel not found")
#     return db_personnel

@router.put("/{personnel_id}", response_model=PersonnelSchema)
def update_personnel(personnel_id: int, personnel_update: PersonnelUpdate, db: Session = Depends(get_db)):
    db_personnel = db.query(PersonnelDB).filter(PersonnelDB.id == personnel_id).first()
    if db_personnel is None:
        raise HTTPException(status_code=404, detail="Personnel not found")
    
    update_data = personnel_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_personnel, field, value)
    
    db.commit()
    db.refresh(db_personnel)
    return db_personnel



@router.delete("/{personnel_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_personnel(personnel_id: int, db: Session = Depends(get_db)):
    db_personnel = db.query(PersonnelDB).filter(PersonnelDB.id == personnel_id).first()
    if db_personnel is None:
        raise HTTPException(status_code=404, detail="Personnel not found")
    
    images = db.query(PersonnelImage).filter(PersonnelImage.personnel_id == personnel_id).all()
    
    # Delete each image from vector database first
    for image in images:
        print(f"\n🗑️ Deleting face embeddings for image ID: {image.id}")
        # delete_faces_from_vector_database(image.id)
        model_mgr.DeletePointVD(image.id)
    
    # Also delete physical files (optional - you might want to do this too)
    from pathlib import Path
    for image in images:
        try:
            file_path = Path(image.image_url)
            if file_path.exists():
                file_path.unlink()
                print(f"✅ Deleted image file: {file_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not delete file {image.image_url}: {e}")


    db.delete(db_personnel)
    db.commit()
    return None

#sdfsdfsdf
@router.delete("/images/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_personnel_image(
    image_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a specific personnel image
    - Removes database record
    - Deletes physical file from disk
    - Removes from vector database using ref_img_id
    """

    # Find the image
    image = db.query(PersonnelImage).filter(PersonnelImage.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Get information before deletion
    file_path = Path(image.image_url)
    
    print(f"\n🗑️ Deleting image ID: {image_id}")
    
    # STEP 1: Delete from vector database
    model_mgr.DeletePointVD(image_id)
    
    # STEP 2: Delete from database
    try:
        db.delete(image)
        db.commit()
        print(f"✅ Deleted image record from database")
    except Exception as e:
        db.rollback()
        print(f"❌ Database deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete image record: {str(e)}")
    
    # STEP 3: Delete physical file
    try:
        if file_path.exists():
            file_path.unlink()
            print(f"✅ Deleted image file: {file_path}")
        
        # Clean up empty folder
        parent_folder = file_path.parent
        if parent_folder.exists() and not any(parent_folder.iterdir()):
            parent_folder.rmdir()
            print(f"✅ Deleted empty folder: {parent_folder}")
            
    except Exception as e:
        print(f"⚠️ Warning: Could not delete file/folder: {e}")
    
    return None



    
@router.get("/{personnel_id}/with-images", response_model=PersonnelWithImages)
async def get_personnel_with_images(
    request: Request,
    personnel_id: int,
    db: Session = Depends(get_db)
):
    """
    Get personnel details along with all their images and rooms
    """
    # Get personnel with images and rooms (eager loading)
    personnel = db.query(Personnel)\
        .options(joinedload(Personnel.images))\
        .options(joinedload(Personnel.rooms))\
        .filter(Personnel.id == personnel_id)\
        .first()

    if not personnel:
        raise HTTPException(status_code=404, detail=f"Personnel with ID {personnel_id} not found")

    # Convert image paths to URLs
    base_url = str(request.base_url).rstrip('/')
    images_response = []

    for img in personnel.images:
        images_response.append(
            PersonnelImageResponse(
                id=img.id,
                image_url=f"{base_url}/api/v1/personnel/{personnel_id}/images/{img.id}/file",
                personnel_id=img.personnel_id,
                uploaded_at=img.uploaded_at if hasattr(img, 'uploaded_at') else None
            )
        )

    # Convert rooms to response schema
    rooms_response = [
        RoomResponse(
            id=room.id,
            room_number=room.room_number,
            room_name=room.room_name,
            room_type=room.room_type,
            capacity=room.capacity,
            description=room.description,
            is_active=room.is_active,
            created_at=room.created_at,
            updated_at=room.updated_at
        )
        for room in personnel.rooms
    ]

    return PersonnelWithImages(
        id=personnel.id,
        fname=personnel.fname,
        lname=personnel.lname,
        national_code=personnel.national_code,
        staff=personnel.staff,
        department=personnel.department,
        created_at=personnel.created_at,
        rooms=rooms_response,  # Include rooms
        images=images_response
    )




# Optional: Get only rooms for a personnel
# ==================== ROOM-RELATED PERSONNEL ENDPOINTS ====================

@router.get("/{personnel_id}/rooms", response_model=List[RoomResponse])
def get_personnel_rooms_endpoint(
    personnel_id: int,
    db: Session = Depends(get_db)
):
    """Get all rooms a personnel has access to"""
    # Use PersonnelDB (SQLAlchemy model) for query
    personnel = db.query(PersonnelDB).filter(PersonnelDB.id == personnel_id).first()
    if not personnel:
        raise HTTPException(status_code=404, detail="Personnel not found")
    
    # Convert SQLAlchemy rooms to Pydantic RoomResponse
    return [
        RoomResponse(
            id=room.id,
            room_number=room.room_number,
            room_name=room.room_name,
            room_type=room.room_type,
            capacity=room.capacity,
            description=room.description,
            is_active=room.is_active,
            created_at=room.created_at,
            updated_at=room.updated_at
        )
        for room in personnel.rooms
    ]


@router.get("/by-room/{room_id}", response_model=List[PersonnelSchema])
def get_personnel_by_room(
    room_id: int,
    db: Session = Depends(get_db)
):
    """Get all personnel who have access to a specific room"""
    # Use PersonnelDB (SQLAlchemy model) for query
    personnel_list = db.query(PersonnelDB)\
        .join(PersonnelDB.rooms)\
        .filter(RoomDB.id == room_id, RoomDB.is_active == True)\
        .all()
    
    return personnel_list





@router.get("/{personnel_id}/logs", response_model=List[DetectionLogResponse])
def get_personnel_logs_by_id(
    personnel_id: int,
    request: Request,
    start_date: Optional[datetime] = Query(None, description="Filter logs from this date (YYYY-MM-DDTHH:MM:SS)"),
    end_date: Optional[datetime] = Query(None, description="Filter logs until this date (YYYY-MM-DDTHH:MM:SS)"),
    db: Session = Depends(get_db)
):
    """
    Get all detection logs for a specific personnel by their personnel ID
    Optional date filtering with start_date and end_date parameters
    """
    
    # Get personnel by ID
    # personnel = db.query({PersonnelDB}).filter({PersonnelDB}.id == personnel_id).first()
    personnel = db.query(PersonnelDB).filter(PersonnelDB.id == personnel_id).first()
    if not personnel:
        raise HTTPException(status_code=404, detail=f"Personnel with ID {personnel_id} not found")
    
    # Build query for detection logs
    query = db.query(DetectionLog).filter(DetectionLog.person == personnel.national_code)
    
    # Apply date filters if provided
    if start_date:
        query = query.filter(DetectionLog.detection_time >= start_date)
    if end_date:
        query = query.filter(DetectionLog.detection_time <= end_date)
    
    # Order by detection time (newest first)
    logs = query.order_by(DetectionLog.detection_time.desc()).all()
    
    # Get the image URL function (assuming it exists in your utilities)
    
    # Create response for each log
    result = []
    for log in logs:
        log_response = DetectionLogResponse(
            id=log.id,
            person=log.person,
            confidence=float(log.confidence) if log.confidence else None,
            detection_time=log.detection_time,
            face_image_url=get_face_image_url(request, log.id) if log.face_image_path else None,
            video_url=get_video_url(request, log.id) if log.video_path else None,
            fname=personnel.fname,
            lname=personnel.lname,
            full_name=f"{personnel.fname} {personnel.lname}".strip()
        )
        result.append(log_response)
    
    return result


@router.get("/by-national-code/{national_code}/logs", response_model=List[DetectionLogResponse])
def get_personnel_logs_by_national_code(
    national_code: str,
    request: Request,
    start_date: Optional[datetime] = Query(None, description="Filter logs from this date (YYYY-MM-DDTHH:MM:SS)"),
    end_date: Optional[datetime] = Query(None, description="Filter logs until this date (YYYY-MM-DDTHH:MM:SS)"),
    db: Session = Depends(get_db)
):
    """
    Get all detection logs for a specific personnel by their national code
    Optional date filtering with start_date and end_date parameters
    """
    
    # First normalize/validate the national code
    try:
        # If you have normalize function, use it, otherwise just strip
        normalized_code = national_code.strip()
        if not validate_iran_national_code(normalized_code):
            raise HTTPException(status_code=400, detail="کد ملی نامعتبر است")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid national code format: {str(e)}")
    
    # Get personnel by national code
    personnel = db.query(PersonnelDB).filter(PersonnelDB.national_code == normalized_code).first()
    if not personnel:
        raise HTTPException(status_code=404, detail=f"Personnel with national code {national_code} not found")
    
    # Build query for detection logs
    query = db.query(DetectionLog).filter(DetectionLog.person == normalized_code)
    
    # Apply date filters if provided
    if start_date:
        query = query.filter(DetectionLog.detection_time >= start_date)
    if end_date:
        query = query.filter(DetectionLog.detection_time <= end_date)
    
    # Order by detection time (newest first)
    logs = query.order_by(DetectionLog.detection_time.desc()).all()
    
    # Create response for each log
    result = []
    for log in logs:
        log_response = DetectionLogResponse(
            id=log.id,
            person=log.person,
            confidence=float(log.confidence) if log.confidence else None,
            detection_time=log.detection_time,
            face_image_url=get_face_image_url(request, log.id) if log.face_image_path else None,
            video_url=get_video_url(request, log.id) if log.video_path else None,
            fname=personnel.fname,
            lname=personnel.lname,
            full_name=f"{personnel.fname} {personnel.lname}".strip()
        )
        result.append(log_response)
    
    return result



import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)




@router.get("/{personnel_id}/logs/summary")
def get_personnel_logs_summary(
    personnel_id: int,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: Session = Depends(get_db)
):
    """Get summary statistics of detection logs for a specific personnel"""
    from sqlalchemy import func
    
    personnel = db.query(PersonnelDB).filter(PersonnelDB.id == personnel_id).first()
    if not personnel:
        raise HTTPException(status_code=404, detail="Personnel not found")
    
    query = db.query(DetectionLog).filter(DetectionLog.person == personnel.national_code)
    
    if start_date:
        query = query.filter(DetectionLog.detection_time >= start_date)
    if end_date:
        query = query.filter(DetectionLog.detection_time <= end_date)
    
    stats = query.with_entities(
        func.count(DetectionLog.id).label('total'),
        func.avg(DetectionLog.confidence).label('avg_conf'),
        func.min(DetectionLog.detection_time).label('first'),
        func.max(DetectionLog.detection_time).label('last')
    ).first()
    
    return {
        "personnel_id": personnel_id,
        "full_name": f"{personnel.fname} {personnel.lname}".strip(),
        "national_code": personnel.national_code,
        "total_detections": stats.total or 0,
        "average_confidence": round(stats.avg_conf, 2) if stats.avg_conf else 0,
        "first_detection": stats.first,
        "last_detection": stats.last,
        "date_range_days": (stats.last - stats.first).days if stats.first and stats.last else 0
    }





@router.post("/{personnel_id}/images", response_model=PersonnelWithImages, status_code=status.HTTP_201_CREATED)
async def add_images_to_personnel(
    request: Request,
    personnel_id: int,
    images: List[UploadFile] = File(...),
    is_primary: Optional[bool] = Form(False),  # ADD is_primary parameter
    db: Session = Depends(get_db)
):
    """
    Add multiple images to an existing personnel
    """
    # Check if personnel exists
    personnel = db.query(PersonnelDB).filter(PersonnelDB.id == personnel_id).first()
    if not personnel:
        raise HTTPException(status_code=404, detail=f"Personnel with ID {personnel_id} not found")
    
    # Validate at least one image is provided
    if not images or len(images) == 0:
        raise HTTPException(status_code=400, detail="حد اقل یک تصویر نیاز است!")
    
    # Validate file types
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    processed_images = []
    
    for image in images:
        file_extension = os.path.splitext(image.filename)[1].lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"نوع داده غیر مجاز! '{image.filename}' انواع داده مجاز: {', '.join(allowed_extensions)}"
            )
        
        # Read and validate image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail=f"Could not read image: {image.filename}")
        
        processed_images.append({
            'file': image,
            'filename': image.filename,
            'contents': contents,
            'img': img
        })
    
    try:
        # Create personnel images directory
        personnel_images_dir = FACE_STORAGE_DIR / "personnel" / str(personnel_id)
        personnel_images_dir.mkdir(parents=True, exist_ok=True)
        
        saved_images = []
        
        # Save each image and create database records
        for idx, img_data in enumerate(processed_images):
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            file_extension = os.path.splitext(img_data['filename'])[1].lower()
            safe_filename = f"{personnel.fname}_{personnel.lname}_{timestamp}_{unique_id}{file_extension}"
            safe_filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe_filename)
            
            # Save file
            file_path = personnel_images_dir / safe_filename
            
            with open(file_path, "wb") as buffer:
                buffer.write(img_data['contents'])  # Use stored contents instead of seeking
            
            # Determine if this should be primary
            is_primary_img = is_primary and idx == 0  # Only first image if is_primary=True
            
            # If setting as primary, unset other primary flags
            if is_primary_img:
                db.query(PersonnelImage).filter(
                    PersonnelImage.personnel_id == personnel_id,
                    PersonnelImage.is_primary == True
                ).update({PersonnelImage.is_primary: False})
            
            # Create database record for image
            db_image = PersonnelImage(
                image_url=str(file_path),
                personnel_id=personnel_id,
                is_primary=is_primary_img  # ADD is_primary
            )
            
            db.add(db_image)
            db.flush()  # Get ID for this image
            
            # Add to vector database with ref_img_id
            try:
                # Make sure model_mgr is imported/available
                
                # from .model_manager import model_mgr  # Adjust import as needed
                model_mgr.FaceEmbeddingWithoutDetection(
                    img_data['img'], 
                    personnel.national_code, 
                    ref_img_id=db_image.id
                )
                print(f"✅ Added to vector DB with ref_img_id: {db_image.id}")
            except Exception as e:
                print(f"⚠️ Warning: Vector DB insertion failed for image {db_image.id}: {e}")
                # Continue even if vector DB fails
            
            saved_images.append(db_image)
        
        # Commit all changes
        db.commit()
        
        # Refresh to get updated relationships
        db.refresh(personnel)
        
        # Generate URLs for all images
        base_url = str(request.base_url).rstrip('/')
        all_images = []
        primary_image_url = None
        
        # Get all images for this personnel (both existing and new)
        all_images_db = db.query(PersonnelImage).filter(
            PersonnelImage.personnel_id == personnel_id
        ).order_by(PersonnelImage.is_primary.desc(), PersonnelImage.id.desc()).all()
        
        for img in all_images_db:
            image_response = PersonnelImageResponse(
                id=img.id,
                image_url=f"{base_url}/api/v1/personnel/{personnel_id}/images/{img.id}/file",
                personnel_id=img.personnel_id,
                is_primary=img.is_primary,  # ADD is_primary
                uploaded_at=img.uploaded_at if hasattr(img, 'uploaded_at') else None
            )
            all_images.append(image_response)
            
            if img.is_primary:
                primary_image_url = image_response.image_url
        
        # If no primary image set and there are images, set first as primary
        if not primary_image_url and all_images:
            primary_image_url = all_images[0].image_url
        
        return PersonnelWithImages(
            id=personnel.id,
            fname=personnel.fname,
            lname=personnel.lname,
            national_code=personnel.national_code,
            staff=personnel.staff,
            department=personnel.department,
            created_at=personnel.created_at,
            rooms=[],  # Add rooms if needed
            images=all_images,
            primary_image=primary_image_url  # ADD primary_image
        )
        
    except Exception as e:
        # Rollback database changes
        db.rollback()
        
        # Clean up any saved files for this batch
        if 'personnel_images_dir' in locals() and personnel_images_dir.exists():
            for img in saved_images:
                try:
                    if hasattr(img, 'image_url') and Path(img.image_url).exists():
                        Path(img.image_url).unlink()
                except:
                    pass
        
        print(f"Error adding images to personnel: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to add images: {str(e)}")
    




# creating personnel with increamental images
@router.post("/with-images", response_model=PersonnelWithImages, status_code=status.HTTP_201_CREATED)
async def create_personnel_with_images(
    request: Request,
    fname: str = Form(...),
    lname: str = Form(...),
    national_code: str = Form(...),
    staff: Optional[bool] = Form(None),
    department: Optional[str] = Form(None),
    images: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Create a new personnel with multiple images in one request
    """
    # Check if national code already exists
    existing = db.query(PersonnelDB).filter(PersonnelDB.national_code == national_code).first()
    if existing:
        raise HTTPException(status_code=400, detail="کد ملی موجود است!")
    
    # Validate at least one image is provided
    if not images or len(images) == 0:
        raise HTTPException(status_code=400, detail="حد اقل یک تصویر نیاز است!")
    
    # Create new personnel
    db_personnel = PersonnelDB(
        fname=fname,
        lname=lname,
        national_code=national_code,
        staff=staff,
        department=department
    )
    
    db.add(db_personnel)
    db.flush()  # This assigns an ID to db_personnel without committing
    
    try:
        # Create personnel images directory
        personnel_images_dir = FACE_STORAGE_DIR / "personnel" / str(db_personnel.id)
        personnel_images_dir.mkdir(parents=True, exist_ok=True)
        
        saved_images = []
        
        # Save each image
        for image in images:
            # Validate file type
            allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
            file_extension = os.path.splitext(image.filename)[1].lower()
            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"نوع داده غیر مجاز! '{image.filename}' انواع داده مجاز: {', '.join(allowed_extensions)}"
                )
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            safe_filename = f"{fname}_{lname}_{timestamp}_{unique_id}{file_extension}"
            safe_filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe_filename)
            
            # Save file
            file_path = personnel_images_dir / safe_filename
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            
            # Create database record for image
            db_image = PersonnelImage(
                image_url=str(file_path),
                personnel_id=db_personnel.id
            )
            db.add(db_image)
            db.flush()  # Get ID for this image
            
            # ✅ ADD THIS: Create embedding in Qdrant
            img = cv2.imread(str(file_path))
            if img is None:
                raise HTTPException(status_code=400, detail=f"Cannot read image: {image.filename}")
            
            # Call the correct function to create Qdrant points
            model_mgr.FaceEmbeddingWithoutDetection(img, national_code, ref_img_id=db_image.id)
            
            saved_images.append(db_image)
        
        # Commit all changes
        db.commit()
        
        # Refresh to get updated relationships
        db.refresh(db_personnel)
        
        # Generate URLs for images
        base_url = str(request.base_url).rstrip('/')
        images_response = []
        
        for img in saved_images:
            images_response.append(
                PersonnelImageResponse(
                    id=img.id,
                    image_url=f"{base_url}/api/v1/personnel/{db_personnel.id}/images/{img.id}/file",
                    personnel_id=img.personnel_id,
                    uploaded_at=img.uploaded_at if hasattr(img, 'uploaded_at') else None
                )
            )
        
        return PersonnelWithImages(
            id=db_personnel.id,
            fname=db_personnel.fname,
            lname=db_personnel.lname,
            national_code=db_personnel.national_code,
            staff=db_personnel.staff,
            department=db_personnel.department,
            created_at=db_personnel.created_at,
            images=images_response
        )
        
    except Exception as e:
        # Rollback database changes
        db.rollback()
        
        # Clean up any saved files
        if 'personnel_images_dir' in locals() and personnel_images_dir.exists():
            shutil.rmtree(personnel_images_dir)
        
        print(f"Error creating personnel with images: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create personnel: {str(e)}")




@router.get("/{personnel_id}/images", response_model=List[PersonnelImageResponse])
async def get_personnel_images(
    personnel_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Get all images for a specific personnel
    """
    # Check if personnel exists
    personnel = db.query(PersonnelDB).filter(PersonnelDB.id == personnel_id).first()
    if not personnel:
        raise HTTPException(status_code=404, detail=f"Personnel with ID {personnel_id} not found")
    
    # Get images
    images = db.query(PersonnelImage).filter(
        PersonnelImage.personnel_id == personnel_id
    ).order_by(PersonnelImage.id.desc()).all()
    
    # Convert file paths to URLs
    base_url = str(request.base_url).rstrip('/')
    result = []
    for img in images:
        result.append(
            PersonnelImageResponse(
                id=img.id,
                image_url=f"{base_url}/api/v1/personnel/{personnel_id}/images/{img.id}/file",
                personnel_id=img.personnel_id,
                uploaded_at=img.uploaded_at if hasattr(img, 'uploaded_at') else None
            )
        )
    
    return result




@router.get("/{personnel_id}/with-images", response_model=PersonnelWithImages)
async def get_personnel_with_images(
    request: Request,
    personnel_id: int,
    db: Session = Depends(get_db)
):
    """
    Get personnel details along with all their images
    """
    # Get personnel with images
    personnel = db.query(PersonnelDB).filter(PersonnelDB.id == personnel_id).first()
    
    if not personnel:
        raise HTTPException(status_code=404, detail=f"Personnel with ID {personnel_id} not found")
    
    # Convert image paths to URLs
    base_url = str(request.base_url).rstrip('/')
    images_response = []
    
    for img in personnel.images:
        images_response.append(
            PersonnelImageResponse(
                id=img.id,
                image_url=f"{base_url}/api/v1/personnel/{personnel_id}/images/{img.id}/file",
                personnel_id=img.personnel_id,
                uploaded_at=img.uploaded_at if hasattr(img, 'uploaded_at') else None
            )
        )
    
    return PersonnelWithImages(
        id=personnel.id,
        fname=personnel.fname,
        lname=personnel.lname,
        national_code=personnel.national_code,
        staff=personnel.staff,
        department=personnel.department,
        created_at=personnel.created_at,
        images=images_response
    )



from typing import Optional

@router.get("/images", response_model=List[PersonnelImageResponse])
async def get_all_images(
    request: Request,
    skip: int = 0,
    limit: int = 100,
    personnel_id: Optional[int] = None,  # Optional filter by personnel
    db: Session = Depends(get_db)
):
    """
    Get all images with optional filtering
    - skip: Number of records to skip (pagination)
    - limit: Maximum number of records to return
    - personnel_id: Filter by specific personnel (optional)
    """
    query = db.query(PersonnelImage)
    
    # Apply filter if personnel_id is provided
    if personnel_id:
        query = query.filter(PersonnelImage.personnel_id == personnel_id)
    
    # Get images with pagination
    images = query.order_by(PersonnelImage.id.desc()).offset(skip).limit(limit).all()
    
    # Generate URLs
    base_url = str(request.base_url).rstrip('/')
    result = []
    
    for img in images:
        result.append(
            PersonnelImageResponse(
                id=img.id,
                image_url=f"{base_url}/api/v1/personnel/{img.personnel_id}/images/{img.id}/file",
                personnel_id=img.personnel_id,
                uploaded_at=img.uploaded_at if hasattr(img, 'uploaded_at') else None
            )
        )
    
    return result











@router.get("/{personnel_id}/images/{image_id}/file")
async def serve_personnel_image(
    # personnel_id: int,
    image_id: int,
    db: Session = Depends(get_db)
):
    """
    Serve the actual image file
    """
    # Get image record
    image = db.query(PersonnelImage).filter(
        PersonnelImage.id == image_id,
        # PersonnelImage.personnel_id == personnel_id
    ).first()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Check if file exists
    if not os.path.exists(image.image_url):
        raise HTTPException(status_code=404, detail="Image file not found on disk")
    
    # Determine media type based on file extension
    file_extension = os.path.splitext(image.image_url)[1].lower()
    media_type = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp'
    }.get(file_extension, 'application/octet-stream')
    
    return FileResponse(
        path=image.image_url,
        media_type=media_type,
        filename=os.path.basename(image.image_url)
    )


@router.post("/from-log",response_model=List[PersonnelWithImages],
    status_code=status.HTTP_201_CREATED
)
async def create_personnel_from_multiple_logs(
    request: Request,
    data: PersonnelFromLogsRequest,
    db: Session = Depends(get_db)
    ):
    
    fname = data.fname
    lname = data.lname
    national_code = data.national_code
    staff = data.staff
    department = data.department
    log_ids = data.log_ids


   
    """
    Create a new personnel OR attach images to an existing personnel
    for multiple detection logs.
    Each log generates one new PersonnelImage.
    """

    # Check if personnel already exists
    existing = db.query(PersonnelDB).filter(PersonnelDB.national_code == national_code).first()

    # If not exists → create personnel
    if not existing:
        db_personnel = PersonnelDB(
            fname=fname,
            lname=lname,
            national_code=national_code,
            staff=staff,
            department=department
        )
        db.add(db_personnel)
        db.flush()   # get db_personnel.id
    else:
        db_personnel = existing

    personnel_id = db_personnel.id

    # Ensure folder exists
    personnel_images_dir = FACE_STORAGE_DIR / "personnel" / str(personnel_id)
    personnel_images_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for log_id in log_ids:

        detection_log = db.query(DetectionLog).filter(DetectionLog.id == log_id).first()
        if not detection_log:
            raise HTTPException(status_code=404, detail=f"Detection log {log_id} not found")

        if not detection_log.face_image_path:
            raise HTTPException(status_code=400, detail=f"Log {log_id} has no face image")

        face_image_path = Path(detection_log.face_image_path)
        if not face_image_path.exists():
            raise HTTPException(status_code=404, detail=f"Face image for log {log_id} not found")

        # Validate extension
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        ext = face_image_path.suffix.lower()
        if ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Invalid extension for log {log_id}")

        # Copy image to personnel folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        safe_name = f"{fname}_{lname}_log{log_id}_{timestamp}_{unique_id}{ext}"
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe_name)

        save_path = personnel_images_dir / safe_name
        shutil.copy2(face_image_path, save_path)

        # Create DB Image record
        db_image = PersonnelImage(
            image_url=str(save_path),
            personnel_id=personnel_id
        )
        db.add(db_image)
        db.flush()

        # Read and embed face
        img = cv2.imread(str(face_image_path))
        if img is None:
            raise HTTPException(status_code=400, detail=f"Cannot read image for log {log_id}")

        model_mgr.FaceEmbeddingWithoutDetection(img, national_code, ref_img_id=db_image.id)

        db.commit()
        db.refresh(db_personnel)

        # Build response object for this log
        base_url = str(request.base_url).rstrip("/")
        image_response = PersonnelImageResponse(
            id=db_image.id,
            image_url=f"{base_url}/api/v1/personnel/{personnel_id}/images/{db_image.id}/file",
            personnel_id=personnel_id,
            uploaded_at=db_image.uploaded_at
        )

        # Mark log as processed
        detection_log.personnel_id = personnel_id
        detection_log.is_processed = True
        db.add(detection_log)
        db.commit()

        # Add to results
        results.append(
            PersonnelWithImages(
                id=personnel_id,
                fname=db_personnel.fname,
                lname=db_personnel.lname,
                national_code=db_personnel.national_code,
                staff=db_personnel.staff,
                department=db_personnel.department,
                created_at=db_personnel.created_at,
                images=[image_response]
            )
        )

    return results

    



@router.post("/upload-personnel-zip")
async def upload_personnel_zip(
    file: UploadFile = File(...),
    skip_invalid_national_codes: bool = False,
    db: Session = Depends(get_db)  # Add database session
):
    # Validate file type
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only zip files are accepted")
    
    # Allowed image extensions
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    try:
        contents = await file.read()
        print(f"📦 Zip file size: {len(contents)} bytes")
        zip_data = BytesIO(contents)
        
        if not zipfile.is_zipfile(zip_data):
            raise HTTPException(status_code=400, detail="Invalid zip file")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read zip file: {str(e)}")
    
    processed_count = 0
    error_count = 0
    results = []
    skipped_folders = []
    all_saved_images = []  # Track all saved images for response
    
    with zipfile.ZipFile(zip_data, 'r') as zip_ref:
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"\n📂 Extracting to: {temp_dir}")
            zip_ref.extractall(temp_dir)
            temp_path = Path(temp_dir)
            
            for person_dir in temp_path.iterdir():
                if not person_dir.is_dir() or person_dir.name == '__MACOSX':
                    continue
                
                national_code = person_dir.name
                print(f"\n{'='*50}")
                print(f"👤 Processing person with national code: {national_code}")
                
                # Validate national code format
                is_valid_national_code = national_code.isdigit() and len(national_code) == 10
                
                if not is_valid_national_code:
                    if skip_invalid_national_codes:
                        print(f"⚠️ Invalid national code - SKIPPING")
                        skipped_folders.append({
                            "folder": national_code,
                            "reason": "Invalid national code format"
                        })
                        continue
                
                # Check if personnel exists in database, if not create it
                personnel = db.query(PersonnelDB).filter(PersonnelDB.national_code == national_code).first()
                
                if not personnel:
                    # Create new personnel if it doesn't exist
                    print(f"👤 Personnel not found, creating new record")
                    personnel = PersonnelDB(
                        fname=f"فرد_{national_code}",  # Placeholder name
                        lname="",
                        national_code=national_code,
                        staff=False,
                        department=None
                    )
                    db.add(personnel)
                    db.flush()  # Get ID without committing
                    print(f"✅ Created new personnel with ID: {personnel.id}")
                else:
                    print(f"✅ Found existing personnel with ID: {personnel.id}")
                
                # Create personnel images directory
                personnel_images_dir = FACE_STORAGE_DIR / "personnel" / str(personnel.id)
                personnel_images_dir.mkdir(parents=True, exist_ok=True)
                
                # Get unique image files
                image_files = set()
                for ext in ALLOWED_EXTENSIONS:
                    for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                        for img_path in person_dir.glob(pattern):
                            if not img_path.name.startswith('._'):
                                image_files.add(img_path)
                
                image_files = list(image_files)
                print(f"  🖼️ Found {len(image_files)} unique images")
                
                if not image_files:
                    results.append({
                        "national_code": national_code,
                        "status": "warning",
                        "message": "No images found in folder",
                        "valid_format": is_valid_national_code
                    })
                    continue
                
                person_processed = 0
                person_errors = 0
                error_details = []
                saved_images = []
                
                for img_path in image_files:
                    try:
                        print(f"\n  📸 Processing: {img_path.name}")
                        
                        # Read image
                        img = cv2.imread(str(img_path))
                        if img is None:
                            raise ValueError("Could not decode image")
                        
                        # Generate filename for saving
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        unique_id = str(uuid.uuid4())[:8]
                        safe_filename = f"{personnel.fname}_{timestamp}_{unique_id}{img_path.suffix.lower()}"
                        safe_filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe_filename)
                        
                        file_path = personnel_images_dir / safe_filename
                        
                        # Copy file to permanent storage
                        
                        # Create database record
                        db_image = PersonnelImage(
                            image_url=str(file_path),
                            personnel_id=personnel.id
                        )
                        db.add(db_image)
                        db.flush()  # Get ID without committing
                        
                        print(f"     🆔 Database record created with ID: {db_image.id}")
                        
                        # Save to vector database with ref_img_id
                        print(f"     🔄 Adding to vector database with ref_img_id: {db_image.id}")
                        cropped_img = model_mgr.FaceEmbeddingCropping(img, national_code,  ref_img_id=db_image.id)
                        cv2.imwrite(str(file_path), cropped_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                      
                        saved_images.append({
                            "id": db_image.id,
                            "file_name": safe_filename,
                            "file_path": str(file_path)
                        })
                        
                        print(f"  ✅ Successfully processed: {img_path.name}")
                        person_processed += 1
                        
                    except Exception as e:
                        print(f"  ❌ Error processing {img_path.name}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        person_errors += 1
                        error_details.append({
                            "file": img_path.name,
                            "error": str(e)
                        })
                
                # Commit all changes for this personnel
                db.commit()
                
                processed_count += person_processed
                error_count += person_errors
                
                results.append({
                    "national_code": national_code,
                    "personnel_id": personnel.id,
                    "status": "success" if person_processed > 0 else "error",
                    "valid_format": is_valid_national_code,
                    "processed": person_processed,
                    "errors": person_errors,
                    "total_images": len(image_files),
                    "saved_images": saved_images,
                    "error_details": error_details if error_details else None
                })
                
                all_saved_images.extend(saved_images)
                print(f"\n  📊 Summary for {national_code}: {person_processed}/{len(image_files)} processed")
    
    print(f"\n{'='*50}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"Total processed: {processed_count}")
    print(f"Total errors: {error_count}")
    print(f"Total persons: {len(results)}")
    print(f"Total images saved: {len(all_saved_images)}")
    
    return {
        "success": True,
        "filename": file.filename,
        "message": f"Processed {processed_count} images, {error_count} errors",
        "summary": {
            "total_processed": processed_count,
            "total_errors": error_count,
            "total_persons": len(results),
            "total_images_saved": len(all_saved_images),
            "skipped_folders": len(skipped_folders)
        },
        "details": results,
        "skipped_folders": skipped_folders if skipped_folders else None
    }


# Add these new endpoints for personnel-room relationships
@router.get("/{personnel_id}/rooms")
def get_personnel_rooms_list(
    personnel_id: int,
    db: Session = Depends(get_db)
):
    """Get all rooms a personnel has access to"""
    rooms = get_personnel_rooms(personnel_id=personnel_id, db=db)
    return rooms