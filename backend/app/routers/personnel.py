from fastapi import (
    APIRouter, 
    Depends, 
    HTTPException, 
    status ,
    UploadFile, 
    Query, 
    Request, 
    File, 
    Response
)
from io import BytesIO
import re
from sqlalchemy.orm import Session
from typing import List
from ..models.database import Personnel, get_db
from ..models.schemas import Personnel as PersonnelSchema, PersonnelCreate, PersonnelUpdate
router = APIRouter(prefix="/personnel", tags=["personnel"])
from ..models.database import PersonnelImage
from pathlib import Path
from qdrant_client.http import models
from Face_ai.main import client
from backend.app.models.schemas import (
    DetectionLogCreate, DetectionLogResponse, 
    PersonnelImageCreate, PersonnelImageResponse, PersonnelWithImages
)
from backend.app.models.database import FACE_STORAGE_DIR
import pandas as pd
from ..validators import normalize_national_code, validate_iran_national_code
import io
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
    

from fastapi.responses import FileResponse
import tempfile
import os



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
    existing = db.query(Personnel).filter(Personnel.national_code == personnel.national_code).first()
    if existing:
        raise HTTPException(status_code=400, detail="کد ملی موجود است!")
    
    db_personnel = Personnel(**personnel.dict())
    db.add(db_personnel)
    db.commit()
    db.refresh(db_personnel)
    return db_personnel

@router.get("/", response_model=List[PersonnelSchema])
def read_personnel(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    personnel = db.query(Personnel).order_by(Personnel.id).offset(skip).limit(limit).all()
    return personnel

@router.get("/{personnel_id}", response_model=PersonnelSchema)
def read_personnel(personnel_id: int, db: Session = Depends(get_db)):
    db_personnel = db.query(Personnel).filter(Personnel.id == personnel_id).first()
    if db_personnel is None:
        raise HTTPException(status_code=404, detail="Personnel not found")
    return db_personnel



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
    personnel = db.query(Personnel).filter(Personnel.id == personnel_id).first()
    
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




@router.put("/{personnel_id}", response_model=PersonnelSchema)
async def update_personnel(
    personnel_id: int,
    personnel_update: PersonnelUpdate,
    db: Session = Depends(get_db)
):
    # Get existing personnel
    personnel = db.query(Personnel).filter(Personnel.id == personnel_id).first()
    if not personnel:
        raise HTTPException(status_code=404, detail="Personnel not found")
    
    # Check unique national code ONLY if it's being updated
    if personnel_update.national_code and personnel_update.national_code != personnel.national_code:
        existing = db.query(Personnel).filter(
            Personnel.national_code == personnel_update.national_code,
            Personnel.id != personnel_id  # Exclude current record
        ).first()
        if existing:
            raise HTTPException(
                status_code=400, 
                detail="کد ملی موجود است!"
            )
    
    # Update fields
    for field, value in personnel_update.dict(exclude_unset=True).items():
        setattr(personnel, field, value)
    
    db.commit()
    db.refresh(personnel)
    return personnel

@router.delete("/{personnel_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_personnel(personnel_id: int, db: Session = Depends(get_db)):
    db_personnel = db.query(Personnel).filter(Personnel.id == personnel_id).first()
    if db_personnel is None:
        raise HTTPException(status_code=404, detail="Personnel not found")
    
    images = db.query(PersonnelImage).filter(PersonnelImage.personnel_id == personnel_id).all()
    
    # Delete each image from vector database first
    for image in images:
        print(f"\n🗑️ Deleting face embeddings for image ID: {image.id}")
        delete_faces_from_vector_database(image.id)
    
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


def delete_faces_from_vector_database(ref_img_id: int, collection_name: str = "n12") -> bool:

    try:
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="ref_img_id",
                    match=models.MatchValue(value=ref_img_id)
                )
            ]
        )
        
        # First, count how many points will be deleted (optional)
        count_result = client.count(
            collection_name=collection_name,
            count_filter=filter_condition
        )
        print(f"   Found {count_result.count} points in vector DB for ref_img_id: {ref_img_id}")
        
        delete_result = client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=filter_condition
            )
        )
        
        print(f"✅ Deleted from vector database: {delete_result}")
        return True
        
    except Exception as e:
        print(f"⚠️ Error deleting from vector database: {e}")
        return False
    



