from fastapi import APIRouter, Depends, Request, Query, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, date, time, timedelta
from sqlalchemy import func
from pydantic import BaseModel
from fastapi.responses import FileResponse, StreamingResponse
import os
from .routers.personnel import delete_faces_from_vector_database
from .models.database import DetectionLog, get_db
from .models.schemas import (
    DetectionLogCreate, DetectionLogResponse, 
    PersonnelImageCreate, PersonnelImageResponse, PersonnelWithImages
)
import traceback

from io import BytesIO
import zipfile
import tempfile

import uuid  # Add this import
from Face_ai.main import FaceEmbedding
from .routers import personnel
from fastapi import APIRouter, Depends, Request, Query, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, date, time, timedelta
from sqlalchemy import func
from pydantic import BaseModel
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
import os
import shutil
from pathlib import Path
import uuid
from .routers import personnel, detections
from .models.database import DetectionLog, get_db, Personnel, PersonnelImage, FACE_STORAGE_DIR
from .models.schemas import (
    DetectionLogCreate, DetectionLogResponse, 
    PersonnelImageCreate, PersonnelImageResponse, PersonnelWithImages
)
from .routers import personnel
#increamental images
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
# from Face_ai.main import vectorDatabase
import cv2
import numpy as np
from Face_ai.main import client
import time
from typing import Optional




router = APIRouter()
# Include the routers from your modules
router.include_router(personnel.router)  # personnel router already has prefix="/personnel"
router.include_router(detections.router)  # detections router already has prefix="/detections"


# Keep only helper functions if needed elsewhere
def get_face_image_url(request: Request, detection_id: int) -> Optional[str]:
    """Generate URL for face image"""
    if not detection_id:
        return None
    base_url = str(request.base_url).rstrip('/')
    return f"{base_url}/api/v1/detections/{detection_id}/face"

def get_video_url(request: Request, detection_id: int) -> Optional[str]:
    """Generate URL for video clip"""
    if not detection_id:
        return None
    base_url = str(request.base_url).rstrip('/')
    return f"{base_url}/api/v1/detections/{detection_id}/video"


@router.get("/{detection_id}/face")
async def get_detection_face(detection_id: int, db: Session = Depends(get_db)):
    """Serve face image"""
    detection = db.query(DetectionLog).filter(DetectionLog.id == detection_id).first()
    if not detection or not detection.face_image_path:
        raise HTTPException(status_code=404, detail="Face image not found")
        
    if not os.path.exists(detection.face_image_path):
        raise HTTPException(status_code=404, detail="Face image file not found")
    
    return FileResponse(
        detection.face_image_path,
        media_type="image/jpeg",
        filename=os.path.basename(detection.face_image_path)
    )

# @router.get("/{detection_id}/video")
# async def get_detection_video(
#     detection_id: int, 
#     request: Request,
#     db: Session = Depends(get_db)
# ):
#     try:
#         """Serve video file with range support"""
#         detection = db.query(DetectionLog).filter(DetectionLog.id == detection_id).first()
#         if not detection or not detection.video_path:
#             raise HTTPException(status_code=404, detail="Video not found")
        
#         if not os.path.exists(detection.video_path):
#             raise HTTPException(status_code=404, detail="Video file not found")
        
#         file_size = os.path.getsize(detection.video_path)
#         range_header = request.headers.get("range")
        
#         # Get filename for Content-Disposition
#         filename = os.path.basename(detection.video_path)
        
#         # Log for debugging
#         print(f"Serving video: {detection.video_path}, size: {file_size}")
#         print(f"Range header: {range_header}")
        
#         # Common headers for video streaming
#         common_headers = {
#             "Accept-Ranges": "bytes",
#             "Cache-Control": "no-cache, no-store, must-revalidate",
#             "Pragma": "no-cache",
#             "Expires": "0",
#             "Content-Type": "video/mp4",
#             "Access-Control-Allow-Origin": "*",
#             "Access-Control-Allow-Methods": "GET, OPTIONS",
#             "Access-Control-Allow-Headers": "*",
#         }
        
#         if range_header:
#             # Handle range request for video streaming
#             byte_range = range_header.replace("bytes=", "").split("-")
#             start = int(byte_range[0])
#             end = int(byte_range[1]) if byte_range[1] else file_size - 1
#             end = min(end, file_size - 1)
#             content_length = end - start + 1
            
#             def read_range():
#                 with open(detection.video_path, "rb") as video_file:
#                     video_file.seek(start)
#                     yield video_file.read(min(8192, content_length))
            
#             headers = {
#                 **common_headers,
#                 "Content-Range": f"bytes {start}-{end}/{file_size}",
#                 "Content-Length": str(content_length),
#                 "Content-Disposition": f"inline; filename=\"{filename}\"",
#             }
            
#             return StreamingResponse(
#                 read_range(),
#                 status_code=206,
#                 headers=headers
#             )
#         else:
#             def read_file():
#                 with open(detection.video_path, "rb") as video_file:
#                     while chunk := video_file.read(8192):
#                         yield chunk
            
#             headers = {
#                 **common_headers,
#                 "Content-Length": str(file_size),
#                 "Content-Disposition": f"inline; filename=\"{filename}\"",
#             }
            
#             return StreamingResponse(
#                 read_file(),
#                 status_code=200,
#                 headers=headers
#             )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error streaming video: {str(e)}")
    
    

# Existing endpoints
@router.post("/log", response_model=DetectionLogResponse)
def log_detection(
    request: Request,
    person_data: DetectionLogCreate, 
    db: Session = Depends(get_db)
):
    """Log a face detection"""
    db_log = DetectionLog(
        person=person_data.person,
        confidence=person_data.confidence,
        face_image_path=person_data.face_image_path if hasattr(person_data, 'face_image_path') else None,
        detection_time=datetime.now()
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    
    return DetectionLogResponse(
        id=db_log.id,
        person=db_log.person,
        confidence=float(db_log.confidence) if db_log.confidence else None,
        detection_time=db_log.detection_time,
        face_image_url=get_face_image_url(request, db_log.id) if db_log.face_image_path else None,
        video_url=get_video_url(request, db_log.id) if db_log.video_path else None
    )





@router.get("/logs", response_model=List[DetectionLogResponse])
def get_logs(request: Request, db: Session = Depends(get_db)):
    """Get all detection logs with face image and video URLs and personnel names if available"""
    
    # Get all logs
    logs = db.query(DetectionLog).order_by(DetectionLog.detection_time.desc()).all()
    
    # Get all personnel for quick lookup
    all_personnel = db.query(Personnel).all()
    
    # Create a lookup dictionary: national_code -> personnel
    personnel_lookup = {p.national_code: p for p in all_personnel}
    
    result = []
    for log in logs:
        # Initialize variables
        fname = None
        lname = None
        full_name = None
        
        # Check if the person field is a 10-digit number (potential national code)
        person_value = log.person
        if person_value and person_value.isdigit() and len(person_value) == 10:
            # Look up in our personnel dictionary
            personnel = personnel_lookup.get(person_value)
            if personnel:
                fname = personnel.fname
                lname = personnel.lname
                full_name = f"{personnel.fname} {personnel.lname}".strip()
        
        # Create log response with all fields
        log_response = DetectionLogResponse(
            id=log.id,
            person=log.person,
            confidence=float(log.confidence) if log.confidence else None,
            detection_time=log.detection_time,
            face_image_url=get_face_image_url(request, log.id) if log.face_image_path else None,
            video_url=get_video_url(request, log.id) if log.video_path else None,
            fname=fname,
            lname=lname,
            full_name=full_name
        )
        
        result.append(log_response)
    
    return result





@router.get("/{detection_id}/video")
async def get_detection_video(
    detection_id: int, 
    db: Session = Depends(get_db)
):
    """Serve video file - FORCE DOWNLOAD THEN PLAY LOCALLY"""
    
    detection = db.query(DetectionLog).filter(DetectionLog.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    if not detection.video_path:
        raise HTTPException(status_code=404, detail="No video available")
    
    if not os.path.exists(detection.video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    filename = os.path.basename(detection.video_path)
    
    # Force download instead of playback
    return FileResponse(
        path=detection.video_path,
        media_type="video/mp4",
        filename=filename,
        headers={
            "Content-Disposition": f"attachment; filename=\"{filename}\"",  # Forces download
            "Accept-Ranges": "none",
        }
    )
 

# creating personnel with increamental images
@router.post("/personnel/with-images", response_model=PersonnelWithImages, status_code=status.HTTP_201_CREATED)
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
    existing = db.query(Personnel).filter(Personnel.national_code == national_code).first()
    if existing:
        raise HTTPException(status_code=400, detail="کد ملی موجود است!")
    
    # Validate at least one image is provided
    if not images or len(images) == 0:
        raise HTTPException(status_code=400, detail="حد اقل یک تصویر نیاز است!")
    
    # First, create the personnel record
    new_personnel = Personnel(
        fname=fname,
        lname=lname,
        national_code=national_code,
        staff=staff,
        department=department
    )
    
    db.add(new_personnel)
    db.flush()  # This assigns an ID to new_personnel without committing the transaction
    
    # Create personnel images directory
    personnel_images_dir = FACE_STORAGE_DIR / "personnel" / str(new_personnel.id)
    personnel_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate file types and save images
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    saved_images = []  # This will store the actual database objects
    
    for image in images:
        file_extension = os.path.splitext(image.filename)[1].lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"نوع داده غیر مجاز!'{image.filename}'انواع داده مجاز: {', '.join(allowed_extensions)}"
            )
        
        # Read and validate image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail=f"Could not read image: {image.filename}")
        
        # Generate filename for saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        safe_filename = f"{fname}_{timestamp}_{unique_id}{file_extension}"
        safe_filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe_filename)
        
        file_path = personnel_images_dir / safe_filename
        
        # Save image to disk
        cv2.imwrite(str(file_path), img)
        
        # Create image record in database
        db_image = PersonnelImage(
            image_url=str(file_path),
            personnel_id=new_personnel.id
        )
        db.add(db_image)
        db.flush()  # Get the image ID
        
        print(f"🆔 Database record created with ID: {db_image.id}")
        
        # Now call FaceEmbedding with the image ID
        FaceEmbedding(img, national_code, ref_img_id=db_image.id)
        
        # Append the database object, not a dictionary
        saved_images.append(db_image)
        
        await image.seek(0)  # Reset file pointer
    
    # Commit all changes
    db.commit()
    
    # Refresh to get updated relationships
    db.refresh(new_personnel)
    
    print(f"✅ Successfully created personnel {national_code} with {len(saved_images)} images")
    
    # Return the created personnel with images
    return PersonnelWithImages(
        id=new_personnel.id,
        fname=new_personnel.fname,
        lname=new_personnel.lname,
        national_code=new_personnel.national_code,
        staff=new_personnel.staff,
        department=new_personnel.department,
        images=saved_images  # Now passing actual objects, not dictionaries
    )

    
@router.post("/personnel/{personnel_id}/with-images", response_model=PersonnelWithImages, status_code=status.HTTP_201_CREATED)
async def add_images_to_personnel(
    request: Request,
    personnel_id: int,
    images: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Add multiple images to an existing personnel
    """
    # Check if personnel exists
    print('######################hi')

    personnel = db.query(Personnel).filter(Personnel.id == personnel_id).first()
    if not personnel:
        raise HTTPException(status_code=404, detail=f"Personnel with ID {personnel_id} not found")
    
    # Validate at least one image is provided
    if not images or len(images) == 0:
        raise HTTPException(status_code=400, detail="حد اقل یک تصویر نیاز است!")
    
    # Validate file types
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    for image in images:
        file_extension = os.path.splitext(image.filename)[1].lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"نوع داده غیر مجاز! '{image.filename}' انواع داده مجاز: {', '.join(allowed_extensions)}"
            )
        
        # Read and validate image for vector database
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail=f"Could not read image: {image.filename}")
        
        # Add to vector database (will be updated with ref_img_id after DB record creation)
        # We'll store the image data for later use with ref_img_id
        image.img_data = img
        image.contents = contents
        await image.seek(0)
    
    try:
        # Create personnel images directory
        personnel_images_dir = FACE_STORAGE_DIR / "personnel" / str(personnel_id)
        personnel_images_dir.mkdir(parents=True, exist_ok=True)
        
        saved_images = []
        
        # Save each image and create database records
        for image in images:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            file_extension = os.path.splitext(image.filename)[1].lower()
            safe_filename = f"{personnel.fname}_{personnel.lname}_{timestamp}_{unique_id}{file_extension}"
            safe_filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe_filename)
            
            # Save file
            file_path = personnel_images_dir / safe_filename
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            
            # Create database record for image
            db_image = PersonnelImage(
                image_url=str(file_path),
                personnel_id=personnel_id
            )
            
            db.add(db_image)
            db.flush()  # Get ID for this image
            
            # Now add to vector database with ref_img_id
            try:
                # Use the stored image data
                FaceEmbedding(image.img_data, personnel.national_code, ref_img_id=db_image.id)
                print(f"✅ Added to vector DB with ref_img_id: {db_image.id}")
            except Exception as e:
                print(f"⚠️ Warning: Vector DB insertion failed for image {db_image.id}: {e}")
                # Continue even if vector DB fails
            
            saved_images.append(db_image)
        
        # Commit all changes
        db.commit()
        
        # Refresh to get updated relationships
        db.refresh(personnel)
        
        # Generate URLs for all images (including existing ones)
        base_url = str(request.base_url).rstrip('/')
        all_images = []
        
        # Get all images for this personnel (both existing and new)
        all_images_db = db.query(PersonnelImage).filter(
            PersonnelImage.personnel_id == personnel_id
        ).order_by(PersonnelImage.id.desc()).all()
        
        for img in all_images_db:
            all_images.append(
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
            images=all_images
        )
        
    except Exception as e:
        # Rollback database changes
        db.rollback()
        
        # Clean up any saved files for this batch
        if 'personnel_images_dir' in locals() and personnel_images_dir.exists():
            # Only delete the files we just created (optional)
            for image in saved_images:
                try:
                    Path(image.image_url).unlink()
                except:
                    pass
        
        print(f"Error adding images to personnel: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add images: {str(e)}")



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
    delete_faces_from_vector_database(image_id)
    
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
                personnel = db.query(Personnel).filter(Personnel.national_code == national_code).first()
                
                if not personnel:
                    # Create new personnel if it doesn't exist
                    print(f"👤 Personnel not found, creating new record")
                    personnel = Personnel(
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
                        shutil.copy2(str(img_path), file_path)
                        print(f"     💾 Saved to: {file_path}")
                        
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
                        # vectorDatabase(img, national_code, save_mode=True, ref_img_id=db_image.id)
                        
                        FaceEmbedding(img, national_code, ref_img_id=db_image.id)
                        saved_images.append({
                            "id": db_image.id,
                            "file_name": safe_filename,
                            "file_path": str(file_path)
                        })
                        
                        print(f"  ✅ Successfully processed: {img_path.name}")
                        person_processed += 1
                        
                    except Exception as e:
                        print(f"  ❌ Error processing {img_path.name}: {str(e)}")
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


    
    
    







# @router.get("/personnel/{personnel_id}/images/{image_id}/file")
# async def serve_personnel_image(
#     # personnel_id: int,
#     image_id: int,
#     db: Session = Depends(get_db)
# ):
#     """
#     Serve the actual image file
#     """
#     # Get image record
#     image = db.query(PersonnelImage).filter(
#         PersonnelImage.id == image_id,
#         # PersonnelImage.personnel_id == personnel_id
#     ).first()
    
#     if not image:
#         raise HTTPException(status_code=404, detail="Image not found")
    
#     # Check if file exists
#     if not os.path.exists(image.image_url):
#         raise HTTPException(status_code=404, detail="Image file not found on disk")
    
#     # Determine media type based on file extension
#     file_extension = os.path.splitext(image.image_url)[1].lower()
#     media_type = {
#         '.jpg': 'image/jpeg',
#         '.jpeg': 'image/jpeg',
#         '.png': 'image/png',
#         '.gif': 'image/gif',
#         '.bmp': 'image/bmp'
#     }.get(file_extension, 'application/octet-stream')
    
#     return FileResponse(
#         path=image.image_url,
#         media_type=media_type,
#         filename=os.path.basename(image.image_url)
#     )



# @router.get("/personnel/{personnel_id}/images", response_model=List[PersonnelImageResponse])
# async def get_personnel_images(
#     personnel_id: int,
#     request: Request,
#     db: Session = Depends(get_db)
# ):
#     """
#     Get all images for a specific personnel
#     """
#     # Check if personnel exists
#     personnel = db.query(Personnel).filter(Personnel.id == personnel_id).first()
#     if not personnel:
#         raise HTTPException(status_code=404, detail=f"Personnel with ID {personnel_id} not found")
    
#     # Get images
#     images = db.query(PersonnelImage).filter(
#         PersonnelImage.personnel_id == personnel_id
#     ).order_by(PersonnelImage.id.desc()).all()
    
#     # Convert file paths to URLs
#     base_url = str(request.base_url).rstrip('/')
#     result = []
#     for img in images:
#         result.append(
#             PersonnelImageResponse(
#                 id=img.id,
#                 image_url=f"{base_url}/api/v1/personnel/{personnel_id}/images/{img.id}/file",
#                 personnel_id=img.personnel_id,
#                 uploaded_at=img.uploaded_at if hasattr(img, 'uploaded_at') else None
#             )
#         )
    
#     return result




# @router.post("/personnel/from-log/{log_id}", response_model=PersonnelWithImages, status_code=status.HTTP_201_CREATED)
# async def create_personnel_from_log(
#     request: Request,
#     log_id: int,
#     fname: str = Form(...),
#     lname: str = Form(...),
#     national_code: str = Form(...),
#     staff: Optional[bool] = Form(None),
#     department: Optional[str] = Form(None),
#     db: Session = Depends(get_db)
# ):
#     """
#     Create a new personnel using face image from a detection log
#     """
#     # Check if national code already exists
#     existing = db.query(Personnel).filter(Personnel.national_code == national_code).first()
#     if existing:
#         raise HTTPException(status_code=400, detail="کد ملی موجود است!")
    
#     # Get the detection log
#     detection_log = db.query(DetectionLog).filter(DetectionLog.id == log_id).first()
#     if not detection_log:
#         raise HTTPException(status_code=404, detail=f"Detection log with id {log_id} not found")
    
#     # Get the face image path from the detection log
#     if not detection_log.face_image_path:
#         raise HTTPException(status_code=400, detail="No face image found in this detection log")
    
#     # Read and validate the face image
#     face_image_path = Path(detection_log.face_image_path)
#     if not face_image_path.exists():
#         raise HTTPException(status_code=404, detail="Face image file not found")
    
#     # Validate image format
#     allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
#     file_extension = face_image_path.suffix.lower()
#     if file_extension not in allowed_extensions:
#         raise HTTPException(
#             status_code=400, 
#             detail=f"نوع داده غیر مجاز! انواع داده مجاز: {', '.join(allowed_extensions)}"
#         )
    
#     # Create new personnel first
#     db_personnel = Personnel(
#         fname=fname,
#         lname=lname,
#         national_code=national_code,
#         staff=staff,
#         department=department
#     )
    
#     db.add(db_personnel)
#     db.flush()
    
#     try:
#         # Create personnel images directory
#         personnel_images_dir = FACE_STORAGE_DIR / "personnel" / str(db_personnel.id)
#         personnel_images_dir.mkdir(parents=True, exist_ok=True)
        
#         saved_images = []
        
#         # Copy and save the face image from log
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         unique_id = str(uuid.uuid4())[:8]
#         safe_filename = f"{fname}_{lname}_from_log_{timestamp}_{unique_id}{file_extension}"
#         safe_filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe_filename)
        
#         # Save file
#         file_path = personnel_images_dir / safe_filename
#         shutil.copy2(face_image_path, file_path)
        
#         # Create database record for image first (to get the ID)
#         db_image = PersonnelImage(
#             image_url=str(file_path),
#             personnel_id=db_personnel.id
#         )
        
#         db.add(db_image)
#         db.flush()  # This assigns an ID to db_image
        
#         # Now we have the image ID, process for vector database
#         img = cv2.imread(str(face_image_path))
#         if img is None:
#             raise HTTPException(status_code=400, detail=f"Could not read image from log")
        
#         # Create points in vector database with the image ID
#         # vectorDatabase(img, national_code, save_mode=True, ref_img_id=db_image.id)

#         FaceEmbedding(img, national_code, ref_img_id=db_image.id)

#         saved_images.append(db_image)
#         # Commit all changes
#         db.commit()
        
#         # Refresh to get updated relationships
#         db.refresh(db_personnel)
        
#         # Generate URLs for images
#         base_url = str(request.base_url).rstrip('/')
#         images_response = []
        
#         for img in saved_images:
#             images_response.append(
#                 PersonnelImageResponse(
#                     id=img.id,
#                     image_url=f"{base_url}/api/v1/personnel/{db_personnel.id}/images/{img.id}/file",
#                     personnel_id=img.personnel_id,
#                     uploaded_at=img.uploaded_at if hasattr(img, 'uploaded_at') else None
#                 )
#             )
        
#         # Update the detection log to mark it as processed
#         detection_log.personnel_id = db_personnel.id
#         detection_log.is_processed = True
#         db.add(detection_log)
#         db.commit()
        
#         return PersonnelWithImages(
#             id=db_personnel.id,
#             fname=db_personnel.fname,
#             lname=db_personnel.lname,
#             national_code=db_personnel.national_code,
#             staff=db_personnel.staff,
#             department=db_personnel.department,
#             created_at=db_personnel.created_at,
#             images=images_response
#         )
        
#     except Exception as e:
#         # Rollback database changes
#         db.rollback()
        
#         # Clean up any saved files
#         if 'personnel_images_dir' in locals() and personnel_images_dir.exists():
#             shutil.rmtree(personnel_images_dir)
        
#         print(f"Error creating personnel from log: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to create personnel from log: {str(e)}")
    
