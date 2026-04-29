from fastapi import APIRouter, Depends, Request, Query, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, date, time, timedelta
from sqlalchemy import func
from pydantic import BaseModel
from fastapi.responses import FileResponse, StreamingResponse
import os
from .routers import personnel, detections
# from .routers.personnel import delete_faces_from_vector_database

from .models.database import DetectionLog, get_db
from .models.schemas import (
    DetectionLogCreate, DetectionLogResponse, 
    PersonnelImageCreate, PersonnelImageResponse, PersonnelWithImages
)
import uuid  # Add this import

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
    PersonnelImageCreate, PersonnelImageResponse, PersonnelWithImages, PersonnelFromLogsRequest
)
from .routers import personnel
#increamental images
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
# from Detection.TensorRT.infer import vectorDatabase
from Face_ai.main import FaceEmbedding, DeletePointVD, FaceEmbeddingWithoutDetection, FaceCropping
import cv2
import numpy as np


from io import BytesIO
import zipfile
import tempfile




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



@router.get("/detections/{detection_id}/video")
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
            "Content-Disposition": f"attachment; filename=\"{filename}\"",
            "Accept-Ranges": "none",
        }
    )


@router.get("/detections/{detection_id}/face")
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







# @router.get("/debug/check-video/{detection_id}")
# async def check_video_quality(detection_id: int, db: Session = Depends(get_db)):
#     """Check if video is properly formatted for web"""
#     detection = db.query(DetectionLog).filter(DetectionLog.id == detection_id).first()
#     if not detection or not detection.video_path:
#         return {"error": "Video not found"}
    
#     video_path = detection.video_path
#     results = {
#         "path": video_path,
#         "exists": os.path.exists(video_path),
#     }
    
#     if results["exists"]:
#         # Get file size
#         results["file_size"] = os.path.getsize(video_path)
        
#         # Check with OpenCV
#         cap = cv2.VideoCapture(video_path)
#         if cap.isOpened():
#             results["readable"] = True
#             results["frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             results["fps"] = cap.get(cv2.CAP_PROP_FPS)
#             results["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             results["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             results["codec"] = int(cap.get(cv2.CAP_PROP_FOURCC))
#             cap.release()
            
#             # Check if web-optimized (moov atom at start)
#             try:
#                 with open(video_path, 'rb') as f:
#                     header = f.read(100)
#                     results["has_moov_at_start"] = b'moov' in header[:50]
#             except:
#                 pass
#         else:
#             results["readable"] = False
    
#     return results

   

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
    
    # Create new personnel
    db_personnel = Personnel(
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
            FaceEmbeddingWithoutDetection(img, national_code, ref_img_id=db_image.id)
            
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


#add image to an existing personnel
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
                FaceEmbeddingWithoutDetection(image.img_data, personnel.national_code, ref_img_id=db_image.id)

                print(f"imgae data {image.img_data}, national code {personnel.national_code}, ref img id {db_image.id}")
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









@router.get("/personnel/{personnel_id}/images", response_model=List[PersonnelImageResponse])
async def get_personnel_images(
    personnel_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Get all images for a specific personnel
    """
    # Check if personnel exists
    personnel = db.query(Personnel).filter(Personnel.id == personnel_id).first()
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




@router.get("/personnel/{personnel_id}/with-images", response_model=PersonnelWithImages)
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











@router.get("/personnel/{personnel_id}/images/{image_id}/file")
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


@router.post("/personnel/from-log",response_model=List[PersonnelWithImages],
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
    existing = db.query(Personnel).filter(Personnel.national_code == national_code).first()

    # If not exists → create personnel
    if not existing:
        db_personnel = Personnel(
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

        FaceEmbeddingWithoutDetection(img, national_code, ref_img_id=db_image.id)

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

    



@router.get("/detections/{detection_id}/extract-frames")
async def extract_frames_from_detection_video(
    detection_id: int,
    frame_interval: int = Query(30, description="Extract every Nth frame"),
    max_frames: Optional[int] = Query(None, description="Maximum number of frames to extract"),
    db: Session = Depends(get_db)
):
    """
    Extract frames from video and return them in memory (without saving to disk)
    """
    
    # Get detection log from database
    detection = db.query(DetectionLog).filter(DetectionLog.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    if not detection.video_path:
        raise HTTPException(status_code=404, detail="No video available for this detection")
    
    video_path = Path(detection.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found at: {video_path}")
    
    try:
        # Open video using OpenCV
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        frame_count = 0
        extracted_frames = []
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame based on interval
            if frame_count % frame_interval == 0:
                # Encode frame to JPEG in memory
                success, encoded_image = cv2.imencode('.jpg', frame)
                if success:
                    # Convert to base64 for JSON response
                    import base64
                    frame_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
                    
                    extracted_frames.append({
                        "frame_number": frame_count,
                        "frame_index": extracted_count,
                        "image_base64": frame_base64,
                        "image_size_kb": len(encoded_image.tobytes()) / 1024
                    })
                    extracted_count += 1
                
                # Stop if max frames reached
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        
        return {
            "success": True,
            "detection_id": detection_id,
            "video_path": str(video_path),
            "total_frames_in_video": frame_count,
            "frames_extracted": extracted_count,
            "frame_interval": frame_interval,
            "frames": extracted_frames
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting frames: {str(e)}")




