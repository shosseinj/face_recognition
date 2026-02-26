from fastapi import APIRouter, Depends, HTTPException, status
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
from fastapi import APIRouter, Depends, Request, Query, HTTPException, UploadFile, File, Form

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
def update_personnel(personnel_id: int, personnel_update: PersonnelUpdate, db: Session = Depends(get_db)):
    db_personnel = db.query(Personnel).filter(Personnel.id == personnel_id).first()
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