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



# @router.delete("/personnel/{personnel_id}", status_code=status.HTTP_204_NO_CONTENT)
# async def delete_personnel(
#     personnel_id: int,
#     db: Session = Depends(get_db)
# ):
#     """
#     Delete a personnel and all associated images (cascade delete)
#     """
#     # Find the personnel
#     personnel = db.query(Personnel).filter(Personnel.id == personnel_id).first()
#     if not personnel:
#         raise HTTPException(status_code=404, detail="Personnel not found")
    
#     # Get information before deletion
#     national_code = personnel.national_code
#     image_ids = [img.id for img in personnel.images]
#     image_paths = [img.image_url for img in personnel.images]
    
#     print(f"\n{'='*50}")
#     print(f"🗑️ Deleting personnel ID: {personnel_id}")
#     print(f"   National code: {national_code}")
#     print(f"   Associated images: {len(image_ids)}")
#     print(f"   Image IDs: {image_ids}")
    
#     try:
#         # STEP 1: Delete from vector database
#         print(f"\n📊 STEP 1: Deleting from vector database...")
        
#         try:
#             from qdrant_client.http import models
            
#             print(f"   Qdrant client: {client}")
#             print(f"   Collection: n12")
            
#             # First, check how many points exist for this national code
#             count_filter = models.Filter(
#                 must=[
#                     models.FieldCondition(
#                         key="person",
#                         match=models.MatchValue(value=national_code)
#                     )
#                 ]
#             )
            
#             count_result = client.count(
#                 collection_name="n12",
#                 count_filter=count_filter
#             )
#             print(f"   Found {count_result.count} points in vector DB for national code: {national_code}")
            
#             if count_result.count > 0:
#                 # Delete the points
#                 delete_result = client.delete(
#                     collection_name="n12",
#                     points_selector=models.FilterSelector(
#                         filter=count_filter
#                     )
#                 )
#                 print(f"   Delete result: {delete_result}")
#                 print(f"✅ Deleted {count_result.count} face embeddings for {national_code}")
#             else:
#                 print(f"   No points found for national code: {national_code}")
            
#             # Also try deleting by individual image IDs
#             for image_id in image_ids:
#                 try:
#                     img_filter = models.Filter(
#                         must=[
#                             models.FieldCondition(
#                                 key="ref_img_id",
#                                 match=models.MatchValue(value=image_id)
#                             )
#                         ]
#                     )
                    
#                     img_count = client.count(
#                         collection_name="n12",
#                         count_filter=img_filter
#                     )
                    
#                     if img_count.count > 0:
#                         img_delete = client.delete(
#                             collection_name="n12",
#                             points_selector=models.FilterSelector(filter=img_filter)
#                         )
#                         print(f"   Deleted {img_count.count} points for ref_img_id: {image_id}")
                    
#                 except Exception as e:
#                     print(f"   ⚠️ Error deleting by image ID {image_id}: {e}")
            
#         except ImportError as e:
#             print(f"⚠️ Import error: {e}")
#         except Exception as e:
#             print(f"⚠️ Vector DB error: {type(e).__name__}: {e}")
#             import traceback
#             traceback.print_exc()
        
#         # STEP 2: Delete from database
#         print(f"\n📊 STEP 2: Deleting from database...")
#         db.delete(personnel)
#         db.commit()
#         print(f"✅ Deleted personnel record from database")
        
#         # STEP 3: Delete physical image files
#         print(f"\n📊 STEP 3: Deleting physical files...")
#         files_deleted = 0
#         for image_path in image_paths:
#             try:
#                 file_path = Path(image_path)
#                 if file_path.exists():
#                     file_path.unlink()
#                     files_deleted += 1
#                     print(f"   ✅ Deleted: {image_path}")
#             except Exception as e:
#                 print(f"   ⚠️ Could not delete {image_path}: {e}")
#         print(f"   Deleted {files_deleted}/{len(image_paths)} files")
        
#         # STEP 4: Delete folder
#         print(f"\n📊 STEP 4: Deleting folder...")
#         personnel_folder = FACE_STORAGE_DIR / "personnel" / str(personnel_id)
#         if personnel_folder.exists():
#             try:
#                 import shutil
#                 shutil.rmtree(personnel_folder)
#                 print(f"✅ Deleted folder: {personnel_folder}")
#             except Exception as e:
#                 print(f"⚠️ Could not delete folder: {e}")
        
#         print(f"\n{'='*50}")
#         print(f"✅ Personnel {personnel_id} deletion complete")
        
#     except Exception as e:
#         db.rollback()
#         print(f"❌ Error: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Failed to delete personnel: {str(e)}")
    
#     return None


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