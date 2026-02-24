from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from ..models.database import Personnel, get_db
from ..models.schemas import Personnel as PersonnelSchema, PersonnelCreate, PersonnelUpdate

router = APIRouter(prefix="/personnel", tags=["personnel"])

@router.post("/", response_model=PersonnelSchema, status_code=status.HTTP_201_CREATED)
def create_personnel(personnel: PersonnelCreate, db: Session = Depends(get_db)):
    # Check if national code already exists
    existing = db.query(Personnel).filter(Personnel.national_code == personnel.national_code).first()
    if existing:
        raise HTTPException(status_code=400, detail="National code already exists")
    
    db_personnel = Personnel(**personnel.dict())
    db.add(db_personnel)
    db.commit()
    db.refresh(db_personnel)
    return db_personnel

@router.get("/", response_model=List[PersonnelSchema])
def read_personnel(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    personnel = db.query(Personnel).offset(skip).limit(limit).all()
    return personnel

@router.get("/{personnel_id}", response_model=PersonnelSchema)
def read_personnel(personnel_id: int, db: Session = Depends(get_db)):
    db_personnel = db.query(Personnel).filter(Personnel.id == personnel_id).first()
    if db_personnel is None:
        raise HTTPException(status_code=404, detail="Personnel not found")
    return db_personnel

@router.get("/national-code/{national_code}", response_model=PersonnelSchema)
def read_personnel_by_national(national_code: str, db: Session = Depends(get_db)):
    db_personnel = db.query(Personnel).filter(Personnel.national_code == national_code).first()
    if db_personnel is None:
        raise HTTPException(status_code=404, detail="Personnel not found")
    return db_personnel

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
    
    db.delete(db_personnel)
    db.commit()
    return None