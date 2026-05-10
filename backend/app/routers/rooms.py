from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from ..models.schemas import RoomCreate, RoomResponse, RoomUpdate
from ..models.db_functions import (
    get_db,
    create_room,
    get_room,
    get_all_rooms,
    update_room,
    delete_room,
    grant_room_access,
    revoke_room_access,
    get_personnel_rooms,
    get_room_personnel,
    check_room_access,
    # log_room_access,
    # get_access_logs,
    # get_access_statistics
)
from ..models.database import Room, Personnel
from enum import Enum


router = APIRouter(prefix="/rooms", tags=["Rooms"])

# ==================== ROOM MANAGEMENT ====================


@router.post("/", response_model=RoomResponse)
def create_new_room(
    room_data: RoomCreate,  # ← Use the schema here
    db: Session = Depends(get_db)
):
    """Create a new room"""
    room = create_room(
        room_number=room_data.room_number,
        room_name=room_data.room_name,
        room_type=room_data.room_type,
        description=room_data.description,
        polygon=room_data.polygon,
        camera_id=room_data.camera_id,
        db=db
    )
    if not room:
        raise HTTPException(status_code=400, detail="ناحیه با این کد موجود است!")
    return room

@router.get("/")
def list_rooms(
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """Get all rooms"""
    rooms = get_all_rooms(active_only=active_only, db=db)
    return rooms

@router.get("/{room_id}", response_model=RoomResponse)
def get_room_by_id(
    room_id: int,
    db: Session = Depends(get_db)
):
    """Get room by ID"""
    room = get_room(room_id=room_id, db=db)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room

@router.put("/{room_id}", response_model=RoomResponse)
def update_room_info(
    room_id: int,
    room_data: RoomUpdate,  # ← Use the schema here
    db: Session = Depends(get_db)
):
    """Update room information"""
    # Filter out None values (only update provided fields)
    update_data = {k: v for k, v in room_data.dict().items() if v is not None}
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    room = update_room(room_id=room_id, db=db, **update_data)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room

@router.delete("/{room_id}")
def remove_room(
    room_id: int,
    hard_delete: bool = False,
    db: Session = Depends(get_db)
):
    """Delete or deactivate a room"""
    success = delete_room(room_id=room_id, hard_delete=hard_delete, db=db)
    if not success:
        raise HTTPException(status_code=404, detail="Room not found")
    return {"message": "Room deleted successfully" if hard_delete else "Room deactivated successfully"}




# ==================== ACCESS MANAGEMENT ====================
@router.post("/{room_id}/grant/{personnel_id}")
def grant_access(
    room_id: int,
    personnel_id: int,
    assigned_by: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Grant a personnel access to a room"""
    success = grant_room_access(
        personnel_id=personnel_id,
        room_id=room_id,
        assigned_by=assigned_by,
        db=db
    )
    if not success:
        raise HTTPException(status_code=400, detail="Failed to grant access. Check if personnel/room exists or access already granted")
    return {"message": "Access granted successfully"}

@router.delete("/{room_id}/revoke/{personnel_id}")
def revoke_access(
    room_id: int,
    personnel_id: int,
    db: Session = Depends(get_db)
):
    """Revoke a personnel's access to a room"""
    success = revoke_room_access(
        personnel_id=personnel_id,
        room_id=room_id,
        db=db
    )
    if not success:
        raise HTTPException(status_code=404, detail="Access not found")
    return {"message": "Access revoked successfully"}

@router.get("/{room_id}/personnel")
def get_room_access_list(
    room_id: int,
    db: Session = Depends(get_db)
):
    """Get all personnel who have access to a room"""
    personnel_list = get_room_personnel(room_id=room_id, db=db)
    return personnel_list

@router.get("/personnel/{personnel_id}/rooms")
def get_personnel_rooms_list(
    personnel_id: int,
    db: Session = Depends(get_db)
):
    """Get all rooms a personnel has access to"""
    rooms = get_personnel_rooms(personnel_id=personnel_id, db=db)
    return rooms

@router.get("/check-access/{personnel_id}/{room_id}")
def check_access(
    personnel_id: int,
    room_id: int,
    db: Session = Depends(get_db)
):
    """Check if a personnel has access to a room"""
    has_access = check_room_access(
        personnel_id=personnel_id,
        room_id=room_id,
        db=db
    )
    return {"has_access": has_access}