from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

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


router = APIRouter(prefix="/rooms", tags=["Rooms"])

# ==================== ROOM MANAGEMENT ====================
@router.post("/")
def create_new_room(
    room_number: str,
    room_name: Optional[str] = None,
    room_type: Optional[str] = None,
    capacity: Optional[int] = None,
    description: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Create a new room"""
    room = create_room(
        room_number=room_number,
        room_name=room_name,
        room_type=room_type,
        capacity=capacity,
        description=description,
        db=db
    )
    if not room:
        raise HTTPException(status_code=400, detail="Room already exists or invalid data")
    return room


@router.get("/")
def list_rooms(
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """Get all rooms"""
    rooms = get_all_rooms(active_only=active_only, db=db)
    return rooms


@router.get("/{room_id}")
def get_room_by_id(
    room_id: int,
    db: Session = Depends(get_db)
):
    """Get room by ID"""
    room = get_room(room_id=room_id, db=db)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room


@router.put("/{room_id}")
def update_room_info(
    room_id: int,
    room_number: Optional[str] = None,
    room_name: Optional[str] = None,
    room_type: Optional[str] = None,
    capacity: Optional[int] = None,
    is_active: Optional[bool] = None,
    description: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Update room information"""
    update_data = {}
    if room_number is not None:
        update_data['room_number'] = room_number
    if room_name is not None:
        update_data['room_name'] = room_name
    if room_type is not None:
        update_data['room_type'] = room_type
    if capacity is not None:
        update_data['capacity'] = capacity
    if is_active is not None:
        update_data['is_active'] = is_active
    if description is not None:
        update_data['description'] = description
    
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


# ==================== ACCESS LOGS ====================
# @router.post("/{room_id}/access-log")
# def create_access_log(
#     room_id: int,
#     personnel_id: int,
#     access_granted: bool,
#     face_confidence: Optional[float] = None,
#     denial_reason: Optional[str] = None,
#     camera_id: Optional[int] = None,
#     db: Session = Depends(get_db)
# ):
#     """Log an access attempt to a room"""
#     access_log = log_room_access(
#         personnel_id=personnel_id,
#         room_id=room_id,
#         access_granted=access_granted,
#         face_confidence=face_confidence,
#         denial_reason=denial_reason,
#         camera_id=camera_id,
#         db=db
#     )
#     if not access_log:
#         raise HTTPException(status_code=400, detail="Failed to log access")
#     return access_log


# @router.get("/{room_id}/access-logs")
# def get_room_access_logs(
#     room_id: int,
#     personnel_id: Optional[int] = None,
#     start_date: Optional[datetime] = None,
#     end_date: Optional[datetime] = None,
#     access_granted: Optional[bool] = None,
#     limit: int = Query(100, ge=1, le=1000),
#     db: Session = Depends(get_db)
# ):
#     """Get access logs for a room"""
#     logs = get_access_logs(
#         room_id=room_id,
#         personnel_id=personnel_id,
#         start_date=start_date,
#         end_date=end_date,
#         access_granted=access_granted,
#         limit=limit,
#         db=db
#     )
#     return logs


# @router.get("/{room_id}/statistics")
# def get_room_statistics(
#     room_id: int,
#     days: int = Query(30, ge=1, le=365),
#     db: Session = Depends(get_db)
# ):
#     """Get access statistics for a room"""
#     stats = get_access_statistics(room_id=room_id, days=days, db=db)
#     return stats


# @router.get("/access-logs/personnel/{personnel_id}")
# def get_personnel_access_logs(
#     personnel_id: int,
#     start_date: Optional[datetime] = None,
#     end_date: Optional[datetime] = None,
#     access_granted: Optional[bool] = None,
#     limit: int = Query(100, ge=1, le=1000),
#     db: Session = Depends(get_db)
# ):
#     """Get access logs for a specific personnel"""
#     logs = get_access_logs(
#         personnel_id=personnel_id,
#         start_date=start_date,
#         end_date=end_date,
#         access_granted=access_granted,
#         limit=limit,
#         db=db
#     )
#     return logs