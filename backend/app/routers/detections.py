from fastapi import APIRouter, Depends, Request, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
from fastapi.responses import FileResponse, StreamingResponse
import os
from ..models.database import DetectionLog, Personnel
from ..models.db_functions import get_db
from ..models.schemas import DetectionLogResponse, DetectionLogCreate
from fastapi import Query
from ..utils import get_face_image_url, get_video_url
# Changed prefix from "/detections" to "/logs"
router = APIRouter(prefix="/logs", tags=["logs"])
from typing import Optional
# def get_face_image_url(request: Request, detection_id: int) -> str | None:
#     """Generate URL for face image"""
#     if not detection_id:
#         return None
#     base_url = str(request.base_url).rstrip('/')
#     return f"{base_url}/api/v1/logs/{detection_id}/face"

# def get_video_url(request: Request, detection_id: int) -> str | None:
#     """Generate URL for video clip"""
#     if not detection_id:
#         return None
#     base_url = str(request.base_url).rstrip('/')
#     return f"{base_url}/api/v1/logs/{detection_id}/video"

from typing import Optional





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





@router.get("/", response_model=List[DetectionLogResponse])
def get_logs(
    request: Request, 
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    db: Session = Depends(get_db),

    ):
    """Get all detection logs with face image and video URLs and personnel names if available"""
    
    # Get all logs
    logs = db.query(DetectionLog)\
        .order_by(DetectionLog.detection_time.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
    
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


@router.get("/illegal-access")
def get_illegal_access_logs(
    request : Request,
    date: Optional[datetime] = Query(None, description="Filter by specific date (YYYY-MM-DD)"),
    from_date: Optional[datetime] = Query(None, description="Start date range"),
    to_date: Optional[datetime] = Query(None, description="End date range"),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """
    Get detection logs where access was denied (illegal entrance attempts)
    Can filter by:
    - Specific date
    - Date range (from_date to to_date)
    """
    
    # Build query for denied access only
    query = db.query(DetectionLog).filter(
        DetectionLog.access_granted == False
    )
    
    # Apply date filters
    if date:
        # Filter by specific date
        start_of_day = datetime.combine(date.date(), datetime.min.time())
        end_of_day = datetime.combine(date.date(), datetime.max.time())
        query = query.filter(
            DetectionLog.detection_time >= start_of_day,
            DetectionLog.detection_time <= end_of_day
        )
    elif from_date and to_date:
        # Filter by date range
        query = query.filter(
            DetectionLog.detection_time >= from_date,
            DetectionLog.detection_time <= to_date
        )
    elif from_date:
        # Filter from date to now
        query = query.filter(DetectionLog.detection_time >= from_date)
    elif to_date:
        # Filter up to date
        query = query.filter(DetectionLog.detection_time <= to_date)
    
    # Order by most recent first
    query = query.order_by(DetectionLog.detection_time.desc())
    
    # Get total count before pagination
    total_count = query.count()
    
    # Apply limit
    logs = query.limit(limit).all()
    
    # Get personnel for name lookup
    all_personnel = db.query(Personnel).all()
    personnel_lookup = {p.national_code: p for p in all_personnel}
    
    # Prepare response
    result = []
    for log in logs:
        personnel = personnel_lookup.get(log.person)
        
        result.append({
            "id": log.id,
            "person": {
                "national_code": log.person,
                "name": f"{personnel.fname} {personnel.lname}".strip() if personnel else "Unknown",
                "fname": personnel.fname if personnel else None,
                "lname": personnel.lname if personnel else None
            } if personnel else {
                "national_code": log.person,
                "name": log.person,
                "fname": None,
                "lname": None
            },
            "access_granted": log.access_granted,  # ← ADD THIS FIELD
            "area": log.area,
            "confidence": log.confidence,
            "detection_time": log.detection_time,
            "room_id": log.room_id,
            "room": log.room.room_name if log.room else None,
            "face_image_url": get_face_image_url(request, log.id) if log.face_image_path else None,
            "video_url": get_video_url(request, log.id) if log.video_path else None
        })
    
    return {
        "total_illegal_attempts": total_count,
        "date_filter": {
            "date": date.isoformat() if date else None,
            "from_date": from_date.isoformat() if from_date else None,
            "to_date": to_date.isoformat() if to_date else None
        },
        "logs": result
    }




@router.get("/reports/daily")
def get_daily_report(
    date: datetime = Query(..., description="Date for report (YYYY-MM-DD)"),
    db: Session = Depends(get_db)
):
    """
    Daily report: list of detected personnel with:
    - number of detections
    - first seen time
    - last seen time  
    - total presence (minutes/hours)
    """
    from sqlalchemy import func
    
    # Set date range
    start_of_day = datetime.combine(date.date(), datetime.min.time())
    end_of_day = datetime.combine(date.date(), datetime.max.time())
    
    # Get all personnel for name lookup
    all_personnel = db.query(Personnel).all()
    personnel_lookup = {p.national_code: p for p in all_personnel}
    
    # Get detections grouped by person
    results = db.query(
        DetectionLog.person,
        func.count(DetectionLog.id).label('detection_count'),
        func.min(DetectionLog.detection_time).label('first_seen'),
        func.max(DetectionLog.detection_time).label('last_seen')
    ).filter(
        DetectionLog.detection_time >= start_of_day,
        DetectionLog.detection_time <= end_of_day
    ).group_by(DetectionLog.person).all()
    
    # Prepare response
    report_data = []
    
    for result in results:
        # Get personnel info
        personnel = personnel_lookup.get(result.person)
        
        # Calculate presence duration
        presence_seconds = (result.last_seen - result.first_seen).total_seconds()
        presence_minutes = round(presence_seconds / 60, 1)
        presence_hours = round(presence_seconds / 3600, 1)
        
        # Format duration
        if presence_hours >= 1:
            duration_str = f"{int(presence_hours)}h {int(presence_minutes % 60)}m"
        else:
            duration_str = f"{presence_minutes}m"
        
        report_data.append({
            "person": {
                "national_code": result.person,
                "name": f"{personnel.fname} {personnel.lname}".strip() if personnel else "Unknown",
                "department": personnel.department if personnel else None
            },
            "detection_count": result.detection_count,
            "first_seen": result.first_seen,
            "last_seen": result.last_seen,
            "presence_duration": duration_str
        })
    
    # Sort by detection count
    report_data.sort(key=lambda x: x['detection_count'], reverse=True)
    
    return {
        "date": date.date().isoformat(),
        "summary": {
            "total_personnel": len(report_data),
            "total_detections": sum(item['detection_count'] for item in report_data)
        },
        "personnel": report_data
    }



@router.get("/{log_id}/video")
async def get_detection_video(
    detection_id: int, 
    request: Request,
    db: Session = Depends(get_db)
):
    """Serve video file with range support"""
    detection = db.query(DetectionLog).filter(DetectionLog.id == detection_id).first()
    if not detection or not detection.video_path:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if not os.path.exists(detection.video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    file_size = os.path.getsize(detection.video_path)
    range_header = request.headers.get("range")
    
    # Get filename for Content-Disposition
    filename = os.path.basename(detection.video_path)
    
    # Log for debugging
    print(f"Serving video: {detection.video_path}, size: {file_size}")
    print(f"Range header: {range_header}")
    
    # Common headers for video streaming
    common_headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "Content-Type": "video/mp4",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    }
    
    if range_header:
        # Handle range request for video streaming
        byte_range = range_header.replace("bytes=", "").split("-")
        start = int(byte_range[0])
        end = int(byte_range[1]) if byte_range[1] else file_size - 1
        end = min(end, file_size - 1)
        content_length = end - start + 1
        
        def read_range():
            with open(detection.video_path, "rb") as video_file:
                video_file.seek(start)
                yield video_file.read(min(8192, content_length))
        
        headers = {
            **common_headers,
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(content_length),
            "Content-Disposition": f"inline; filename=\"{filename}\"",
        }
        
        return StreamingResponse(
            read_range(),
            status_code=206,
            headers=headers
        )
    else:
        def read_file():
            with open(detection.video_path, "rb") as video_file:
                while chunk := video_file.read(8192):
                    yield chunk
        
        headers = {
            **common_headers,
            "Content-Length": str(file_size),
            "Content-Disposition": f"inline; filename=\"{filename}\"",
        }
        
        return StreamingResponse(
            read_file(),
            status_code=200,
            headers=headers
        )
    




@router.get("{log_id}/face")
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






###########


@router.get("/filter")
def get_logs_by_date_range(
    request: Request,
    from_date: datetime = Query(..., description="Start date (YYYY-MM-DDTHH:MM:SS)"),
    to_date: datetime = Query(..., description="End date (YYYY-MM-DDTHH:MM:SS)"),
    personnel_national_code: str = Query(None, description="Optional: Filter by personnel national code"),
    db: Session = Depends(get_db)
):
    """
    Get detection logs filtered by date range (from - to)
    Optionally filter by specific personnel using national code
    """
    
    # Validate date range
    if from_date > to_date:
        raise HTTPException(
            status_code=400, 
            detail="from_date must be less than or equal to to_date"
        )
    
    # Build query
    query = db.query(DetectionLog).filter(
        DetectionLog.detection_time >= from_date,
        DetectionLog.detection_time <= to_date
    )
    
    # Apply personnel filter if provided
    if personnel_national_code:
        query = query.filter(DetectionLog.person == personnel_national_code)
    
    # Get logs ordered by detection time
    logs = query.order_by(DetectionLog.detection_time.desc()).all()
    
    # Get personnel lookup for names
    all_personnel = db.query(Personnel).all()
    personnel_lookup = {p.national_code: p for p in all_personnel}
    
    # Prepare response
    result = []
    for log in logs:
        fname = None
        lname = None
        full_name = None
        
        # Look up personnel if this is a national code
        if log.person and log.person.isdigit() and len(log.person) == 10:
            personnel = personnel_lookup.get(log.person)
            if personnel:
                fname = personnel.fname
                lname = personnel.lname
                full_name = f"{personnel.fname} {personnel.lname}".strip()
        
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


@router.get("/filter/summary")
def get_logs_summary_by_date_range(
    from_date: datetime = Query(..., description="Start date (YYYY-MM-DDTHH:MM:SS)"),
    to_date: datetime = Query(..., description="End date (YYYY-MM-DDTHH:MM:SS)"),
    personnel_national_code: str = Query(None, description="Optional: Filter by personnel national code"),
    db: Session = Depends(get_db)
):
    """
    Get summary statistics for detection logs within a date range
    Returns counts, unique personnel, and activity metrics
    """
    from sqlalchemy import func
    
    # Validate date range
    if from_date > to_date:
        raise HTTPException(
            status_code=400, 
            detail="from_date must be less than or equal to to_date"
        )
    
    # Build query
    query = db.query(DetectionLog).filter(
        DetectionLog.detection_time >= from_date,
        DetectionLog.detection_time <= to_date
    )
    
    # Apply personnel filter if provided
    if personnel_national_code:
        query = query.filter(DetectionLog.person == personnel_national_code)
    
    # Get overall statistics
    stats = query.with_entities(
        func.count(DetectionLog.id).label('total_detections'),
        func.avg(DetectionLog.confidence).label('avg_confidence'),
        func.min(DetectionLog.detection_time).label('first_detection'),
        func.max(DetectionLog.detection_time).label('last_detection')
    ).first()
    
    # Get unique personnel count
    unique_personnel = query.with_entities(DetectionLog.person).distinct().count()
    
    # Get detection counts by personnel (top 10)
    personnel_counts = query.with_entities(
        DetectionLog.person,
        func.count(DetectionLog.id).label('detection_count')
    ).group_by(DetectionLog.person).order_by(func.count(DetectionLog.id).desc()).limit(10).all()
    
    # Get personnel names for the top list
    all_personnel = db.query(Personnel).all()
    personnel_lookup = {p.national_code: p for p in all_personnel}
    
    top_personnel_list = []
    for person_code, count in personnel_counts:
        personnel = personnel_lookup.get(person_code)
        top_personnel_list.append({
            "national_code": person_code,
            "full_name": f"{personnel.fname} {personnel.lname}".strip() if personnel else None,
            "detection_count": count
        })
    
    # Get daily breakdown
    daily_counts = query.with_entities(
        func.date(DetectionLog.detection_time).label('date'),
        func.count(DetectionLog.id).label('count')
    ).group_by(func.date(DetectionLog.detection_time)).order_by(func.date(DetectionLog.detection_time)).all()
    
    # Get hourly breakdown
    hourly_counts = query.with_entities(
        func.strftime('%H', DetectionLog.detection_time).label('hour'),
        func.count(DetectionLog.id).label('count')
    ).group_by(func.strftime('%H', DetectionLog.detection_time)).order_by('hour').all()
    
    return {
        "date_range": {
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "days": (to_date - from_date).days
        },
        "overall_stats": {
            "total_detections": stats.total_detections or 0,
            "unique_personnel": unique_personnel,
            "average_confidence": round(stats.avg_confidence, 2) if stats.avg_confidence else 0,
            "first_detection": stats.first_detection,
            "last_detection": stats.last_detection
        },
        "top_personnel": top_personnel_list,
        "daily_breakdown": [
            {"date": str(day.date), "detections": day.count}
            for day in daily_counts
        ],
        "hourly_breakdown": [
            {"hour": int(hour.hour), "detections": hour.count}
            for hour in hourly_counts
        ]
    }


@router.get("/filter/export")
async def export_logs_by_date_range(
    from_date: datetime = Query(..., description="Start date (YYYY-MM-DDTHH:MM:SS)"),
    to_date: datetime = Query(..., description="End date (YYYY-MM-DDTHH:MM:SS)"),
    personnel_national_code: str = Query(None, description="Optional: Filter by personnel national code"),
    format: str = Query("json", description="Export format: json or csv"),
    db: Session = Depends(get_db)
):
    """
    Export detection logs within a date range to JSON or CSV format
    """
    import json
    import csv
    from io import StringIO
    
    # Validate date range
    if from_date > to_date:
        raise HTTPException(
            status_code=400, 
            detail="from_date must be less than or equal to to_date"
        )
    
    # Build query
    query = db.query(DetectionLog).filter(
        DetectionLog.detection_time >= from_date,
        DetectionLog.detection_time <= to_date
    ).order_by(DetectionLog.detection_time.desc())
    
    # Apply personnel filter if provided
    if personnel_national_code:
        query = query.filter(DetectionLog.person == personnel_national_code)
    
    logs = query.all()
    
    # Get personnel lookup
    all_personnel = db.query(Personnel).all()
    personnel_lookup = {p.national_code: p for p in all_personnel}
    
    # Prepare export data
    export_data = []
    for log in logs:
        personnel = personnel_lookup.get(log.person) if log.person else None
        
        export_data.append({
            "id": log.id,
            "person": log.person,
            "person_name": f"{personnel.fname} {personnel.lname}".strip() if personnel else "Unknown",
            "confidence": float(log.confidence) if log.confidence else None,
            "detection_time": log.detection_time.isoformat() if log.detection_time else None,
            "face_image_path": log.face_image_path,
            "video_path": log.video_path
        })
    
    # Export based on format
    if format.lower() == "csv":
        # Create CSV
        output = StringIO()
        if export_data:
            writer = csv.DictWriter(output, fieldnames=export_data[0].keys())
            writer.writeheader()
            writer.writerows(export_data)
        
        csv_content = output.getvalue()
        output.close()
        
        filename = f"detection_logs_{from_date.strftime('%Y%m%d')}_to_{to_date.strftime('%Y%m%d')}.csv"
        
        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    
    else:  # JSON format (default)
        filename = f"detection_logs_{from_date.strftime('%Y%m%d')}_to_{to_date.strftime('%Y%m%d')}.json"
        
        return StreamingResponse(
            iter([json.dumps(export_data, indent=2, ensure_ascii=False)]),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        

@router.get("/by-room/{room_id}")
def get_detections_by_room(
    room_id: int,
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Get detection logs for a specific room"""
    detections = db.query(DetectionLog)\
        .filter(DetectionLog.room_id == room_id)\
        .order_by(DetectionLog.detection_time.desc())\
        .limit(limit)\
        .all()
    return detections
        
        
# from fastapi import APIRouter, Depends, Request, HTTPException
# from sqlalchemy.orm import Session
# from typing import List
# from datetime import datetime
# from fastapi.responses import FileResponse, StreamingResponse
# import os
# from ..models.database import DetectionLog, Personnel, get_db
# from ..models.schemas import DetectionLogResponse

# from fastapi import Query



# router = APIRouter(prefix="/detections", tags=["detections"])

# def get_face_image_url(request: Request, detection_id: int) -> str | None:
#     """Generate URL for face image"""
#     if not detection_id:
#         return None
#     base_url = str(request.base_url).rstrip('/')
#     return f"{base_url}/api/v1/detections/{detection_id}/face"

# def get_video_url(request: Request, detection_id: int) -> str | None:
#     """Generate URL for video clip"""
#     if not detection_id:
#         return None
#     base_url = str(request.base_url).rstrip('/')
#     return f"{base_url}/api/v1/detections/{detection_id}/video"



# @router.get("/{detection_id}/video")
# async def get_detection_video(
#     detection_id: int, 
#     request: Request,
#     db: Session = Depends(get_db)
# ):
#     """Serve video file with range support"""
#     detection = db.query(DetectionLog).filter(DetectionLog.id == detection_id).first()
#     if not detection or not detection.video_path:
#         raise HTTPException(status_code=404, detail="Video not found")
    
#     if not os.path.exists(detection.video_path):
#         raise HTTPException(status_code=404, detail="Video file not found")
    
#     file_size = os.path.getsize(detection.video_path)
#     range_header = request.headers.get("range")
    
#     # Get filename for Content-Disposition
#     filename = os.path.basename(detection.video_path)
    
#     # Log for debugging
#     print(f"Serving video: {detection.video_path}, size: {file_size}")
#     print(f"Range header: {range_header}")
    
#     # Common headers for video streaming
#     common_headers = {
#         "Accept-Ranges": "bytes",
#         "Cache-Control": "no-cache, no-store, must-revalidate",
#         "Pragma": "no-cache",
#         "Expires": "0",
#         "Content-Type": "video/mp4",
#         "Access-Control-Allow-Origin": "*",
#         "Access-Control-Allow-Methods": "GET, OPTIONS",
#         "Access-Control-Allow-Headers": "*",
#     }
    
#     if range_header:
#         # Handle range request for video streaming
#         byte_range = range_header.replace("bytes=", "").split("-")
#         start = int(byte_range[0])
#         end = int(byte_range[1]) if byte_range[1] else file_size - 1
#         end = min(end, file_size - 1)
#         content_length = end - start + 1
        
#         def read_range():
#             with open(detection.video_path, "rb") as video_file:
#                 video_file.seek(start)
#                 yield video_file.read(min(8192, content_length))
        
#         headers = {
#             **common_headers,
#             "Content-Range": f"bytes {start}-{end}/{file_size}",
#             "Content-Length": str(content_length),
#             "Content-Disposition": f"inline; filename=\"{filename}\"",
#         }
        
#         return StreamingResponse(
#             read_range(),
#             status_code=206,
#             headers=headers
#         )
#     else:
#         def read_file():
#             with open(detection.video_path, "rb") as video_file:
#                 while chunk := video_file.read(8192):
#                     yield chunk
        
#         headers = {
#             **common_headers,
#             "Content-Length": str(file_size),
#             "Content-Disposition": f"inline; filename=\"{filename}\"",
#         }
        
#         return StreamingResponse(
#             read_file(),
#             status_code=200,
#             headers=headers
#         )
    



# ###########

# @router.get("/logs", response_model=List[DetectionLogResponse])
# def get_logs(request: Request, db: Session = Depends(get_db)):
#     """Get all detection logs with face image and video URLs and personnel names if available"""
    
#     # Get all logs
#     logs = db.query(DetectionLog).order_by(DetectionLog.detection_time.desc()).all()
    
#     # Get all personnel for quick lookup
#     all_personnel = db.query(Personnel).all()
    
#     # Create a lookup dictionary: national_code -> personnel
#     personnel_lookup = {p.national_code: p for p in all_personnel}
    
#     result = []
#     for log in logs:
#         # Initialize variables
#         fname = None
#         lname = None
#         full_name = None
        
#         # Check if the person field is a 10-digit number (potential national code)
#         person_value = log.person
#         if person_value and person_value.isdigit() and len(person_value) == 10:
#             # Look up in our personnel dictionary
#             personnel = personnel_lookup.get(person_value)
#             if personnel:
#                 fname = personnel.fname
#                 lname = personnel.lname
#                 full_name = f"{personnel.fname} {personnel.lname}".strip()
        
#         # Create log response with all fields
#         log_response = DetectionLogResponse(
#             id=log.id,
#             person=log.person,
#             confidence=float(log.confidence) if log.confidence else None,
#             detection_time=log.detection_time,
#             face_image_url=get_face_image_url(request, log.id) if log.face_image_path else None,
#             video_url=get_video_url(request, log.id) if log.video_path else None,
#             fname=fname,
#             lname=lname,
#             full_name=full_name
#         )
        
#         result.append(log_response)
    
#     return result





# @router.get("/logs/filter")
# def get_logs_by_date_range(
#     request: Request,
#     from_date: datetime = Query(..., description="Start date (YYYY-MM-DDTHH:MM:SS)"),
#     to_date: datetime = Query(..., description="End date (YYYY-MM-DDTHH:MM:SS)"),
#     personnel_national_code: str = Query(None, description="Optional: Filter by personnel national code"),
#     db: Session = Depends(get_db)
# ):
#     """
#     Get detection logs filtered by date range (from - to)
#     Optionally filter by specific personnel using national code
#     """
    
#     # Validate date range
#     if from_date > to_date:
#         raise HTTPException(
#             status_code=400, 
#             detail="from_date must be less than or equal to to_date"
#         )
    
#     # Build query
#     query = db.query(DetectionLog).filter(
#         DetectionLog.detection_time >= from_date,
#         DetectionLog.detection_time <= to_date
#     )
    
#     # Apply personnel filter if provided
#     if personnel_national_code:
#         query = query.filter(DetectionLog.person == personnel_national_code)
    
#     # Get logs ordered by detection time
#     logs = query.order_by(DetectionLog.detection_time.desc()).all()
    
#     # Get personnel lookup for names
#     from ..models.database import Personnel
#     all_personnel = db.query(Personnel).all()
#     personnel_lookup = {p.national_code: p for p in all_personnel}
    
#     # Prepare response
#     result = []
#     for log in logs:
#         fname = None
#         lname = None
#         full_name = None
        
#         # Look up personnel if this is a national code
#         if log.person and log.person.isdigit() and len(log.person) == 10:
#             personnel = personnel_lookup.get(log.person)
#             if personnel:
#                 fname = personnel.fname
#                 lname = personnel.lname
#                 full_name = f"{personnel.fname} {personnel.lname}".strip()
        
#         log_response = DetectionLogResponse(
#             id=log.id,
#             person=log.person,
#             confidence=float(log.confidence) if log.confidence else None,
#             detection_time=log.detection_time,
#             face_image_url=get_face_image_url(request, log.id) if log.face_image_path else None,
#             video_url=get_video_url(request, log.id) if log.video_path else None,
#             fname=fname,
#             lname=lname,
#             full_name=full_name
#         )
#         result.append(log_response)
    
#     return result


# @router.get("/logs/filter/summary")
# def get_logs_summary_by_date_range(
#     from_date: datetime = Query(..., description="Start date (YYYY-MM-DDTHH:MM:SS)"),
#     to_date: datetime = Query(..., description="End date (YYYY-MM-DDTHH:MM:SS)"),
#     personnel_national_code: str = Query(None, description="Optional: Filter by personnel national code"),
#     db: Session = Depends(get_db)
# ):
#     """
#     Get summary statistics for detection logs within a date range
#     Returns counts, unique personnel, and activity metrics
#     """
#     from sqlalchemy import func
#     from ..models.database import Personnel
    
#     # Validate date range
#     if from_date > to_date:
#         raise HTTPException(
#             status_code=400, 
#             detail="from_date must be less than or equal to to_date"
#         )
    
#     # Build query
#     query = db.query(DetectionLog).filter(
#         DetectionLog.detection_time >= from_date,
#         DetectionLog.detection_time <= to_date
#     )
    
#     # Apply personnel filter if provided
#     if personnel_national_code:
#         query = query.filter(DetectionLog.person == personnel_national_code)
    
#     # Get overall statistics
#     stats = query.with_entities(
#         func.count(DetectionLog.id).label('total_detections'),
#         func.avg(DetectionLog.confidence).label('avg_confidence'),
#         func.min(DetectionLog.detection_time).label('first_detection'),
#         func.max(DetectionLog.detection_time).label('last_detection')
#     ).first()
    
#     # Get unique personnel count
#     unique_personnel = query.with_entities(DetectionLog.person).distinct().count()
    
#     # Get detection counts by personnel (top 10)
#     personnel_counts = query.with_entities(
#         DetectionLog.person,
#         func.count(DetectionLog.id).label('detection_count')
#     ).group_by(DetectionLog.person).order_by(func.count(DetectionLog.id).desc()).limit(10).all()
    
#     # Get personnel names for the top list
#     all_personnel = db.query(Personnel).all()
#     personnel_lookup = {p.national_code: p for p in all_personnel}
    
#     top_personnel_list = []
#     for person_code, count in personnel_counts:
#         personnel = personnel_lookup.get(person_code)
#         top_personnel_list.append({
#             "national_code": person_code,
#             "full_name": f"{personnel.fname} {personnel.lname}".strip() if personnel else None,
#             "detection_count": count
#         })
    
#     # Get daily breakdown
#     daily_counts = query.with_entities(
#         func.date(DetectionLog.detection_time).label('date'),
#         func.count(DetectionLog.id).label('count')
#     ).group_by(func.date(DetectionLog.detection_time)).order_by(func.date(DetectionLog.detection_time)).all()
    
#     # Get hourly breakdown
#     hourly_counts = query.with_entities(
#         func.strftime('%H', DetectionLog.detection_time).label('hour'),
#         func.count(DetectionLog.id).label('count')
#     ).group_by(func.strftime('%H', DetectionLog.detection_time)).order_by('hour').all()
    
#     return {
#         "date_range": {
#             "from": from_date.isoformat(),
#             "to": to_date.isoformat(),
#             "days": (to_date - from_date).days
#         },
#         "overall_stats": {
#             "total_detections": stats.total_detections or 0,
#             "unique_personnel": unique_personnel,
#             "average_confidence": round(stats.avg_confidence, 2) if stats.avg_confidence else 0,
#             "first_detection": stats.first_detection,
#             "last_detection": stats.last_detection
#         },
#         "top_personnel": top_personnel_list,
#         "daily_breakdown": [
#             {"date": str(day.date), "detections": day.count}
#             for day in daily_counts
#         ],
#         "hourly_breakdown": [
#             {"hour": int(hour.hour), "detections": hour.count}
#             for hour in hourly_counts
#         ]
#     }


# @router.get("/logs/filter/export")
# async def export_logs_by_date_range(
#     from_date: datetime = Query(..., description="Start date (YYYY-MM-DDTHH:MM:SS)"),
#     to_date: datetime = Query(..., description="End date (YYYY-MM-DDTHH:MM:SS)"),
#     personnel_national_code: str = Query(None, description="Optional: Filter by personnel national code"),
#     format: str = Query("json", description="Export format: json or csv"),
#     db: Session = Depends(get_db)
# ):
#     """
#     Export detection logs within a date range to JSON or CSV format
#     """
#     import json
#     import csv
#     from io import StringIO
#     from ..models.database import Personnel
    
#     # Validate date range
#     if from_date > to_date:
#         raise HTTPException(
#             status_code=400, 
#             detail="from_date must be less than or equal to to_date"
#         )
    
#     # Build query
#     query = db.query(DetectionLog).filter(
#         DetectionLog.detection_time >= from_date,
#         DetectionLog.detection_time <= to_date
#     ).order_by(DetectionLog.detection_time.desc())
    
#     # Apply personnel filter if provided
#     if personnel_national_code:
#         query = query.filter(DetectionLog.person == personnel_national_code)
    
#     logs = query.all()
    
#     # Get personnel lookup
#     all_personnel = db.query(Personnel).all()
#     personnel_lookup = {p.national_code: p for p in all_personnel}
    
#     # Prepare export data
#     export_data = []
#     for log in logs:
#         personnel = personnel_lookup.get(log.person) if log.person else None
        
#         export_data.append({
#             "id": log.id,
#             "person": log.person,
#             "person_name": f"{personnel.fname} {personnel.lname}".strip() if personnel else "Unknown",
#             "confidence": float(log.confidence) if log.confidence else None,
#             "detection_time": log.detection_time.isoformat() if log.detection_time else None,
#             "face_image_path": log.face_image_path,
#             "video_path": log.video_path
#         })
    
#     # Export based on format
#     if format.lower() == "csv":
#         # Create CSV
#         output = StringIO()
#         if export_data:
#             writer = csv.DictWriter(output, fieldnames=export_data[0].keys())
#             writer.writeheader()
#             writer.writerows(export_data)
        
#         csv_content = output.getvalue()
#         output.close()
        
#         filename = f"detection_logs_{from_date.strftime('%Y%m%d')}_to_{to_date.strftime('%Y%m%d')}.csv"
        
#         return StreamingResponse(
#             iter([csv_content]),
#             media_type="text/csv",
#             headers={
#                 "Content-Disposition": f"attachment; filename={filename}"
#             }
#         )
    
#     else:  # JSON format (default)
#         filename = f"detection_logs_{from_date.strftime('%Y%m%d')}_to_{to_date.strftime('%Y%m%d')}.json"
        
#         return StreamingResponse(
#             iter([json.dumps(export_data, indent=2, ensure_ascii=False)]),
#             media_type="application/json",
#             headers={
#                 "Content-Disposition": f"attachment; filename={filename}"
#             }
#         )