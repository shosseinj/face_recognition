import os
import base64
import asyncio
from typing import Optional, Set, Dict
from collections import defaultdict, deque, Counter
from pathlib import Path
import cv2
import av
from datetime import datetime, date, time
import time  # Add this line
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
import uuid  # Add this import

import zipfile
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi import Form  # If needed
import zipfile
from io import BytesIO  # IMPORT THIS!
import tempfile
from pathlib import Path
import cv2
import numpy as np
import os
from typing import List, Optional  # If needed
from .models.database import DetectionLog, get_db, Personnel, PersonnelImage, FACE_STORAGE_DIR

from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, WebSocket, Depends, HTTPException, WebSocketDisconnect, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from .models.database import save_detection_with_face, get_db, DetectionLog, Personnel
from .api import router as api_router
from Detection.TensorRT.infer import ws_transfer, vectorDatabase, removeDatabase
from io import BytesIO  # IMPORT THIS!



processor_task = None
BASE_DIR = Path(__file__).resolve().parent  # This points to backend/app/
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True, parents=True)
print(f"📁 Static files directory: {STATIC_DIR}")


app = FastAPI(
    title="Face Recognition API",
    version="1.0.0",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
UNKNOWN_MIN_FRAMES = 10      # object must appear enough frames
UNKNOWN_SAVE_CLIP = True     # save frames for unknown too


# Custom Swagger UI using local files
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_favicon_url="/static/favicon.png",
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()

# Optional: ReDoc with local files
@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="/static/redoc.standalone.js",  # Download this too if you want ReDoc
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix="/api/v1")
# app.include_router(router, prefix="/api/v1")

class Config:
    RTSP_URL = "rtsp://Jafari:Asd@98500@192.168.110.20:554/Streaming/Channels/301"
    # RTSP_URL = "rtsp://Jafari:Asd@98500@192.168.110.20:554/Streaming/Channels/101"
    # RTSP_URL = "rtsp://Jafari:Asd@98500@192.168.110.14:554/Streaming/Channels/301"
    BASE_URL = "http://192.168.10.9:9000"
    CLIP_LENGTH = 100
    HALF_CLIP = CLIP_LENGTH // 2
    VOTE_WINDOW = 1000

config = Config()


# At the top of your main.py, add this class
class WebSocketManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()  # For thread safety
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        print(f"✅ Client connected. Total: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket):  # ✅ Add async here
        async with self._lock:
            self.active_connections.discard(websocket)
        print(f"❌ Client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast_json(self, message: dict):
        """Send JSON to all connected clients"""
        if not self.active_connections:
            return
        
        disconnected = []
        async with self._lock:
            connections = list(self.active_connections)
        
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Failed to send to client: {e}")
                disconnected.append(connection)
        
        # Clean up dead connections
        async with self._lock:
            for conn in disconnected:
                self.active_connections.discard(conn)
    
    async def broadcast_bytes(self, data: bytes):
        """Send bytes to all connected clients"""
        if not self.active_connections:
            return
        
        disconnected = []
        async with self._lock:
            connections = list(self.active_connections)
        
        for connection in connections:
            try:
                await connection.send_bytes(data)
            except Exception:
                disconnected.append(connection)
        
        async with self._lock:
            for conn in disconnected:
                self.active_connections.discard(conn)
    
    @property
    def count(self) -> int:
        return len(self.active_connections)



# Create global instance
manager = WebSocketManager()



class StreamConfig(BaseModel):
    webcam: bool= False
    gpu: int = 0
    threshold: float = 0.5
    collection: str = "n3"


@app.get("/")
async def index():
    return {
        "message": "Welcome to Face Recognition API",
        "description": "Real-time face recognition with FastAPI WebSocket streaming",
        "endpoints": {
            "docs": "/docs",
            "video_page": "/ws-video",
            "stream_status": "/api/v1/stream/status",
            "reports": "/reports/daily?from_date=YYYY-MM-DD&to_date=YYYY-MM-DD"
        }
    }



@app.get("/ws-video")
async def video_page():
    """Serve the video page HTML"""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "video_page.html")
    
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        # Fallback in case file is not found
        return HTMLResponse(content="<h1>Video page template not found</h1>", status_code=404)




@app.websocket("/ws/video")
async def video_ws(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await save_in_database()
    except Exception as e:
        print(f"⚠️ Error sending initial logs: {e}")

    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(), 
                    timeout=30.0
                )
                # Process messages...
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue
    except WebSocketDisconnect:
        await manager.disconnect(websocket)  # ✅ Add await
    except Exception as e:
        print(f"Error: {e}")
        await manager.disconnect(websocket)  # ✅ Add await




# Global state
processor_task: Optional[asyncio.Task] = None
VOTE_WINDOW = 1000  # how many last frames used

track_history = defaultdict(lambda: {
    "names": deque(maxlen=VOTE_WINDOW),
    "scores": deque(maxlen=VOTE_WINDOW),
    "faces": deque(maxlen=VOTE_WINDOW),
    "frames": deque(maxlen=VOTE_WINDOW),
})







async def video_broadcaster():
    """Background task that streams video and saves detections automatically"""
    print("🎬 Starting video broadcaster...")
    
    # Try camera first, fallback to RTSP
    cap = cv2.VideoCapture(0)
    use_camera = cap.isOpened()
    
    if not use_camera:
        cap.release()
        print("📹 Camera not available, trying RTSP stream...")
    
    # Track last time we sent periodic logs
    last_log_send = time.time()
    
    while True:
        try:
            if not use_camera:
                container = av.open(config.RTSP_URL, options={
                    "rtsp_transport": "tcp", 
                    "flags": "low_delay", 
                    "fflags": "nobuffer"
                })
                print('✅ Connected to RTSP stream')
            
            frame_count = 0
            clip_length = 100
            half_clip = clip_length // 2

            while True:
                # Yield control to event loop
                await asyncio.sleep(0.001)
                
                # REMOVED THE CLIENT COUNT CHECK - Always process frames
                # We still want to save logs periodically
                current_time = time.time()
                if current_time - last_log_send > 5:
                    last_log_send = current_time
                    asyncio.create_task(save_in_database())
                
                # Get frame from either camera or RTSP
                if use_camera:
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:
                    try:
                        frame = next(container.decode(video=0))
                        frame = frame.to_ndarray(format="bgr24")
                    except StopIteration:
                        break
                
                frame_count += 1
                
                # Process with TensorRT
                try:
                    data = ws_transfer(frame)
                    
                    # Process detections and update track history
                    persons = data['persons']
                    scores = data['scores']
                    faces = data['faces']
                    objs = data['objs']
                    
                    current_objs = set(objs)
                    
                    # Update history for current objects
                    for i, (person, score, obj, face) in enumerate(zip(persons, scores, objs, faces)):
                        history = track_history[obj]
                        history["frames"].append(frame.copy())
                        if person:
                            history["names"].append(person)
                            history["scores"].append(score)
                            history["faces"].append(face)
                        
                    # Handle disappeared objects
                    disappeared_objs = set(track_history.keys()) - current_objs
                    
                    for obj in disappeared_objs:
                        history = track_history[obj]

                        total_frames = len(history["frames"])

                        if history["names"]:
                            name_counts = Counter(history["names"])
                            final_name, count = name_counts.most_common(1)[0]

                            if final_name == "Unknown" and count < UNKNOWN_MIN_FRAMES:
                                del track_history[obj]
                                continue
                            # -------- Known person --------
                            final_name = Counter(history["names"]).most_common(1)[0][0]

                            valid_indices = [i for i, n in enumerate(history["names"]) if n == final_name]
                            if valid_indices:
                                best_idx = max(valid_indices, key=lambda i: history["scores"][i])
                                final_score = history["scores"][best_idx]
                                final_face = history["faces"][best_idx]

                                start_idx = max(0, best_idx - half_clip)
                                end_idx = min(len(history["frames"]), best_idx + half_clip)
                                frames_to_save = list(history["frames"])[start_idx:end_idx]

                                log_id = save_detection_with_face(
                                    person=final_name,
                                    confidence=float(final_score),
                                    face_image=final_face,
                                    frames_to_save=frames_to_save
                                )

                                if log_id is not None:
                                    asyncio.create_task(save_in_database())
                                    last_log_send = time.time()

                        else:
                            # -------- Unknown person --------
                            if total_frames >= UNKNOWN_MIN_FRAMES:
                                print(f"👤 Unknown detected (frames={total_frames})")

                                # Take middle frame for clip
                                mid_idx = total_frames // 2
                                start_idx = max(0, mid_idx - half_clip)
                                end_idx = min(total_frames, mid_idx + half_clip)
                                frames_to_save = list(history["frames"])[start_idx:end_idx]

                                log_id = save_detection_with_face(
                                    person="Unknown",
                                    confidence=0.0,
                                    face_image=None,
                                    frames_to_save=frames_to_save
                                )

                                if log_id is not None:
                                    asyncio.create_task(save_in_database())
                                    last_log_send = time.time()

                        del track_history[obj]
                    
                    # Broadcast video frame to clients IF ANY ARE CONNECTED
                    if manager.count > 0:
                        asyncio.create_task(broadcast_frame(data))
                    
                    # Yield again after processing
                    await asyncio.sleep(0)
                    
                except Exception as e:
                    print(f"⚠️ Error processing frame: {e}")
                    continue
                        
        except Exception as e:
            print(f"⚠️ Stream connection error: {e}")
            print("🔄 Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
        finally:
            if not use_camera and 'container' in locals():
                container.close()


def save_disappeared_object(obj, history):
    """Save a disappeared object to database"""
    final_name = Counter(history["names"]).most_common(1)[0][0]
    
    valid_indices = [i for i, n in enumerate(history["names"]) if n == final_name]
    if valid_indices:
        best_idx = max(valid_indices, key=lambda i: history["scores"][i])
        final_score = history["scores"][best_idx]
        final_face = history["faces"][best_idx]
        
        start_idx = max(0, best_idx - config.HALF_CLIP)
        end_idx = min(len(history["frames"]), best_idx + config.HALF_CLIP)
        frames_to_save = list(history["frames"])[start_idx:end_idx]
        
        save_detection_with_face(
            person=final_name,
            confidence=float(final_score),
            face_image=final_face,
            frames_to_save=frames_to_save
        )

async def broadcast_frame(data: dict):
    """Broadcast frame and metadata using the manager"""
    frame_bytes = data['frame']
    metadata = {
        "type": "video_metadata",
        "persons": data['persons'],
        "scores": [float(s) if s is not None else 0.0 for s in data['scores']],
        "objs": data['objs'],
    }
    
    # Send to all clients - manager handles dead connections
    if manager.count > 0:
        # Don't await these - let them run in background
        asyncio.create_task(manager.broadcast_bytes(frame_bytes))
        asyncio.create_task(manager.broadcast_json(metadata))



async def save_in_database():
    """Broadcast new detection to all connected clients with full name"""
    offset = 0
    limit = 20
    db = next(get_db())
    try:
        # Get all logs
        logs = db.query(DetectionLog)\
            .order_by(DetectionLog.detection_time.desc())\
            .offset(offset)\
            .limit(limit)\
            .all()
        
        # Get all personnel for quick lookup
        all_personnel = db.query(Personnel).all()
        personnel_lookup = {p.national_code: p for p in all_personnel}
        
        logs_data = []
        base_url = config.BASE_URL
        
        for log in logs:
            face_image_base64 = None
            if log.face_image_path and os.path.exists(log.face_image_path):
                try:
                    img = cv2.imread(log.face_image_path)
                    if img is not None:
                        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        face_image_base64 = base64.b64encode(buffer).decode('utf-8')
                except:
                    pass
            
            video_url = None
            if log.video_path and os.path.exists(log.video_path):
                video_url = f"{base_url}/api/v1/detections/{log.id}/video"
            
            # Calculate full name if personnel exists
            full_name = None
            person_value = log.person
            
            if person_value and person_value.isdigit() and len(person_value) == 10:
                personnel = personnel_lookup.get(person_value)
                if personnel:
                    full_name = f"{personnel.fname} {personnel.lname}".strip()
            
            logs_data.append({
                "id": log.id,
                "person": log.person,  # National code
                "full_name": full_name,  # Full name if found, else None
                "confidence": float(log.confidence) if log.confidence is not None else 0.0,
                "detection_time": log.detection_time.isoformat() if log.detection_time else None,
                "face_image_base64": face_image_base64,
                "video_url": video_url
            })
        
        detection_info = {
            "type": "new_detection",
            "detection": logs_data
        }
        
        await manager.broadcast_json(detection_info)
        print(f"📊 Broadcast to {manager.count} clients")
    finally:
        db.close()


@app.get("/connections/status")
async def connection_status():
    """Get current connection statistics"""
    return {
        "active_connections": manager.count,  # Changed from connection_count to count
        "status": "healthy"
    }


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("Starting up Face Recognition API...")
    
    # Start video broadcaster when server starts
    global processor_task
    processor_task = asyncio.create_task(video_broadcaster())
    print("🎬 Video broadcaster started automatically")
    
    print("Models will be loaded on first WebSocket connection")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    global processor_task
    if processor_task:
        processor_task.cancel()
        try:
            await processor_task
        except asyncio.CancelledError:
            pass
    print("Video broadcaster stopped")






# Define Pydantic model for request body
class SaveRequest(BaseModel):
    id: int
    name_front: str

@app.post("/ws-save")
async def save_detection(
    request: SaveRequest,
    db: Session = Depends(get_db)
):
    """Save a detection to the vector database"""
    print(f'Processing save request for ID: {request.id}, name: {request.name_front}')
    
    try:
        # Get detection log
        detection_log = db.query(DetectionLog).filter(DetectionLog.id == request.id).first()
        
        if not detection_log:
            raise HTTPException(status_code=404, detail=f"Detection log with ID {request.id} not found")
        
        if not detection_log.face_image_path:
            raise HTTPException(status_code=400, detail="No face image path found in this log entry")
        
        if not os.path.exists(detection_log.face_image_path):
            raise HTTPException(status_code=404, detail=f"Face image file not found at: {detection_log.face_image_path}")
        
        # Read and process image
        img = cv2.imread(detection_log.face_image_path)
        if img is None:
            raise HTTPException(status_code=500, detail=f"Failed to read image file from: {detection_log.face_image_path}")
        
        # Add to vector database
        result = vectorDatabase(img, request.name_front)
        
        # Update the person name
        detection_log.person = request.name_front
        db.commit()
        
        return {
            "success": True,
            "message": f"Image added to vector database for {request.name_front}",
            "log_id": request.id,
            "corrected_person": request.name_front,
            "image_shape": img.shape,
            "detection_time": detection_log.detection_time.isoformat() if detection_log.detection_time else None
        }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing log ID {request.id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    print(f"Global error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred"}
    )



 
@app.post("/upload-personnel-zip")
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
                        import shutil
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
                        vectorDatabase(img, national_code, save_mode=True, ref_img_id=db_image.id)
                        
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


@app.get("/qd-remove")
async def qd_remove(
    # request: SaveRequest,
    # db: Session = Depends(get_db)
):
    # print('id', request.id)
    # # name_to_ai = dictionary.get(request.name_front)
    # name_to_ai = request.name_front
    try:
        name_to_ai = ['0311344119']
        result = removeDatabase(name_to_ai)
   
        return {
            "success": True,
            "message": f"Image added to vector database for {name_to_ai}",
            # "log_id": request.id,
            "corrected_person": name_to_ai,
            "vector_database_result": result if result else "Success",
        }
            
    except Exception as e:
        # print(f"Error processing log ID {request.id}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        print('Processing complete')
    
    
    
    
    
    


























































































# async def video_broadcaster():
#     """Background task that streams video to all connected clients"""
#     print("🎬 Starting video broadcaster...")
    
#     # Try camera first, fallback to RTSP
#     cap = cv2.VideoCapture(0)
#     use_camera = cap.isOpened()
    
#     if not use_camera:
#         cap.release()
#         print("📹 Camera not available, trying RTSP stream...")
    
#     # Track last time we sent periodic logs
#     last_log_send = time.time()
    
#     while True:
#         try:
#             if not use_camera:
#                 container = av.open(config.RTSP_URL, options={
#                     "rtsp_transport": "tcp", 
#                     "flags": "low_delay", 
#                     "fflags": "nobuffer"
#                 })
#                 print('✅ Connected to RTSP stream')
            
#             frame_count = 0
#             clip_length = 100
#             half_clip = clip_length // 2

#             while True:
#                 # Yield control to event loop
#                 await asyncio.sleep(0.001)
                
#                 if manager.count == 0:
#                     await asyncio.sleep(0.1)
#                     continue
                
#                 # Send logs periodically (every 5 seconds) as backup
#                 current_time = time.time()
#                 if current_time - last_log_send > 5:
#                     last_log_send = current_time
#                     asyncio.create_task(save_in_database())
                
#                 # Get frame from either camera or RTSP
#                 if use_camera:
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
#                 else:
#                     try:
#                         frame = next(container.decode(video=0))
#                         frame = frame.to_ndarray(format="bgr24")
#                     except StopIteration:
#                         break
                
#                 frame_count += 1
                
#                 # Process with TensorRT
#                 try:
#                     data = ws_transfer(frame)
                    
#                     # Process detections and update track history
#                     persons = data['persons']
#                     scores = data['scores']
#                     faces = data['faces']
#                     objs = data['objs']
                    
#                     current_objs = set(objs)
                    
#                     # Update history for current objects
#                     for i, (person, score, obj, face) in enumerate(zip(persons, scores, objs, faces)):
#                         history = track_history[obj]
#                         history["frames"].append(frame.copy())
#                         if person:
#                             history["names"].append(person)
#                             history["scores"].append(score)
#                             history["faces"].append(face)
                        
#                     # Handle disappeared objects
#                     disappeared_objs = set(track_history.keys()) - current_objs
                    
#                     for obj in disappeared_objs:
#                         history = track_history[obj]

#                         total_frames = len(history["frames"])

#                         if history["names"]:
#                             name_counts = Counter(history["names"])
#                             final_name, count = name_counts.most_common(1)[0]

#                             if final_name == "Unknown" and count < UNKNOWN_MIN_FRAMES:
#                                 del track_history[obj]
#                                 continue
#                             # -------- Known person --------
#                             final_name = Counter(history["names"]).most_common(1)[0][0]

#                             valid_indices = [i for i, n in enumerate(history["names"]) if n == final_name]
#                             if valid_indices:
#                                 best_idx = max(valid_indices, key=lambda i: history["scores"][i])
#                                 final_score = history["scores"][best_idx]
#                                 final_face = history["faces"][best_idx]

#                                 start_idx = max(0, best_idx - half_clip)
#                                 end_idx = min(len(history["frames"]), best_idx + half_clip)
#                                 frames_to_save = list(history["frames"])[start_idx:end_idx]

#                                 log_id = save_detection_with_face(
#                                     person=final_name,
#                                     confidence=float(final_score),
#                                     face_image=final_face,
#                                     frames_to_save=frames_to_save
#                                 )

#                                 if log_id is not None:
#                                     asyncio.create_task(save_in_database())
#                                     last_log_send = time.time()

#                         else:
#                             # -------- Unknown person --------
#                             if total_frames >= UNKNOWN_MIN_FRAMES:
#                                 print(f"👤 Unknown detected (frames={total_frames})")

#                                 # Take middle frame for clip
#                                 mid_idx = total_frames // 2
#                                 start_idx = max(0, mid_idx - half_clip)
#                                 end_idx = min(total_frames, mid_idx + half_clip)
#                                 frames_to_save = list(history["frames"])[start_idx:end_idx]

#                                 log_id = save_detection_with_face(
#                                     person="Unknown",
#                                     confidence=0.0,
#                                     face_image=None,   # or history["faces"][mid_idx] if available
#                                     frames_to_save=frames_to_save
#                                 )

#                                 if log_id is not None:
#                                     asyncio.create_task(save_in_database())
#                                     last_log_send = time.time()

#                         del track_history[obj]
                    
#                     # Broadcast video frame to clients
#                     asyncio.create_task(broadcast_frame(data))
                    
#                     # Yield again after processing
#                     await asyncio.sleep(0)
                    
#                 except Exception as e:
#                     print(f"⚠️ Error processing frame: {e}")
#                     continue
                        
#         except Exception as e:
#             print(f"⚠️ Stream connection error: {e}")
#             print("🔄 Reconnecting in 5 seconds...")
#             await asyncio.sleep(5)
#         finally:
#             if not use_camera and 'container' in locals():
#                 container.close()












# @app.post("/upload-personnel-zip")
# async def upload_personnel_zip(
#     file: UploadFile = File(...),
#     skip_invalid_national_codes: bool = False  # Optional parameter
# ):
#     """
#     Upload a zip file containing folders named with national codes
    
#     - **file**: ZIP file containing folders named with national codes
#     - **skip_invalid_national_codes**: Skip folders that don't have valid national code format (default: False)
#     """
#     # Validate file type
#     if not file.filename.endswith('.zip'):
#         raise HTTPException(status_code=400, detail="Only zip files are accepted")
    
#     # Allowed image extensions
#     ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    
#     try:
#         contents = await file.read()
#         zip_data = BytesIO(contents)
        
#         # Quick validation - check if it's a valid zip
#         if not zipfile.is_zipfile(zip_data):
#             raise HTTPException(status_code=400, detail="Invalid zip file")
            
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Could not read zip file: {str(e)}")
    
#     processed_count = 0
#     error_count = 0
#     results = []
#     skipped_folders = []
    
#     with zipfile.ZipFile(zip_data, 'r') as zip_ref:
#         with tempfile.TemporaryDirectory() as temp_dir:
#             zip_ref.extractall(temp_dir)
#             temp_path = Path(temp_dir)
            
#             print(f"\n📦 Processing zip: {file.filename}")
            
#             for person_dir in temp_path.iterdir():
#                 if not person_dir.is_dir():
#                     continue
                
#                 national_code = person_dir.name
                
#                 # Validate national code format (10 digits)
#                 is_valid_national_code = national_code.isdigit() and len(national_code) == 10
                
#                 if not is_valid_national_code:
#                     warning_msg = f"⚠️ '{national_code}' is not a valid 10-digit national code"
#                     if skip_invalid_national_codes:
#                         print(f"{warning_msg} - SKIPPING")
#                         skipped_folders.append({
#                             "folder": national_code,
#                             "reason": "Invalid national code format"
#                         })
#                         continue
#                     else:
#                         print(f"{warning_msg} - PROCESSING ANYWAY")
                
#                 print(f"\n👤 Processing: {national_code}")
                
#                 # Get all image files
#                 image_files = []
#                 for ext in ALLOWED_EXTENSIONS:
#                     image_files.extend(person_dir.glob(f"*{ext}"))
#                     image_files.extend(person_dir.glob(f"*{ext.upper()}"))
                
#                 if not image_files:
#                     print(f"  ⚠️ No images found")
#                     results.append({
#                         "national_code": national_code,
#                         "status": "warning",
#                         "message": "No images found in folder",
#                         "valid_format": is_valid_national_code
#                     })
#                     continue
                
#                 person_processed = 0
#                 person_errors = 0
#                 error_details = []
                
#                 for img_path in image_files:
#                     try:
#                         # Read image
#                         img = cv2.imread(str(img_path))
#                         if img is None:
#                             raise ValueError("Could not decode image")
                        
#                         # Save to vector database
#                         vectorDatabase(img, national_code, save_mode=True)
                        
#                         print(f"  ✅ {img_path.name}")
#                         person_processed += 1
                        
#                     except Exception as e:
#                         print(f"  ❌ {img_path.name}: {str(e)}")
#                         person_errors += 1
#                         error_details.append({
#                             "file": img_path.name,
#                             "error": str(e)
#                         })
                
#                 processed_count += person_processed
#                 error_count += person_errors
                
#                 results.append({
#                     "national_code": national_code,
#                     "status": "success" if person_processed > 0 else "error",
#                     "valid_format": is_valid_national_code,
#                     "processed": person_processed,
#                     "errors": person_errors,
#                     "total_images": len(image_files),
#                     "error_details": error_details if error_details else None
#                 })
    
#     return {
#         "success": True,
#         "filename": file.filename,
#         "message": f"Processed {processed_count} images, {error_count} errors",
#         "summary": {
#             "total_processed": processed_count,
#             "total_errors": error_count,
#             "total_persons": len(results),
#             "skipped_folders": len(skipped_folders)
#         },
#         "details": results,
#         "skipped_folders": skipped_folders if skipped_folders else None
#     }







# from pathlib import Path
# @app.get("/qs-save")
# async def video_ws1(
#     # request: SaveRequest,
#     # db: Session = Depends(get_db)
# ):
#     IMAGE_ROOT = "./Detection/TensorRT/qdrant_images"


#     root = Path(IMAGE_ROOT)

#     for person_dir in root.iterdir():
#         if not person_dir.is_dir():
#             print('empttyyyyyyyyyyyyyyyyyyy')
#             continue

#         person_name = person_dir.name
#         print(f"\n📁 Person: {person_name}")

#         for img_path in person_dir.glob("*.*"):
#             try:
#                 img = cv2.imread(str(img_path))
#                 if img is None:
#                     print("❌ cannot read", img_path.name)
#                     continue
#                 vectorDatabase(img, f'{person_name}', save_mode=True)


#                 print("✅ inserted", img_path.name)

#             except Exception as e:
#                 print("❌ error:", img_path.name, e)
        
#     return {
#             "success": True,
            
#         }

# @app.post("/upload-personnel-zip")
# async def upload_personnel_zip(
#     file: UploadFile = File(...),
#     skip_invalid_national_codes: bool = False
# ):
#     # Validate file type
#     if not file.filename.endswith('.zip'):
#         raise HTTPException(status_code=400, detail="Only zip files are accepted")
    
#     # Allowed image extensions
#     ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    
#     try:
#         contents = await file.read()
#         print(f"📦 Zip file size: {len(contents)} bytes")
#         zip_data = BytesIO(contents)
        
#         if not zipfile.is_zipfile(zip_data):
#             raise HTTPException(status_code=400, detail="Invalid zip file")
            
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Could not read zip file: {str(e)}")
    
#     processed_count = 0
#     error_count = 0
#     results = []
#     skipped_folders = []
    
#     with zipfile.ZipFile(zip_data, 'r') as zip_ref:
#         # List all files in zip for debugging
#         print("\n📋 Files in zip:")
#         file_list = zip_ref.namelist()
#         for f in file_list:
#             print(f"  - {f}")
        
#         with tempfile.TemporaryDirectory() as temp_dir:
#             print(f"\n📂 Extracting to: {temp_dir}")
#             zip_ref.extractall(temp_dir)
#             temp_path = Path(temp_dir)
            
#             # List extracted directories
#             print("\n📁 Extracted directories:")
#             for item in temp_path.iterdir():
#                 print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
            
#             for person_dir in temp_path.iterdir():
#                 if not person_dir.is_dir():
#                     print(f"  ⚠️ Skipping non-directory: {person_dir.name}")
#                     continue
                
#                 national_code = person_dir.name
#                 print(f"\n{'='*50}")
#                 print(f"👤 Processing person: {national_code}")
#                 print(f"📁 Full path: {person_dir}")
                
#                 # List all files in this directory
#                 all_files = list(person_dir.glob("*"))
#                 print(f"  📋 All files in directory ({len(all_files)}):")
#                 for f in all_files:
#                     print(f"    - {f.name} (extension: {f.suffix})")
                
#                 # Validate national code format
#                 is_valid_national_code = national_code.isdigit() and len(national_code) == 10
#                 print(f"  ✅ Valid national code: {is_valid_national_code}")
                
#                 if not is_valid_national_code:
#                     warning_msg = f"⚠️ '{national_code}' is not a valid 10-digit national code"
#                     if skip_invalid_national_codes:
#                         print(f"{warning_msg} - SKIPPING")
#                         skipped_folders.append({
#                             "folder": national_code,
#                             "reason": "Invalid national code format"
#                         })
#                         continue
#                     else:
#                         print(f"{warning_msg} - PROCESSING ANYWAY")
                
#                 # Get image files
#                 image_files = []
#                 for ext in ALLOWED_EXTENSIONS:
#                     # Check both lowercase and uppercase
#                     for pattern in [f"*{ext}", f"*{ext.upper()}"]:
#                         found = list(person_dir.glob(pattern))
#                         if found:
#                             print(f"  🔍 Found {len(found)} files matching {pattern}")
#                             image_files.extend(found)
                
#                 # Remove duplicates while preserving order
#                 unique_files = []
#                 seen = set()
#                 for f in image_files:
#                     if f.name not in seen:
#                         seen.add(f.name)
#                         unique_files.append(f)
                
#                 print(f"  🖼️ Total image files found: {len(image_files)}")
#                 print(f"  🖼️ Unique image files: {len(unique_files)}")
                
#                 if not unique_files:
#                     print(f"  ⚠️ No images found")
#                     results.append({
#                         "national_code": national_code,
#                         "status": "warning",
#                         "message": "No images found in folder",
#                         "valid_format": is_valid_national_code
#                     })
#                     continue
                
#                 person_processed = 0
#                 person_errors = 0
#                 error_details = []
                
#                 for i, img_path in enumerate(unique_files):
#                     try:
#                         print(f"\n  📸 Processing image {i+1}/{len(unique_files)}: {img_path.name}")
                        
#                         # Read image
#                         img = cv2.imread(str(img_path))
#                         if img is None:
#                             raise ValueError("Could not decode image")
                        
#                         print(f"     Image shape: {img.shape if img is not None else 'None'}")
                        
#                         # Save to vector database
#                         print(f"     Calling vectorDatabase with national_code: {national_code}")
#                         vectorDatabase(img, national_code, save_mode=True)
                        
#                         print(f"  ✅ Successfully inserted: {img_path.name}")
#                         person_processed += 1
                        
#                     except Exception as e:
#                         print(f"  ❌ Error processing {img_path.name}: {str(e)}")
#                         import traceback
#                         traceback.print_exc()
#                         person_errors += 1
#                         error_details.append({
#                             "file": img_path.name,
#                             "error": str(e)
#                         })
                
#                 processed_count += person_processed
#                 error_count += person_errors
                
#                 results.append({
#                     "national_code": national_code,
#                     "status": "success" if person_processed > 0 else "error",
#                     "valid_format": is_valid_national_code,
#                     "processed": person_processed,
#                     "errors": person_errors,
#                     "total_images": len(unique_files),
#                     "error_details": error_details if error_details else None
#                 })
                
#                 print(f"\n  📊 Summary for {national_code}: {person_processed}/{len(unique_files)} processed, {person_errors} errors")
    
#     print(f"\n{'='*50}")
#     print(f"📊 FINAL SUMMARY")
#     print(f"{'='*50}")
#     print(f"Total processed: {processed_count}")
#     print(f"Total errors: {error_count}")
#     print(f"Total persons: {len(results)}")
    
#     return {
#         "success": True,
#         "filename": file.filename,
#         "message": f"Processed {processed_count} images, {error_count} errors",
#         "summary": {
#             "total_processed": processed_count,
#             "total_errors": error_count,
#             "total_persons": len(results),
#             "skipped_folders": len(skipped_folders)
#         },
#         "details": results,
#         "skipped_folders": skipped_folders if skipped_folders else None
#     }





































# import os
# import base64
# import asyncio
# from typing import Optional, Set, Dict
# from collections import defaultdict, deque, Counter
# from pathlib import Path
# import cv2
# import av
# from datetime import datetime, date, time
# import time  # Add this line
# from fastapi.openapi.docs import (
#     get_redoc_html,
#     get_swagger_ui_html,
#     get_swagger_ui_oauth2_redirect_html,
# )
# from fastapi.staticfiles import StaticFiles
# from fastapi import FastAPI, WebSocket, Depends, HTTPException, WebSocketDisconnect
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from sqlalchemy.orm import Session
# from .models.database import save_detection_with_face, get_db, DetectionLog, Personnel
# from .api import router as api_router
# from Detection.TensorRT.infer import ws_transfer, vectorDatabase, removeDatabase

# processor_task = None


# BASE_DIR = Path(__file__).resolve().parent  # This points to backend/app/
# STATIC_DIR = BASE_DIR / "static"
# STATIC_DIR.mkdir(exist_ok=True, parents=True)
# print(f"📁 Static files directory: {STATIC_DIR}")


# app = FastAPI(
#     title="Face Recognition API",
#     version="1.0.0",
#     docs_url=None,  # Disable default docs
#     redoc_url=None,  # Disable default redoc
# )

# app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# # Custom Swagger UI using local files
# @app.get("/docs", include_in_schema=False)
# async def custom_swagger_ui_html():
#     return get_swagger_ui_html(
#         openapi_url=app.openapi_url,
#         title=app.title + " - Swagger UI",
#         oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
#         swagger_js_url="/static/swagger-ui-bundle.js",
#         swagger_css_url="/static/swagger-ui.css",
#         swagger_favicon_url="/static/favicon.png",
#     )


# @app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
# async def swagger_ui_redirect():
#     return get_swagger_ui_oauth2_redirect_html()

# # Optional: ReDoc with local files
# @app.get("/redoc", include_in_schema=False)
# async def redoc_html():
#     return get_redoc_html(
#         openapi_url=app.openapi_url,
#         title=app.title + " - ReDoc",
#         redoc_js_url="/static/redoc.standalone.js",  # Download this too if you want ReDoc
#     )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# app.include_router(api_router, prefix="/api/v1")
# # app.include_router(router, prefix="/api/v1")

# class Config:
#     RTSP_URL = "rtsp://Jafari:Asd12345@192.168.110.20:554/Streaming/Channels/301"
#     BASE_URL = "http://192.168.10.9:8001"
#     CLIP_LENGTH = 100
#     HALF_CLIP = CLIP_LENGTH // 2
#     VOTE_WINDOW = 1000

# config = Config()


# # At the top of your main.py, add this class
# class WebSocketManager:
#     def __init__(self):
#         self.active_connections: Set[WebSocket] = set()
#         self._lock = asyncio.Lock()  # For thread safety
    
#     async def connect(self, websocket: WebSocket):
#         await websocket.accept()
#         async with self._lock:
#             self.active_connections.add(websocket)
#         print(f"✅ Client connected. Total: {len(self.active_connections)}")
    
#     async def disconnect(self, websocket: WebSocket):  # ✅ Add async here
#         async with self._lock:
#             self.active_connections.discard(websocket)
#         print(f"❌ Client disconnected. Total: {len(self.active_connections)}")
    
#     async def broadcast_json(self, message: dict):
#         """Send JSON to all connected clients"""
#         if not self.active_connections:
#             return
        
#         disconnected = []
#         async with self._lock:
#             connections = list(self.active_connections)
        
#         for connection in connections:
#             try:
#                 await connection.send_json(message)
#             except Exception as e:
#                 print(f"Failed to send to client: {e}")
#                 disconnected.append(connection)
        
#         # Clean up dead connections
#         async with self._lock:
#             for conn in disconnected:
#                 self.active_connections.discard(conn)
    
#     async def broadcast_bytes(self, data: bytes):
#         """Send bytes to all connected clients"""
#         if not self.active_connections:
#             return
        
#         disconnected = []
#         async with self._lock:
#             connections = list(self.active_connections)
        
#         for connection in connections:
#             try:
#                 await connection.send_bytes(data)
#             except Exception:
#                 disconnected.append(connection)
        
#         async with self._lock:
#             for conn in disconnected:
#                 self.active_connections.discard(conn)
    
#     @property
#     def count(self) -> int:
#         return len(self.active_connections)



# # Create global instance
# manager = WebSocketManager()



# class StreamConfig(BaseModel):
#     webcam: bool= False
#     gpu: int = 0
#     threshold: float = 0.5
#     collection: str = "n3"


# @app.get("/")
# async def index():
#     return {
#         "message": "Welcome to Face Recognition API",
#         "description": "Real-time face recognition with FastAPI WebSocket streaming",
#         "endpoints": {
#             "docs": "/docs",
#             "video_page": "/ws-video",
#             "stream_status": "/api/v1/stream/status",
#             "reports": "/reports/daily?from_date=YYYY-MM-DD&to_date=YYYY-MM-DD"
#         }
#     }



# @app.get("/ws-video")
# async def video_page():
#     """Serve the video page HTML"""
#     html_path = os.path.join(os.path.dirname(__file__), "templates", "video_page.html")
    
#     try:
#         with open(html_path, "r", encoding="utf-8") as f:
#             html_content = f.read()
#         return HTMLResponse(content=html_content)
#     except FileNotFoundError:
#         # Fallback in case file is not found
#         return HTMLResponse(content="<h1>Video page template not found</h1>", status_code=404)




# @app.websocket("/ws/video")
# async def video_ws(websocket: WebSocket):
#     await manager.connect(websocket)
#     try:
#         await save_in_database()
#     except Exception as e:
#         print(f"⚠️ Error sending initial logs: {e}")

#     try:
#         while True:
#             try:
#                 data = await asyncio.wait_for(
#                     websocket.receive_text(), 
#                     timeout=30.0
#                 )
#                 # Process messages...
#             except asyncio.TimeoutError:
#                 await websocket.send_json({"type": "ping"})
#                 continue
#     except WebSocketDisconnect:
#         await manager.disconnect(websocket)  # ✅ Add await
#     except Exception as e:
#         print(f"Error: {e}")
#         await manager.disconnect(websocket)  # ✅ Add await




# # Global state
# processor_task: Optional[asyncio.Task] = None
# VOTE_WINDOW = 1000  # how many last frames used

# track_history = defaultdict(lambda: {
#     "names": deque(maxlen=VOTE_WINDOW),
#     "scores": deque(maxlen=VOTE_WINDOW),
#     "faces": deque(maxlen=VOTE_WINDOW),
#     "frames": deque(maxlen=VOTE_WINDOW),
# })






# async def video_broadcaster():
#     """Background task that streams video to all connected clients"""
#     print("🎬 Starting video broadcaster...")
    
#     # Try camera first, fallback to RTSP
#     cap = cv2.VideoCapture(0)
#     use_camera = cap.isOpened()
    
#     if not use_camera:
#         cap.release()
#         print("📹 Camera not available, trying RTSP stream...")
    
#     # Track last time we sent periodic logs
#     last_log_send = time.time()
    
#     while True:
#         try:
#             if not use_camera:
#                 container = av.open(config.RTSP_URL, options={
#                     "rtsp_transport": "tcp", 
#                     "flags": "low_delay", 
#                     "fflags": "nobuffer"
#                 })
#                 print('✅ Connected to RTSP stream')
            
#             frame_count = 0
#             clip_length = 100
#             half_clip = clip_length // 2

#             while True:
#                 # Yield control to event loop
#                 await asyncio.sleep(0.001)
                
#                 if manager.count == 0:
#                     await asyncio.sleep(0.1)
#                     continue
                
#                 # Send logs periodically (every 5 seconds) as backup
#                 current_time = time.time()
#                 if current_time - last_log_send > 5:
#                     last_log_send = current_time
#                     asyncio.create_task(save_in_database())
                
#                 # Get frame from either camera or RTSP
#                 if use_camera:
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
#                 else:
#                     try:
#                         frame = next(container.decode(video=0))
#                         frame = frame.to_ndarray(format="bgr24")
#                     except StopIteration:
#                         break
                
#                 frame_count += 1
                
#                 # Process with TensorRT
#                 try:
#                     data = ws_transfer(frame)
                    
#                     # Process detections and update track history
#                     persons = data['persons']
#                     scores = data['scores']
#                     faces = data['faces']
#                     objs = data['objs']
                    
#                     current_objs = set(objs)
                    
#                     # Update history for current objects
#                     for i, (person, score, obj, face) in enumerate(zip(persons, scores, objs, faces)):
#                         history = track_history[obj]
#                         history["frames"].append(frame.copy())
#                         if person and person != "Unknown":
#                             history["names"].append(person)
#                             history["scores"].append(score)
#                             history["faces"].append(face)
                    
#                     # Handle disappeared objects
#                     disappeared_objs = set(track_history.keys()) - current_objs
                    
#                     for obj in disappeared_objs:
#                         history = track_history[obj]
#                         if history["names"]:
#                             final_name = Counter(history["names"]).most_common(1)[0][0]
                            
#                             valid_indices = [i for i, n in enumerate(history["names"]) if n == final_name]
#                             if valid_indices:
#                                 best_idx = max(valid_indices, key=lambda i: history["scores"][i])
#                                 final_score = history["scores"][best_idx]
#                                 final_face = history["faces"][best_idx]
                                
#                                 start_idx = max(0, best_idx - half_clip)
#                                 end_idx = min(len(history["frames"]), best_idx + half_clip)
#                                 frames_to_save = list(history["frames"])[start_idx:end_idx]
                                
#                                 log_id = save_detection_with_face(
#                                     person=final_name,
#                                     confidence=float(final_score),
#                                     face_image=final_face,
#                                     frames_to_save=frames_to_save
#                                 )
                                
#                                 if log_id is not None:
#                                     # IMMEDIATELY send the new logs when a detection is saved
#                                     asyncio.create_task(save_in_database())
#                                     last_log_send = time.time()  # Reset the periodic timer
                        
#                         del track_history[obj]
                    
#                     # Broadcast video frame to clients
#                     asyncio.create_task(broadcast_frame(data))
                    
#                     # Yield again after processing
#                     await asyncio.sleep(0)
                    
#                 except Exception as e:
#                     print(f"⚠️ Error processing frame: {e}")
#                     continue
                        
#         except Exception as e:
#             print(f"⚠️ Stream connection error: {e}")
#             print("🔄 Reconnecting in 5 seconds...")
#             await asyncio.sleep(5)
#         finally:
#             if not use_camera and 'container' in locals():
#                 container.close()





# # def process_detections(data: dict, frame_count: int, frame):  # ✅ Added frame parameter
# #     """Process detections and update track history"""
# #     persons = data['persons']
# #     scores = data['scores']
# #     faces = data['faces']
# #     objs = data['objs']
    
# #     current_objs = set(objs)
    
# #     # Update history for current objects
# #     for i, (person, score, obj, face) in enumerate(zip(persons, scores, objs, faces)):
# #         history = track_history[obj]
# #         history["frames"].append(frame.copy())  # ✅ Now frame is defined
# #         if person and person != "Unknown":
# #             history["names"].append(person)
# #             history["scores"].append(score)
# #             history["faces"].append(face)
    
# #     # Handle disappeared objects
# #     disappeared_objs = set(track_history.keys()) - current_objs
    
# #     for obj in disappeared_objs:
# #         history = track_history[obj]
# #         if history["names"]:
# #             save_disappeared_object(obj, history)
# #         del track_history[obj]

# def save_disappeared_object(obj, history):
#     """Save a disappeared object to database"""
#     final_name = Counter(history["names"]).most_common(1)[0][0]
    
#     valid_indices = [i for i, n in enumerate(history["names"]) if n == final_name]
#     if valid_indices:
#         best_idx = max(valid_indices, key=lambda i: history["scores"][i])
#         final_score = history["scores"][best_idx]
#         final_face = history["faces"][best_idx]
        
#         start_idx = max(0, best_idx - config.HALF_CLIP)
#         end_idx = min(len(history["frames"]), best_idx + config.HALF_CLIP)
#         frames_to_save = list(history["frames"])[start_idx:end_idx]
        
#         save_detection_with_face(
#             person=final_name,
#             confidence=float(final_score),
#             face_image=final_face,
#             frames_to_save=frames_to_save
#         )

# async def broadcast_frame(data: dict):
#     """Broadcast frame and metadata using the manager"""
#     frame_bytes = data['frame']
#     metadata = {
#         "type": "video_metadata",
#         "persons": data['persons'],
#         "scores": [float(s) if s is not None else 0.0 for s in data['scores']],
#         "objs": data['objs'],
#     }
    
#     # Send to all clients - manager handles dead connections
#     if manager.count > 0:
#         # Don't await these - let them run in background
#         asyncio.create_task(manager.broadcast_bytes(frame_bytes))
#         asyncio.create_task(manager.broadcast_json(metadata))


# # async def save_in_database():
# #     """Broadcast new detection to all connected clients"""
# #     offset = 0
# #     limit = 20
# #     db = next(get_db())
# #     try:  # ✅ Added try-finally
# #         logs = db.query(DetectionLog)\
# #             .order_by(DetectionLog.detection_time.desc())\
# #             .offset(offset)\
# #             .limit(limit)\
# #             .all()
        
# #         logs_data = []
# #         base_url = config.BASE_URL
        
# #         for log in logs:
# #             face_image_base64 = None
# #             if log.face_image_path and os.path.exists(log.face_image_path):
# #                 try:
# #                     img = cv2.imread(log.face_image_path)
# #                     if img is not None:
# #                         #img_rgb = img#cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ####
# #                         _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
# #                         face_image_base64 = base64.b64encode(buffer).decode('utf-8')
# #                 except:
# #                     pass
            
# #             video_url = None
# #             if log.video_path and os.path.exists(log.video_path):
# #                 video_url = f"{base_url}/api/v1/detections/{log.id}/video"
            
# #             logs_data.append({
# #                 "id": log.id,
# #                 "person": log.person,
# #                 "confidence": float(log.confidence) if log.confidence is not None else 0.0,
# #                 "detection_time": log.detection_time.isoformat() if log.detection_time else None,
# #                 "face_image_path": log.face_image_path,
# #                 "face_image_base64": face_image_base64,
# #                 "video_url": video_url
# #             })
        
# #         detection_info = {
# #             "type": "new_detection",
# #             "detection": logs_data
# #         }
        
# #         await manager.broadcast_json(detection_info)
# #         print(f"📊 Broadcast to {manager.count} clients")  # ✅ Using count property
# #     finally:
# #         db.close()  # ✅ Close the session
      
# async def save_in_database():
#     """Broadcast new detection to all connected clients with full name"""
#     offset = 0
#     limit = 20
#     db = next(get_db())
#     try:
#         # Get all logs
#         logs = db.query(DetectionLog)\
#             .order_by(DetectionLog.detection_time.desc())\
#             .offset(offset)\
#             .limit(limit)\
#             .all()
        
#         # Get all personnel for quick lookup
#         all_personnel = db.query(Personnel).all()
#         personnel_lookup = {p.national_code: p for p in all_personnel}
        
#         logs_data = []
#         base_url = config.BASE_URL
        
#         for log in logs:
#             face_image_base64 = None
#             if log.face_image_path and os.path.exists(log.face_image_path):
#                 try:
#                     img = cv2.imread(log.face_image_path)
#                     if img is not None:
#                         _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
#                         face_image_base64 = base64.b64encode(buffer).decode('utf-8')
#                 except:
#                     pass
            
#             video_url = None
#             if log.video_path and os.path.exists(log.video_path):
#                 video_url = f"{base_url}/api/v1/detections/{log.id}/video"
            
#             # Calculate full name if personnel exists
#             full_name = None
#             person_value = log.person
            
#             if person_value and person_value.isdigit() and len(person_value) == 10:
#                 personnel = personnel_lookup.get(person_value)
#                 if personnel:
#                     full_name = f"{personnel.fname} {personnel.lname}".strip()
            
#             logs_data.append({
#                 "id": log.id,
#                 "person": log.person,  # National code
#                 "full_name": full_name,  # Full name if found, else None
#                 "confidence": float(log.confidence) if log.confidence is not None else 0.0,
#                 "detection_time": log.detection_time.isoformat() if log.detection_time else None,
#                 "face_image_base64": face_image_base64,
#                 "video_url": video_url
#             })
        
#         detection_info = {
#             "type": "new_detection",
#             "detection": logs_data
#         }
        
#         await manager.broadcast_json(detection_info)
#         print(f"📊 Broadcast to {manager.count} clients")
#     finally:
#         db.close()


# @app.get("/connections/status")
# async def connection_status():
#     """Get current connection statistics"""
#     return {
#         "active_connections": manager.count,  # Changed from connection_count to count
#         "status": "healthy"
#     }


# @app.on_event("startup")
# async def startup_event():
#     """Initialize on startup"""
#     print("Starting up Face Recognition API...")
    
#     # Start video broadcaster when server starts
#     global processor_task
#     processor_task = asyncio.create_task(video_broadcaster())
#     print("🎬 Video broadcaster started automatically")
    
#     print("Models will be loaded on first WebSocket connection")

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Clean up on shutdown"""
#     global processor_task
#     if processor_task:
#         processor_task.cancel()
#         try:
#             await processor_task
#         except asyncio.CancelledError:
#             pass
#     print("Video broadcaster stopped")






# # Define Pydantic model for request body
# class SaveRequest(BaseModel):
#     id: int
#     name_front: str

# @app.post("/ws-save")
# async def save_detection(
#     request: SaveRequest,
#     db: Session = Depends(get_db)
# ):
#     """Save a detection to the vector database"""
#     print(f'Processing save request for ID: {request.id}, name: {request.name_front}')
    
#     try:
#         # Get detection log
#         detection_log = db.query(DetectionLog).filter(DetectionLog.id == request.id).first()
        
#         if not detection_log:
#             raise HTTPException(status_code=404, detail=f"Detection log with ID {request.id} not found")
        
#         if not detection_log.face_image_path:
#             raise HTTPException(status_code=400, detail="No face image path found in this log entry")
        
#         if not os.path.exists(detection_log.face_image_path):
#             raise HTTPException(status_code=404, detail=f"Face image file not found at: {detection_log.face_image_path}")
        
#         # Read and process image
#         img = cv2.imread(detection_log.face_image_path)
#         if img is None:
#             raise HTTPException(status_code=500, detail=f"Failed to read image file from: {detection_log.face_image_path}")
        
#         # Add to vector database
#         result = vectorDatabase(img, request.name_front)
        
#         # Update the person name
#         detection_log.person = request.name_front
#         db.commit()
        
#         return {
#             "success": True,
#             "message": f"Image added to vector database for {request.name_front}",
#             "log_id": request.id,
#             "corrected_person": request.name_front,
#             "image_shape": img.shape,
#             "detection_time": detection_log.detection_time.isoformat() if detection_log.detection_time else None
#         }
            
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(f"Error processing log ID {request.id}: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))
    

# @app.exception_handler(Exception)
# async def global_exception_handler(request, exc):
#     """Global exception handler"""
#     print(f"Global error: {exc}")
#     return JSONResponse(
#         status_code=500,
#         content={"detail": "An internal server error occurred"}
#     )


# from pathlib import Path
# @app.get("/qs-save")
# async def video_ws1(
#     # request: SaveRequest,
#     # db: Session = Depends(get_db)
# ):
#     IMAGE_ROOT = "./Detection/TensorRT/qdrant_images"


#     root = Path(IMAGE_ROOT)

#     for person_dir in root.iterdir():
#         if not person_dir.is_dir():
#             print('empttyyyyyyyyyyyyyyyyyyy')
#             continue

#         person_name = person_dir.name
#         print(f"\n📁 Person: {person_name}")

#         for img_path in person_dir.glob("*.*"):
#             try:
#                 img = cv2.imread(str(img_path))
#                 if img is None:
#                     print("❌ cannot read", img_path.name)
#                     continue
#                 vectorDatabase(img, f'{person_name}', save_mode=True)


                

#                 print("✅ inserted", img_path.name)

#             except Exception as e:
#                 print("❌ error:", img_path.name, e)
        
#     return {
#             "success": True,
            
#         }
            


# @app.get("/qd-remove")
# async def qd_remove(
#     # request: SaveRequest,
#     # db: Session = Depends(get_db)
# ):
#     # print('id', request.id)
#     # # name_to_ai = dictionary.get(request.name_front)
#     # name_to_ai = request.name_front
#     try:
#         name_to_ai = ['3', 'ygj', 'حسین زاده', 'hossein_zade', 'امیر زمانی','amir_zamani']
#         result = removeDatabase(name_to_ai)
   
#         return {
#             "success": True,
#             "message": f"Image added to vector database for {name_to_ai}",
#             # "log_id": request.id,
#             "corrected_person": name_to_ai,
#             "vector_database_result": result if result else "Success",
#         }
            
#     except Exception as e:
#         # print(f"Error processing log ID {request.id}: {e}")
#         import traceback
#         traceback.print_exc()
#         return {"error": str(e)}
#     finally:
#         print('Processing complete')
    
    
    
    
    
    
    
    
    

































    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # import os
# import base64
# import asyncio
# from typing import Optional, Set, Dict
# from collections import defaultdict, deque, Counter
# from pathlib import Path
# import cv2
# import av
# from datetime import datetime, date, time
# import time  # Add this line

# from fastapi import FastAPI, WebSocket, Depends, HTTPException, WebSocketDisconnect
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from sqlalchemy.orm import Session
# from .models.database import save_detection_with_face, get_db, DetectionLog
# from .api import router as api_router
# from Detection.TensorRT.infer import ws_transfer, vectorDatabase, removeDatabase

# processor_task = None

# app = FastAPI(title="Face Recognition API", version="1.0.0")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# app.include_router(api_router, prefix="/api/v1")
# # app.include_router(router, prefix="/api/v1")

# class Config:
#     RTSP_URL = "rtsp://Jafari:Asd12345@192.168.110.20:554/Streaming/Channels/301"
#     BASE_URL = "http://192.168.10.9:8001"
#     CLIP_LENGTH = 100
#     HALF_CLIP = CLIP_LENGTH // 2
#     VOTE_WINDOW = 1000

# config = Config()


# # At the top of your main.py, add this class
# class WebSocketManager:
#     def __init__(self):
#         self.active_connections: Set[WebSocket] = set()
#         self._lock = asyncio.Lock()  # For thread safety
    
#     async def connect(self, websocket: WebSocket):
#         await websocket.accept()
#         async with self._lock:
#             self.active_connections.add(websocket)
#         print(f"✅ Client connected. Total: {len(self.active_connections)}")
    
#     async def disconnect(self, websocket: WebSocket):  # ✅ Add async here
#         async with self._lock:
#             self.active_connections.discard(websocket)
#         print(f"❌ Client disconnected. Total: {len(self.active_connections)}")
    
#     async def broadcast_json(self, message: dict):
#         """Send JSON to all connected clients"""
#         if not self.active_connections:
#             return
        
#         disconnected = []
#         async with self._lock:
#             connections = list(self.active_connections)
        
#         for connection in connections:
#             try:
#                 await connection.send_json(message)
#             except Exception as e:
#                 print(f"Failed to send to client: {e}")
#                 disconnected.append(connection)
        
#         # Clean up dead connections
#         async with self._lock:
#             for conn in disconnected:
#                 self.active_connections.discard(conn)
    
#     async def broadcast_bytes(self, data: bytes):
#         """Send bytes to all connected clients"""
#         if not self.active_connections:
#             return
        
#         disconnected = []
#         async with self._lock:
#             connections = list(self.active_connections)
        
#         for connection in connections:
#             try:
#                 await connection.send_bytes(data)
#             except Exception:
#                 disconnected.append(connection)
        
#         async with self._lock:
#             for conn in disconnected:
#                 self.active_connections.discard(conn)
    
#     @property
#     def count(self) -> int:
#         return len(self.active_connections)



# # Create global instance
# manager = WebSocketManager()



# class StreamConfig(BaseModel):
#     webcam: bool= False
#     gpu: int = 0
#     threshold: float = 0.5
#     collection: str = "n3"


# @app.get("/")
# async def index():
#     return {
#         "message": "Welcome to Face Recognition API",
#         "description": "Real-time face recognition with FastAPI WebSocket streaming",
#         "endpoints": {
#             "docs": "/docs",
#             "video_page": "/ws-video",
#             "stream_status": "/api/v1/stream/status",
#             "reports": "/reports/daily?from_date=YYYY-MM-DD&to_date=YYYY-MM-DD"
#         }
#     }



# @app.get("/ws-video")
# async def video_page():
#     """Serve the video page HTML"""
#     html_path = os.path.join(os.path.dirname(__file__), "templates", "video_page.html")
    
#     try:
#         with open(html_path, "r", encoding="utf-8") as f:
#             html_content = f.read()
#         return HTMLResponse(content=html_content)
#     except FileNotFoundError:
#         # Fallback in case file is not found
#         return HTMLResponse(content="<h1>Video page template not found</h1>", status_code=404)




# @app.websocket("/ws/video")
# async def video_ws(websocket: WebSocket):
#     await manager.connect(websocket)
#     try:
#         await save_in_database()
#     except Exception as e:
#         print(f"⚠️ Error sending initial logs: {e}")

#     try:
#         while True:
#             try:
#                 data = await asyncio.wait_for(
#                     websocket.receive_text(), 
#                     timeout=30.0
#                 )
#                 # Process messages...
#             except asyncio.TimeoutError:
#                 await websocket.send_json({"type": "ping"})
#                 continue
#     except WebSocketDisconnect:
#         await manager.disconnect(websocket)  # ✅ Add await
#     except Exception as e:
#         print(f"Error: {e}")
#         await manager.disconnect(websocket)  # ✅ Add await




# # Global state
# processor_task: Optional[asyncio.Task] = None
# VOTE_WINDOW = 1000  # how many last frames used

# track_history = defaultdict(lambda: {
#     "names": deque(maxlen=VOTE_WINDOW),
#     "scores": deque(maxlen=VOTE_WINDOW),
#     "faces": deque(maxlen=VOTE_WINDOW),
#     "frames": deque(maxlen=VOTE_WINDOW),
# })






# async def video_broadcaster():
#     """Background task that streams video to all connected clients"""
#     print("🎬 Starting video broadcaster...")
    
#     # Try camera first, fallback to RTSP
#     cap = cv2.VideoCapture(0)
#     use_camera = cap.isOpened()
    
#     if not use_camera:
#         cap.release()
#         print("📹 Camera not available, trying RTSP stream...")
    
#     # Track last time we sent periodic logs
#     last_log_send = time.time()
    
#     while True:
#         try:
#             if not use_camera:
#                 container = av.open(config.RTSP_URL, options={
#                     "rtsp_transport": "tcp", 
#                     "flags": "low_delay", 
#                     "fflags": "nobuffer"
#                 })
#                 print('✅ Connected to RTSP stream')
            
#             frame_count = 0
#             clip_length = 100
#             half_clip = clip_length // 2

#             while True:
#                 # Yield control to event loop
#                 await asyncio.sleep(0.001)
                
#                 if manager.count == 0:
#                     await asyncio.sleep(0.1)
#                     continue
                
#                 # Send logs periodically (every 5 seconds) as backup
#                 current_time = time.time()
#                 if current_time - last_log_send > 5:
#                     last_log_send = current_time
#                     asyncio.create_task(save_in_database())
                
#                 # Get frame from either camera or RTSP
#                 if use_camera:
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
#                 else:
#                     try:
#                         frame = next(container.decode(video=0))
#                         frame = frame.to_ndarray(format="bgr24")
#                     except StopIteration:
#                         break
                
#                 frame_count += 1
                
#                 # Process with TensorRT
#                 try:
#                     data = ws_transfer(frame)
                    
#                     # Process detections and update track history
#                     persons = data['persons']
#                     scores = data['scores']
#                     faces = data['faces']
#                     objs = data['objs']
                    
#                     current_objs = set(objs)
                    
#                     # Update history for current objects
#                     for i, (person, score, obj, face) in enumerate(zip(persons, scores, objs, faces)):
#                         history = track_history[obj]
#                         history["frames"].append(frame.copy())
#                         if person and person != "Unknown":
#                             history["names"].append(person)
#                             history["scores"].append(score)
#                             history["faces"].append(face)
                    
#                     # Handle disappeared objects
#                     disappeared_objs = set(track_history.keys()) - current_objs
                    
#                     for obj in disappeared_objs:
#                         history = track_history[obj]
#                         if history["names"]:
#                             final_name = Counter(history["names"]).most_common(1)[0][0]
                            
#                             valid_indices = [i for i, n in enumerate(history["names"]) if n == final_name]
#                             if valid_indices:
#                                 best_idx = max(valid_indices, key=lambda i: history["scores"][i])
#                                 final_score = history["scores"][best_idx]
#                                 final_face = history["faces"][best_idx]
                                
#                                 start_idx = max(0, best_idx - half_clip)
#                                 end_idx = min(len(history["frames"]), best_idx + half_clip)
#                                 frames_to_save = list(history["frames"])[start_idx:end_idx]
                                
#                                 log_id = save_detection_with_face(
#                                     person=final_name,
#                                     confidence=float(final_score),
#                                     face_image=final_face,
#                                     frames_to_save=frames_to_save
#                                 )
                                
#                                 if log_id is not None:
#                                     # IMMEDIATELY send the new logs when a detection is saved
#                                     asyncio.create_task(save_in_database())
#                                     last_log_send = time.time()  # Reset the periodic timer
                        
#                         del track_history[obj]
                    
#                     # Broadcast video frame to clients
#                     asyncio.create_task(broadcast_frame(data))
                    
#                     # Yield again after processing
#                     await asyncio.sleep(0)
                    
#                 except Exception as e:
#                     print(f"⚠️ Error processing frame: {e}")
#                     continue
                        
#         except Exception as e:
#             print(f"⚠️ Stream connection error: {e}")
#             print("🔄 Reconnecting in 5 seconds...")
#             await asyncio.sleep(5)
#         finally:
#             if not use_camera and 'container' in locals():
#                 container.close()





# # def process_detections(data: dict, frame_count: int, frame):  # ✅ Added frame parameter
# #     """Process detections and update track history"""
# #     persons = data['persons']
# #     scores = data['scores']
# #     faces = data['faces']
# #     objs = data['objs']
    
# #     current_objs = set(objs)
    
# #     # Update history for current objects
# #     for i, (person, score, obj, face) in enumerate(zip(persons, scores, objs, faces)):
# #         history = track_history[obj]
# #         history["frames"].append(frame.copy())  # ✅ Now frame is defined
# #         if person and person != "Unknown":
# #             history["names"].append(person)
# #             history["scores"].append(score)
# #             history["faces"].append(face)
    
# #     # Handle disappeared objects
# #     disappeared_objs = set(track_history.keys()) - current_objs
    
# #     for obj in disappeared_objs:
# #         history = track_history[obj]
# #         if history["names"]:
# #             save_disappeared_object(obj, history)
# #         del track_history[obj]

# def save_disappeared_object(obj, history):
#     """Save a disappeared object to database"""
#     final_name = Counter(history["names"]).most_common(1)[0][0]
    
#     valid_indices = [i for i, n in enumerate(history["names"]) if n == final_name]
#     if valid_indices:
#         best_idx = max(valid_indices, key=lambda i: history["scores"][i])
#         final_score = history["scores"][best_idx]
#         final_face = history["faces"][best_idx]
        
#         start_idx = max(0, best_idx - config.HALF_CLIP)
#         end_idx = min(len(history["frames"]), best_idx + config.HALF_CLIP)
#         frames_to_save = list(history["frames"])[start_idx:end_idx]
        
#         save_detection_with_face(
#             person=final_name,
#             confidence=float(final_score),
#             face_image=final_face,
#             frames_to_save=frames_to_save
#         )

# async def broadcast_frame(data: dict):
#     """Broadcast frame and metadata using the manager"""
#     frame_bytes = data['frame']
#     metadata = {
#         "type": "video_metadata",
#         "persons": data['persons'],
#         "scores": [float(s) if s is not None else 0.0 for s in data['scores']],
#         "objs": data['objs'],
#     }
    
#     # Send to all clients - manager handles dead connections
#     if manager.count > 0:
#         # Don't await these - let them run in background
#         asyncio.create_task(manager.broadcast_bytes(frame_bytes))
#         asyncio.create_task(manager.broadcast_json(metadata))


# async def save_in_database():
#     """Broadcast new detection to all connected clients"""
#     offset = 0
#     limit = 20
#     db = next(get_db())
#     try:  # ✅ Added try-finally
#         logs = db.query(DetectionLog)\
#             .order_by(DetectionLog.detection_time.desc())\
#             .offset(offset)\
#             .limit(limit)\
#             .all()
        
#         logs_data = []
#         base_url = config.BASE_URL
        
#         for log in logs:
#             face_image_base64 = None
#             if log.face_image_path and os.path.exists(log.face_image_path):
#                 try:
#                     img = cv2.imread(log.face_image_path)
#                     if img is not None:
#                         #img_rgb = img#cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ####
#                         _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
#                         face_image_base64 = base64.b64encode(buffer).decode('utf-8')
#                 except:
#                     pass
            
#             video_url = None
#             if log.video_path and os.path.exists(log.video_path):
#                 video_url = f"{base_url}/api/v1/detections/{log.id}/video"
            
#             logs_data.append({
#                 "id": log.id,
#                 "person": log.person,
#                 "confidence": float(log.confidence) if log.confidence is not None else 0.0,
#                 "detection_time": log.detection_time.isoformat() if log.detection_time else None,
#                 "face_image_path": log.face_image_path,
#                 "face_image_base64": face_image_base64,
#                 "video_url": video_url
#             })
        
#         detection_info = {
#             "type": "new_detection",
#             "detection": logs_data
#         }
        
#         await manager.broadcast_json(detection_info)
#         print(f"📊 Broadcast to {manager.count} clients")  # ✅ Using count property
#     finally:
#         db.close()  # ✅ Close the session
      


# @app.get("/connections/status")
# async def connection_status():
#     """Get current connection statistics"""
#     return {
#         "active_connections": manager.count,  # Changed from connection_count to count
#         "status": "healthy"
#     }


# @app.on_event("startup")
# async def startup_event():
#     """Initialize on startup"""
#     print("Starting up Face Recognition API...")
    
#     # Start video broadcaster when server starts
#     global processor_task
#     processor_task = asyncio.create_task(video_broadcaster())
#     print("🎬 Video broadcaster started automatically")
    
#     print("Models will be loaded on first WebSocket connection")

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Clean up on shutdown"""
#     global processor_task
#     if processor_task:
#         processor_task.cancel()
#         try:
#             await processor_task
#         except asyncio.CancelledError:
#             pass
#     print("Video broadcaster stopped")






# # Define Pydantic model for request body
# class SaveRequest(BaseModel):
#     id: int
#     name_front: str

# @app.post("/ws-save")
# async def save_detection(
#     request: SaveRequest,
#     db: Session = Depends(get_db)
# ):
#     """Save a detection to the vector database"""
#     print(f'Processing save request for ID: {request.id}, name: {request.name_front}')
    
#     try:
#         # Get detection log
#         detection_log = db.query(DetectionLog).filter(DetectionLog.id == request.id).first()
        
#         if not detection_log:
#             raise HTTPException(status_code=404, detail=f"Detection log with ID {request.id} not found")
        
#         if not detection_log.face_image_path:
#             raise HTTPException(status_code=400, detail="No face image path found in this log entry")
        
#         if not os.path.exists(detection_log.face_image_path):
#             raise HTTPException(status_code=404, detail=f"Face image file not found at: {detection_log.face_image_path}")
        
#         # Read and process image
#         img = cv2.imread(detection_log.face_image_path)
#         if img is None:
#             raise HTTPException(status_code=500, detail=f"Failed to read image file from: {detection_log.face_image_path}")
        
#         # Add to vector database
#         result = vectorDatabase(img, request.name_front)
        
#         # Update the person name
#         detection_log.person = request.name_front
#         db.commit()
        
#         return {
#             "success": True,
#             "message": f"Image added to vector database for {request.name_front}",
#             "log_id": request.id,
#             "corrected_person": request.name_front,
#             "image_shape": img.shape,
#             "detection_time": detection_log.detection_time.isoformat() if detection_log.detection_time else None
#         }
            
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(f"Error processing log ID {request.id}: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))
    

# @app.exception_handler(Exception)
# async def global_exception_handler(request, exc):
#     """Global exception handler"""
#     print(f"Global error: {exc}")
#     return JSONResponse(
#         status_code=500,
#         content={"detail": "An internal server error occurred"}
#     )


# from pathlib import Path
# @app.get("/qs-save")
# async def video_ws1(
#     # request: SaveRequest,
#     # db: Session = Depends(get_db)
# ):
#     IMAGE_ROOT = "./Detection/TensorRT/qdrant_images"


#     root = Path(IMAGE_ROOT)

#     for person_dir in root.iterdir():
#         if not person_dir.is_dir():
#             print('empttyyyyyyyyyyyyyyyyyyy')
#             continue

#         person_name = person_dir.name
#         print(f"\n📁 Person: {person_name}")

#         for img_path in person_dir.glob("*.*"):
#             try:
#                 img = cv2.imread(str(img_path))
#                 if img is None:
#                     print("❌ cannot read", img_path.name)
#                     continue
#                 vectorDatabase(img, f'{person_name}', save_mode=True)


                

#                 print("✅ inserted", img_path.name)

#             except Exception as e:
#                 print("❌ error:", img_path.name, e)
        
#     return {
#             "success": True,
            
#         }
            


# @app.get("/qd-remove")
# async def qd_remove(
#     # request: SaveRequest,
#     # db: Session = Depends(get_db)
# ):
#     # print('id', request.id)
#     # # name_to_ai = dictionary.get(request.name_front)
#     # name_to_ai = request.name_front
#     try:
#         name_to_ai = ['3', 'ygj', 'حسین زاده', 'hossein_zade', 'امیر زمانی','amir_zamani']
#         result = removeDatabase(name_to_ai)
   
#         return {
#             "success": True,
#             "message": f"Image added to vector database for {name_to_ai}",
#             # "log_id": request.id,
#             "corrected_person": name_to_ai,
#             "vector_database_result": result if result else "Success",
#         }
            
#     except Exception as e:
#         # print(f"Error processing log ID {request.id}: {e}")
#         import traceback
#         traceback.print_exc()
#         return {"error": str(e)}
#     finally:
#         print('Processing complete')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # import os
# import base64
# import asyncio
# from typing import Optional, Set, Dict
# from collections import defaultdict, deque, Counter
# from pathlib import Path
# import cv2
# import av
# from datetime import datetime, date, time
# import time  # Add this line

# from fastapi import FastAPI, WebSocket, Depends, HTTPException, WebSocketDisconnect
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from sqlalchemy.orm import Session
# from .models.database import save_detection_with_face, get_db, DetectionLog
# from .api import router as api_router
# from Detection.TensorRT.infer import ws_transfer, vectorDatabase, removeDatabase

# processor_task = None

# app = FastAPI(title="Face Recognition API", version="1.0.0")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# app.include_router(api_router, prefix="/api/v1")
# # app.include_router(router, prefix="/api/v1")

# class Config:
#     RTSP_URL = "rtsp://Jafari:Asd12345@192.168.110.20:554/Streaming/Channels/301"
#     BASE_URL = "http://192.168.10.9:8001"
#     CLIP_LENGTH = 100
#     HALF_CLIP = CLIP_LENGTH // 2
#     VOTE_WINDOW = 1000

# config = Config()


# # At the top of your main.py, add this class
# class WebSocketManager:
#     def __init__(self):
#         self.active_connections: Set[WebSocket] = set()
#         self._lock = asyncio.Lock()  # For thread safety
    
#     async def connect(self, websocket: WebSocket):
#         await websocket.accept()
#         async with self._lock:
#             self.active_connections.add(websocket)
#         print(f"✅ Client connected. Total: {len(self.active_connections)}")
    
#     async def disconnect(self, websocket: WebSocket):  # ✅ Add async here
#         async with self._lock:
#             self.active_connections.discard(websocket)
#         print(f"❌ Client disconnected. Total: {len(self.active_connections)}")
    
#     async def broadcast_json(self, message: dict):
#         """Send JSON to all connected clients"""
#         if not self.active_connections:
#             return
        
#         disconnected = []
#         async with self._lock:
#             connections = list(self.active_connections)
        
#         for connection in connections:
#             try:
#                 await connection.send_json(message)
#             except Exception as e:
#                 print(f"Failed to send to client: {e}")
#                 disconnected.append(connection)
        
#         # Clean up dead connections
#         async with self._lock:
#             for conn in disconnected:
#                 self.active_connections.discard(conn)
    
#     async def broadcast_bytes(self, data: bytes):
#         """Send bytes to all connected clients"""
#         if not self.active_connections:
#             return
        
#         disconnected = []
#         async with self._lock:
#             connections = list(self.active_connections)
        
#         for connection in connections:
#             try:
#                 await connection.send_bytes(data)
#             except Exception:
#                 disconnected.append(connection)
        
#         async with self._lock:
#             for conn in disconnected:
#                 self.active_connections.discard(conn)
    
#     @property
#     def count(self) -> int:
#         return len(self.active_connections)



# # Create global instance
# manager = WebSocketManager()



# class StreamConfig(BaseModel):
#     webcam: bool= False
#     gpu: int = 0
#     threshold: float = 0.5
#     collection: str = "n3"


# @app.get("/")
# async def index():
#     return {
#         "message": "Welcome to Face Recognition API",
#         "description": "Real-time face recognition with FastAPI WebSocket streaming",
#         "endpoints": {
#             "docs": "/docs",
#             "video_page": "/ws-video",
#             "stream_status": "/api/v1/stream/status",
#             "reports": "/reports/daily?from_date=YYYY-MM-DD&to_date=YYYY-MM-DD"
#         }
#     }



# @app.get("/ws-video")
# async def video_page():
#     """Serve the video page HTML"""
#     html_path = os.path.join(os.path.dirname(__file__), "templates", "video_page.html")
    
#     try:
#         with open(html_path, "r", encoding="utf-8") as f:
#             html_content = f.read()
#         return HTMLResponse(content=html_content)
#     except FileNotFoundError:
#         # Fallback in case file is not found
#         return HTMLResponse(content="<h1>Video page template not found</h1>", status_code=404)




# @app.websocket("/ws/video")
# async def video_ws(websocket: WebSocket):
#     await manager.connect(websocket)
#     try:
#         await save_in_database()
#     except Exception as e:
#         print(f"⚠️ Error sending initial logs: {e}")

#     try:
#         while True:
#             try:
#                 data = await asyncio.wait_for(
#                     websocket.receive_text(), 
#                     timeout=30.0
#                 )
#                 # Process messages...
#             except asyncio.TimeoutError:
#                 await websocket.send_json({"type": "ping"})
#                 continue
#     except WebSocketDisconnect:
#         await manager.disconnect(websocket)  # ✅ Add await
#     except Exception as e:
#         print(f"Error: {e}")
#         await manager.disconnect(websocket)  # ✅ Add await




# # Global state
# processor_task: Optional[asyncio.Task] = None
# VOTE_WINDOW = 1000  # how many last frames used

# track_history = defaultdict(lambda: {
#     "names": deque(maxlen=VOTE_WINDOW),
#     "scores": deque(maxlen=VOTE_WINDOW),
#     "faces": deque(maxlen=VOTE_WINDOW),
#     "frames": deque(maxlen=VOTE_WINDOW),
# })






# async def video_broadcaster():
#     """Background task that streams video to all connected clients"""
#     print("🎬 Starting video broadcaster...")
    
#     # Try camera first, fallback to RTSP
#     cap = cv2.VideoCapture(0)
#     use_camera = cap.isOpened()
    
#     if not use_camera:
#         cap.release()
#         print("📹 Camera not available, trying RTSP stream...")
    
#     # Track last time we sent logs
#     last_log_send = time.time()
    
#     while True:
#         try:
#             if not use_camera:
#                 container = av.open(config.RTSP_URL, options={
#                     "rtsp_transport": "tcp", 
#                     "flags": "low_delay", 
#                     "fflags": "nobuffer"
#                 })
#                 print('✅ Connected to RTSP stream')
            
#             frame_count = 0
#             clip_length = 100
#             half_clip = clip_length // 2
#             save_database = False  # ADD THIS FLAG

#             while True:
#                 # Yield control to event loop
#                 await asyncio.sleep(0.001)
                
#                 if manager.count == 0:
#                     await asyncio.sleep(0.1)
#                     continue
                
#                 # Send logs periodically (every 5 seconds) even without new detections
#                 current_time = time.time()
#                 if current_time - last_log_send > 5:
#                     last_log_send = current_time
#                     asyncio.create_task(save_in_database())
                
#                 # Get frame from either camera or RTSP
#                 if use_camera:
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
#                 else:
#                     try:
#                         frame = next(container.decode(video=0))
#                         frame = frame.to_ndarray(format="bgr24")
#                     except StopIteration:
#                         break
                
#                 frame_count += 1
                
#                 # Process with TensorRT
#                 try:
#                     data = ws_transfer(frame)
                    
#                     # Process detections and update track history
#                     persons = data['persons']
#                     scores = data['scores']
#                     faces = data['faces']
#                     objs = data['objs']
                    
#                     current_objs = set(objs)
                    
#                     # Update history for current objects
#                     for i, (person, score, obj, face) in enumerate(zip(persons, scores, objs, faces)):
#                         history = track_history[obj]
#                         history["frames"].append(frame.copy())
#                         if person and person != "Unknown":
#                             history["names"].append(person)
#                             history["scores"].append(score)
#                             history["faces"].append(face)
                    
#                     # Handle disappeared objects
#                     disappeared_objs = set(track_history.keys()) - current_objs
                    
#                     for obj in disappeared_objs:
#                         history = track_history[obj]
#                         if history["names"]:
#                             final_name = Counter(history["names"]).most_common(1)[0][0]
                            
#                             valid_indices = [i for i, n in enumerate(history["names"]) if n == final_name]
#                             if valid_indices:
#                                 best_idx = max(valid_indices, key=lambda i: history["scores"][i])
#                                 final_score = history["scores"][best_idx]
#                                 final_face = history["faces"][best_idx]
                                
#                                 start_idx = max(0, best_idx - half_clip)
#                                 end_idx = min(len(history["frames"]), best_idx + half_clip)
#                                 frames_to_save = list(history["frames"])[start_idx:end_idx]
                                
#                                 log_id = save_detection_with_face(
#                                     person=final_name,
#                                     confidence=float(final_score),
#                                     face_image=final_face,
#                                     frames_to_save=frames_to_save
#                                 )
                                
#                                 if log_id is not None:
#                                     save_database = True  # SET FLAG WHEN DETECTION IS SAVED
                        
#                         del track_history[obj]
                    
#                     # Trigger database save if we have new detections
#                     if save_database:
#                         save_database = False
#                         last_log_send = current_time  # Reset timer
#                         asyncio.create_task(save_in_database())
                    
#                     # Broadcast video frame to clients
#                     asyncio.create_task(broadcast_frame(data))
                    
#                     # Yield again after processing
#                     await asyncio.sleep(0)
                    
#                 except Exception as e:
#                     print(f"⚠️ Error processing frame: {e}")
#                     continue
                        
#         except Exception as e:
#             print(f"⚠️ Stream connection error: {e}")
#             print("🔄 Reconnecting in 5 seconds...")
#             await asyncio.sleep(5)
#         finally:
#             if not use_camera and 'container' in locals():
#                 container.close()





# # def process_detections(data: dict, frame_count: int, frame):  # ✅ Added frame parameter
# #     """Process detections and update track history"""
# #     persons = data['persons']
# #     scores = data['scores']
# #     faces = data['faces']
# #     objs = data['objs']
    
# #     current_objs = set(objs)
    
# #     # Update history for current objects
# #     for i, (person, score, obj, face) in enumerate(zip(persons, scores, objs, faces)):
# #         history = track_history[obj]
# #         history["frames"].append(frame.copy())  # ✅ Now frame is defined
# #         if person and person != "Unknown":
# #             history["names"].append(person)
# #             history["scores"].append(score)
# #             history["faces"].append(face)
    
# #     # Handle disappeared objects
# #     disappeared_objs = set(track_history.keys()) - current_objs
    
# #     for obj in disappeared_objs:
# #         history = track_history[obj]
# #         if history["names"]:
# #             save_disappeared_object(obj, history)
# #         del track_history[obj]

# def save_disappeared_object(obj, history):
#     """Save a disappeared object to database"""
#     final_name = Counter(history["names"]).most_common(1)[0][0]
    
#     valid_indices = [i for i, n in enumerate(history["names"]) if n == final_name]
#     if valid_indices:
#         best_idx = max(valid_indices, key=lambda i: history["scores"][i])
#         final_score = history["scores"][best_idx]
#         final_face = history["faces"][best_idx]
        
#         start_idx = max(0, best_idx - config.HALF_CLIP)
#         end_idx = min(len(history["frames"]), best_idx + config.HALF_CLIP)
#         frames_to_save = list(history["frames"])[start_idx:end_idx]
        
#         save_detection_with_face(
#             person=final_name,
#             confidence=float(final_score),
#             face_image=final_face,
#             frames_to_save=frames_to_save
#         )

# async def broadcast_frame(data: dict):
#     """Broadcast frame and metadata using the manager"""
#     frame_bytes = data['frame']
#     metadata = {
#         "type": "video_metadata",
#         "persons": data['persons'],
#         "scores": [float(s) if s is not None else 0.0 for s in data['scores']],
#         "objs": data['objs'],
#     }
    
#     # Send to all clients - manager handles dead connections
#     if manager.count > 0:
#         # Don't await these - let them run in background
#         asyncio.create_task(manager.broadcast_bytes(frame_bytes))
#         asyncio.create_task(manager.broadcast_json(metadata))


# async def save_in_database():
#     """Broadcast new detection to all connected clients"""
#     offset = 0
#     limit = 20
#     db = next(get_db())
#     try:  # ✅ Added try-finally
#         logs = db.query(DetectionLog)\
#             .order_by(DetectionLog.detection_time.desc())\
#             .offset(offset)\
#             .limit(limit)\
#             .all()
        
#         logs_data = []
#         base_url = config.BASE_URL
        
#         for log in logs:
#             face_image_base64 = None
#             if log.face_image_path and os.path.exists(log.face_image_path):
#                 try:
#                     img = cv2.imread(log.face_image_path)
#                     if img is not None:
#                         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ####
#                         _, buffer = cv2.imencode('.jpg', img_rgb, [cv2.IMWRITE_JPEG_QUALITY, 70])
#                         face_image_base64 = base64.b64encode(buffer).decode('utf-8')
#                 except:
#                     pass
            
#             video_url = None
#             if log.video_path and os.path.exists(log.video_path):
#                 video_url = f"{base_url}/api/v1/detections/{log.id}/video"
            
#             logs_data.append({
#                 "id": log.id,
#                 "person": log.person,
#                 "confidence": float(log.confidence) if log.confidence is not None else 0.0,
#                 "detection_time": log.detection_time.isoformat() if log.detection_time else None,
#                 "face_image_path": log.face_image_path,
#                 "face_image_base64": face_image_base64,
#                 "video_url": video_url
#             })
        
#         detection_info = {
#             "type": "new_detection",
#             "detection": logs_data
#         }
        
#         await manager.broadcast_json(detection_info)
#         print(f"📊 Broadcast to {manager.count} clients")  # ✅ Using count property
#     finally:
#         db.close()  # ✅ Close the session
      


# @app.get("/connections/status")
# async def connection_status():
#     """Get current connection statistics"""
#     return {
#         "active_connections": manager.count,  # Changed from connection_count to count
#         "status": "healthy"
#     }


# @app.on_event("startup")
# async def startup_event():
#     """Initialize on startup"""
#     print("Starting up Face Recognition API...")
    
#     # Start video broadcaster when server starts
#     global processor_task
#     processor_task = asyncio.create_task(video_broadcaster())
#     print("🎬 Video broadcaster started automatically")
    
#     print("Models will be loaded on first WebSocket connection")

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Clean up on shutdown"""
#     global processor_task
#     if processor_task:
#         processor_task.cancel()
#         try:
#             await processor_task
#         except asyncio.CancelledError:
#             pass
#     print("Video broadcaster stopped")






# # Define Pydantic model for request body
# class SaveRequest(BaseModel):
#     id: int
#     name_front: str

# @app.post("/ws-save")
# async def save_detection(
#     request: SaveRequest,
#     db: Session = Depends(get_db)
# ):
#     """Save a detection to the vector database"""
#     print(f'Processing save request for ID: {request.id}, name: {request.name_front}')
    
#     try:
#         # Get detection log
#         detection_log = db.query(DetectionLog).filter(DetectionLog.id == request.id).first()
        
#         if not detection_log:
#             raise HTTPException(status_code=404, detail=f"Detection log with ID {request.id} not found")
        
#         if not detection_log.face_image_path:
#             raise HTTPException(status_code=400, detail="No face image path found in this log entry")
        
#         if not os.path.exists(detection_log.face_image_path):
#             raise HTTPException(status_code=404, detail=f"Face image file not found at: {detection_log.face_image_path}")
        
#         # Read and process image
#         img = cv2.imread(detection_log.face_image_path)
#         if img is None:
#             raise HTTPException(status_code=500, detail=f"Failed to read image file from: {detection_log.face_image_path}")
        
#         # Add to vector database
#         result = vectorDatabase(img, request.name_front)
        
#         # Update the person name
#         detection_log.person = request.name_front
#         db.commit()
        
#         return {
#             "success": True,
#             "message": f"Image added to vector database for {request.name_front}",
#             "log_id": request.id,
#             "corrected_person": request.name_front,
#             "image_shape": img.shape,
#             "detection_time": detection_log.detection_time.isoformat() if detection_log.detection_time else None
#         }
            
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(f"Error processing log ID {request.id}: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))
    

# @app.exception_handler(Exception)
# async def global_exception_handler(request, exc):
#     """Global exception handler"""
#     print(f"Global error: {exc}")
#     return JSONResponse(
#         status_code=500,
#         content={"detail": "An internal server error occurred"}
#     )


# from pathlib import Path
# @app.get("/qs-save")
# async def video_ws1(
#     # request: SaveRequest,
#     # db: Session = Depends(get_db)
# ):
#     IMAGE_ROOT = "./Detection/TensorRT/qdrant_images"


#     root = Path(IMAGE_ROOT)

#     for person_dir in root.iterdir():
#         if not person_dir.is_dir():
#             print('empttyyyyyyyyyyyyyyyyyyy')
#             continue

#         person_name = person_dir.name
#         print(f"\n📁 Person: {person_name}")

#         for img_path in person_dir.glob("*.*"):
#             try:
#                 img = cv2.imread(str(img_path))
#                 if img is None:
#                     print("❌ cannot read", img_path.name)
#                     continue
#                 vectorDatabase(img, f'{person_name}', save_mode=True)


                

#                 print("✅ inserted", img_path.name)

#             except Exception as e:
#                 print("❌ error:", img_path.name, e)
        
#     return {
#             "success": True,
            
#         }
            


# @app.get("/qd-remove")
# async def qd_remove(
#     # request: SaveRequest,
#     # db: Session = Depends(get_db)
# ):
#     # print('id', request.id)
#     # # name_to_ai = dictionary.get(request.name_front)
#     # name_to_ai = request.name_front
#     try:
#         name_to_ai = ['3', 'ygj', 'حسین زاده', 'hossein_zade', 'امیر زمانی','amir_zamani']
#         result = removeDatabase(name_to_ai)
   
#         return {
#             "success": True,
#             "message": f"Image added to vector database for {name_to_ai}",
#             # "log_id": request.id,
#             "corrected_person": name_to_ai,
#             "vector_database_result": result if result else "Success",
#         }
            
#     except Exception as e:
#         # print(f"Error processing log ID {request.id}: {e}")
#         import traceback
#         traceback.print_exc()
#         return {"error": str(e)}
#     finally:
#         print('Processing complete')
    
    
    
    
    
    
    












































































# import os
# import base64
# import asyncio
# from typing import Optional, Set, Dict
# from collections import defaultdict, deque, Counter
# from pathlib import Path
# import cv2
# import av
# from datetime import datetime, date, time

# from fastapi import FastAPI, WebSocket, Depends, HTTPException, WebSocketDisconnect
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from sqlalchemy.orm import Session
# from .models.database import save_detection_with_face, get_db, DetectionLog
# from .api import router as api_router
# from Detection.TensorRT.infer import ws_transfer, vectorDatabase, removeDatabase

# processor_task = None

# app = FastAPI(title="Face Recognition API", version="1.0.0")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# app.include_router(api_router, prefix="/api/v1")
# # app.include_router(router, prefix="/api/v1")

# class Config:
#     RTSP_URL = "rtsp://Jafari:Asd12345@192.168.110.20:554/Streaming/Channels/301"
#     BASE_URL = "http://192.168.10.9:8001"
#     CLIP_LENGTH = 100
#     HALF_CLIP = CLIP_LENGTH // 2
#     VOTE_WINDOW = 1000

# config = Config()


# # At the top of your main.py, add this class
# class WebSocketManager:
#     def __init__(self):
#         self.active_connections: Set[WebSocket] = set()
#         self._lock = asyncio.Lock()  # For thread safety
    
#     async def connect(self, websocket: WebSocket):
#         await websocket.accept()
#         async with self._lock:
#             self.active_connections.add(websocket)
#         print(f"✅ Client connected. Total: {len(self.active_connections)}")
    
#     async def disconnect(self, websocket: WebSocket):  # ✅ Add async here
#         async with self._lock:
#             self.active_connections.discard(websocket)
#         print(f"❌ Client disconnected. Total: {len(self.active_connections)}")
    
#     async def broadcast_json(self, message: dict):
#         """Send JSON to all connected clients"""
#         if not self.active_connections:
#             return
        
#         disconnected = []
#         async with self._lock:
#             connections = list(self.active_connections)
        
#         for connection in connections:
#             try:
#                 await connection.send_json(message)
#             except Exception as e:
#                 print(f"Failed to send to client: {e}")
#                 disconnected.append(connection)
        
#         # Clean up dead connections
#         async with self._lock:
#             for conn in disconnected:
#                 self.active_connections.discard(conn)
    
#     async def broadcast_bytes(self, data: bytes):
#         """Send bytes to all connected clients"""
#         if not self.active_connections:
#             return
        
#         disconnected = []
#         async with self._lock:
#             connections = list(self.active_connections)
        
#         for connection in connections:
#             try:
#                 await connection.send_bytes(data)
#             except Exception:
#                 disconnected.append(connection)
        
#         async with self._lock:
#             for conn in disconnected:
#                 self.active_connections.discard(conn)
    
#     @property
#     def count(self) -> int:
#         return len(self.active_connections)



# # Create global instance
# manager = WebSocketManager()



# class StreamConfig(BaseModel):
#     webcam: bool= False
#     gpu: int = 0
#     threshold: float = 0.5
#     collection: str = "n3"


# @app.get("/")
# async def index():
#     return {
#         "message": "Welcome to Face Recognition API",
#         "description": "Real-time face recognition with FastAPI WebSocket streaming",
#         "endpoints": {
#             "docs": "/docs",
#             "video_page": "/ws-video",
#             "stream_status": "/api/v1/stream/status",
#             "reports": "/reports/daily?from_date=YYYY-MM-DD&to_date=YYYY-MM-DD"
#         }
#     }



# @app.get("/ws-video")
# async def video_page():
#     """Serve the video page HTML"""
#     html_path = os.path.join(os.path.dirname(__file__), "templates", "video_page.html")
    
#     try:
#         with open(html_path, "r", encoding="utf-8") as f:
#             html_content = f.read()
#         return HTMLResponse(content=html_content)
#     except FileNotFoundError:
#         # Fallback in case file is not found
#         return HTMLResponse(content="<h1>Video page template not found</h1>", status_code=404)




# @app.websocket("/ws/video")
# async def video_ws(websocket: WebSocket):
#     await manager.connect(websocket)
#     try:
#         while True:
#             try:
#                 data = await asyncio.wait_for(
#                     websocket.receive_text(), 
#                     timeout=30.0
#                 )
#                 # Process messages...
#             except asyncio.TimeoutError:
#                 await websocket.send_json({"type": "ping"})
#                 continue
#     except WebSocketDisconnect:
#         await manager.disconnect(websocket)  # ✅ Add await
#     except Exception as e:
#         print(f"Error: {e}")
#         await manager.disconnect(websocket)  # ✅ Add await




# # Global state
# processor_task: Optional[asyncio.Task] = None
# VOTE_WINDOW = 1000  # how many last frames used

# track_history = defaultdict(lambda: {
#     "names": deque(maxlen=VOTE_WINDOW),
#     "scores": deque(maxlen=VOTE_WINDOW),
#     "faces": deque(maxlen=VOTE_WINDOW),
#     "frames": deque(maxlen=VOTE_WINDOW),
# })






# async def video_broadcaster():
#     """Background task that streams video to all connected clients"""
#     print("🎬 Starting video broadcaster...")
    
#     # Try camera first, fallback to RTSP
#     cap = cv2.VideoCapture(0)
#     use_camera = cap.isOpened()
    
#     if not use_camera:
#         cap.release()
#         print("📹 Camera not available, trying RTSP stream...")
    
#     while True:
#         try:
#             if not use_camera:
#                 container = av.open(config.RTSP_URL, options={
#                     "rtsp_transport": "tcp", 
#                     "flags": "low_delay", 
#                     "fflags": "nobuffer"
#                 })
#                 print('✅ Connected to RTSP stream')
            
#             frame_count = 0

#             while True:
#                 # ✅ FIXED: Use manager.count instead of old clients set
#                 await asyncio.sleep(0.001)  # Small sleep to let other tasks run
#                 if manager.count == 0:
#                     await asyncio.sleep(0.1)
#                     continue
                
#                 # Get frame from either camera or RTSP
#                 if use_camera:
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
#                 else:
#                     try:
#                         frame = next(container.decode(video=0))
#                         frame = frame.to_ndarray(format="bgr24")
#                     except StopIteration:
#                         break
                
#                 frame_count += 1
                
#                 # Process with TensorRT
#                 try:
#                     data = ws_transfer(frame)
#                     # ✅ FIXED: Pass frame to process_detections
#                     process_detections(data, frame_count, frame)
#                     asyncio.create_task(broadcast_frame(data))
#                     # Broadcast to clients
#                     await asyncio.sleep(0)
                    
#                 except Exception as e:
#                     print(f"⚠️ Error processing frame: {e}")
#                     continue
                        
#         except Exception as e:
#             print(f"⚠️ Stream connection error: {e}")
#             print("🔄 Reconnecting in 5 seconds...")
#             await asyncio.sleep(5)
#         finally:
#             if not use_camera and 'container' in locals():
#                 container.close()



# def process_detections(data: dict, frame_count: int, frame):  # ✅ Added frame parameter
#     """Process detections and update track history"""
#     persons = data['persons']
#     scores = data['scores']
#     faces = data['faces']
#     objs = data['objs']
    
#     current_objs = set(objs)
    
#     # Update history for current objects
#     for i, (person, score, obj, face) in enumerate(zip(persons, scores, objs, faces)):
#         history = track_history[obj]
#         history["frames"].append(frame.copy())  # ✅ Now frame is defined
#         if person and person != "Unknown":
#             history["names"].append(person)
#             history["scores"].append(score)
#             history["faces"].append(face)
    
#     # Handle disappeared objects
#     disappeared_objs = set(track_history.keys()) - current_objs
    
#     for obj in disappeared_objs:
#         history = track_history[obj]
#         if history["names"]:
#             save_disappeared_object(obj, history)
#         del track_history[obj]

# def save_disappeared_object(obj, history):
#     """Save a disappeared object to database"""
#     final_name = Counter(history["names"]).most_common(1)[0][0]
    
#     valid_indices = [i for i, n in enumerate(history["names"]) if n == final_name]
#     if valid_indices:
#         best_idx = max(valid_indices, key=lambda i: history["scores"][i])
#         final_score = history["scores"][best_idx]
#         final_face = history["faces"][best_idx]
        
#         start_idx = max(0, best_idx - config.HALF_CLIP)
#         end_idx = min(len(history["frames"]), best_idx + config.HALF_CLIP)
#         frames_to_save = list(history["frames"])[start_idx:end_idx]
        
#         save_detection_with_face(
#             person=final_name,
#             confidence=float(final_score),
#             face_image=final_face,
#             frames_to_save=frames_to_save
#         )

# async def broadcast_frame(data: dict):
#     """Broadcast frame and metadata using the manager"""
#     frame_bytes = data['frame']
#     metadata = {
#         "type": "video_metadata",
#         "persons": data['persons'],
#         "scores": [float(s) if s is not None else 0.0 for s in data['scores']],
#         "objs": data['objs'],
#     }
    
#     # Send to all clients - manager handles dead connections
#     if manager.count > 0:
#         # Don't await these - let them run in background
#         asyncio.create_task(manager.broadcast_bytes(frame_bytes))
#         asyncio.create_task(manager.broadcast_json(metadata))


# async def save_in_database():
#     """Broadcast new detection to all connected clients"""
#     offset = 0
#     limit = 20
#     db = next(get_db())
#     try:  # ✅ Added try-finally
#         logs = db.query(DetectionLog)\
#             .order_by(DetectionLog.detection_time.desc())\
#             .offset(offset)\
#             .limit(limit)\
#             .all()
        
#         logs_data = []
#         base_url = "http://192.168.10.9:8001"
        
#         for log in logs:
#             face_image_base64 = None
#             if log.face_image_path and os.path.exists(log.face_image_path):
#                 try:
#                     img = cv2.imread(log.face_image_path)
#                     if img is not None:
#                         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                         _, buffer = cv2.imencode('.jpg', img_rgb, [cv2.IMWRITE_JPEG_QUALITY, 70])
#                         face_image_base64 = base64.b64encode(buffer).decode('utf-8')
#                 except:
#                     pass
            
#             video_url = None
#             if log.video_path and os.path.exists(log.video_path):
#                 video_url = f"{base_url}/api/v1/detections/{log.id}/video"
            
#             logs_data.append({
#                 "id": log.id,
#                 "person": log.person,
#                 "confidence": float(log.confidence) if log.confidence is not None else 0.0,
#                 "detection_time": log.detection_time.isoformat() if log.detection_time else None,
#                 "face_image_path": log.face_image_path,
#                 "face_image_base64": face_image_base64,
#                 "video_url": video_url
#             })
        
#         detection_info = {
#             "type": "new_detection",
#             "detection": logs_data
#         }
        
#         await manager.broadcast_json(detection_info)
#         print(f"📊 Broadcast to {manager.count} clients")  # ✅ Using count property
#     finally:
#         db.close()  # ✅ Close the session
      


# @app.get("/connections/status")
# async def connection_status():
#     """Get current connection statistics"""
#     return {
#         "active_connections": manager.count,  # Changed from connection_count to count
#         "status": "healthy"
#     }


# @app.on_event("startup")
# async def startup_event():
#     """Initialize on startup"""
#     print("Starting up Face Recognition API...")
    
#     # Start video broadcaster when server starts
#     global processor_task
#     processor_task = asyncio.create_task(video_broadcaster())
#     print("🎬 Video broadcaster started automatically")
    
#     print("Models will be loaded on first WebSocket connection")

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Clean up on shutdown"""
#     global processor_task
#     if processor_task:
#         processor_task.cancel()
#         try:
#             await processor_task
#         except asyncio.CancelledError:
#             pass
#     print("Video broadcaster stopped")






# # Define Pydantic model for request body
# class SaveRequest(BaseModel):
#     id: int
#     name_front: str

# @app.post("/ws-save")
# async def save_detection(
#     request: SaveRequest,
#     db: Session = Depends(get_db)
# ):
#     """Save a detection to the vector database"""
#     print(f'Processing save request for ID: {request.id}, name: {request.name_front}')
    
#     try:
#         # Get detection log
#         detection_log = db.query(DetectionLog).filter(DetectionLog.id == request.id).first()
        
#         if not detection_log:
#             raise HTTPException(status_code=404, detail=f"Detection log with ID {request.id} not found")
        
#         if not detection_log.face_image_path:
#             raise HTTPException(status_code=400, detail="No face image path found in this log entry")
        
#         if not os.path.exists(detection_log.face_image_path):
#             raise HTTPException(status_code=404, detail=f"Face image file not found at: {detection_log.face_image_path}")
        
#         # Read and process image
#         img = cv2.imread(detection_log.face_image_path)
#         if img is None:
#             raise HTTPException(status_code=500, detail=f"Failed to read image file from: {detection_log.face_image_path}")
        
#         # Add to vector database
#         result = vectorDatabase(img, request.name_front)
        
#         # Update the person name
#         detection_log.person = request.name_front
#         db.commit()
        
#         return {
#             "success": True,
#             "message": f"Image added to vector database for {request.name_front}",
#             "log_id": request.id,
#             "corrected_person": request.name_front,
#             "image_shape": img.shape,
#             "detection_time": detection_log.detection_time.isoformat() if detection_log.detection_time else None
#         }
            
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(f"Error processing log ID {request.id}: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))
    

# @app.exception_handler(Exception)
# async def global_exception_handler(request, exc):
#     """Global exception handler"""
#     print(f"Global error: {exc}")
#     return JSONResponse(
#         status_code=500,
#         content={"detail": "An internal server error occurred"}
#     )


# from pathlib import Path
# @app.get("/qs-save")
# async def video_ws1(
#     # request: SaveRequest,
#     # db: Session = Depends(get_db)
# ):
#     IMAGE_ROOT = "./Detection/TensorRT/qdrant_images"


#     root = Path(IMAGE_ROOT)

#     for person_dir in root.iterdir():
#         if not person_dir.is_dir():
#             print('empttyyyyyyyyyyyyyyyyyyy')
#             continue

#         person_name = person_dir.name
#         print(f"\n📁 Person: {person_name}")

#         for img_path in person_dir.glob("*.*"):
#             try:
#                 img = cv2.imread(str(img_path))
#                 if img is None:
#                     print("❌ cannot read", img_path.name)
#                     continue
#                 vectorDatabase(img, f'{person_name}', save_mode=True)


                

#                 print("✅ inserted", img_path.name)

#             except Exception as e:
#                 print("❌ error:", img_path.name, e)
        
#     return {
#             "success": True,
            
#         }
            


# @app.get("/qd-remove")
# async def qd_remove(
#     # request: SaveRequest,
#     # db: Session = Depends(get_db)
# ):
#     # print('id', request.id)
#     # # name_to_ai = dictionary.get(request.name_front)
#     # name_to_ai = request.name_front
#     try:
#         name_to_ai = ['3', 'ygj', 'حسین زاده', 'hossein_zade', 'امیر زمانی','amir_zamani']
#         result = removeDatabase(name_to_ai)
   
#         return {
#             "success": True,
#             "message": f"Image added to vector database for {name_to_ai}",
#             # "log_id": request.id,
#             "corrected_person": name_to_ai,
#             "vector_database_result": result if result else "Success",
#         }
            
#     except Exception as e:
#         # print(f"Error processing log ID {request.id}: {e}")
#         import traceback
#         traceback.print_exc()
#         return {"error": str(e)}
#     finally:
#         print('Processing complete')
    
    
    
    
    
    

















































    
# # Standard library imports
# import os
# import json
# import base64
# import asyncio
# from io import BytesIO
# from datetime import datetime, date, time
# from typing import List, Optional
# from collections import defaultdict, deque, Counter
# from pathlib import Path
# # from backend.app.api import router
# from .routers import personnel
# # Third-party imports
# import numpy as np
# import cv2
# import av
# from fastapi import FastAPI, WebSocket, Depends, Query, HTTPException, WebSocketDisconnect
# from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from sqlalchemy.orm import Session
# from sqlalchemy import func

# # Local application imports
# from .models.database import save_detection_with_face, get_db, DetectionLog
# from .api import router as api_router
# from Detection.TensorRT.infer import ws_transfer, vectorDatabase, removeDatabase


# from collections import defaultdict, deque, Counter


# # Global state
# clients = set()
# processor_task = None

# app = FastAPI(title="Face Recognition API", version="1.0.0")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# app.include_router(api_router, prefix="/api/v1")
# # app.include_router(router, prefix="/api/v1")


# class DailyAttendanceReport(BaseModel):
#     person: str
#     date: date
#     first_in: time
#     last_out: time
#     total_minutes: int

#     class Config:
#         from_attributes = True


# class StreamConfig(BaseModel):
#     webcam: bool= False
#     gpu: int = 0
#     threshold: float = 0.5
#     collection: str = "n3"


# @app.get("/")
# async def index():
#     return {
#         "message": "Welcome to Face Recognition API",
#         "description": "Real-time face recognition with FastAPI WebSocket streaming",
#         "endpoints": {
#             "docs": "/docs",
#             "video_page": "/ws-video",
#             "stream_status": "/api/v1/stream/status",
#             "reports": "/reports/daily?from_date=YYYY-MM-DD&to_date=YYYY-MM-DD"
#         }
#     }



# @app.get("/ws-video")
# async def video_page():
#     """Serve the video page HTML"""
#     html_path = os.path.join(os.path.dirname(__file__), "templates", "video_page.html")
    
#     try:
#         with open(html_path, "r", encoding="utf-8") as f:
#             html_content = f.read()
#         return HTMLResponse(content=html_content)
#     except FileNotFoundError:
#         # Fallback in case file is not found
#         return HTMLResponse(content="<h1>Video page template not found</h1>", status_code=404)




# @app.websocket("/ws/video")
# async def video_ws(websocket: WebSocket):
#     await websocket.accept()
    
#     # Add client to broadcast list
#     clients.add(websocket)
    
#     try:
#         # Create a database session for this connection
#         db = next(get_db())
    
#         # Keep connection alive
#         while True:
#             try:
#                 data = await websocket.receive_text()
                    
#             except WebSocketDisconnect:
#                 break
#             except Exception as e:
#                 print(f"WebSocket error: {e}")
#                 break
    
#     except Exception as e:
#         print(f"Connection error: {e}")
#     finally:
#         # Remove client on disconnect
#         clients.discard(websocket)
#         try:
#             await websocket.close()
#         except:
#             pass



# # Global state
# clients = set()
# processor_task = None

# VOTE_WINDOW = 1000  # how many last frames used

# track_history = defaultdict(lambda: {
#     "names": deque(maxlen=VOTE_WINDOW),
#     "scores": deque(maxlen=VOTE_WINDOW),
#     "faces": deque(maxlen=VOTE_WINDOW),
#     "frames": deque(maxlen=VOTE_WINDOW),
# })



# async def video_broadcaster():
#     """Background task that streams video to all connected clients"""
#     global track_history
#     RTSP_URL = "rtsp://Jafari:Asd12345@192.168.110.20:554/Streaming/Channels/301"
    
#     print("🎬 Starting video broadcaster...")
    
#     while True:
#         try:
#             # Open connection inside loop to handle reconnections
#             # container = av.open(RTSP_URL, options={"rtsp_transport": "tcp", "flags": "low_delay", "fflags": "nobuffer"}).decode(video=0)

            
#             # print('✅ Connected to RTSP stream')
#             frame_count =0

#             cap = cv2.VideoCapture(0)

#             if not cap.isOpened():
#                 print("❌ Cannot open camera")
#                 exit()
#             # print('clients', clients)
#             if  clients:
#                 await save_in_database()
#             # for packet in container.demux():

#             clip_length = 100
#             half_clip = clip_length // 2
#             save_database = False


#             while True:
#                 #  for frame in container:
#                     if not clients:
#                         await asyncio.sleep(0.1)
#                         continue
#                     # for obj, history in track_history.items():
#                     #     print(f"OBJ {obj}: {list(history['names'])}")
#                     ret, frame = cap.read()
#                     frame_count +=1
              
#                     try:
                    
#                             frame = frame.to_ndarray(format="bgr24")
                            
#                             # Process with TensorRT
#                             data = ws_transfer(frame)
#                             frame_bytes = data['frame']
#                             persons = data['persons']
#                             scores = data['scores']
#                             faces = data['faces']
#                             objs = data['objs']

#                             current_objs = set(objs)
#                             final_persons = []
#                             if len(persons) :
#                                 for i, (person, score, obj, face) in enumerate(zip(persons, scores, objs, faces)):
#                                         history = track_history[obj]
#                                         history["frames"].append(frame.copy())
#                                         if person and person != "Unknown":

#                                             history["names"].append(person)
#                                             history["scores"].append(score)
#                                             history["faces"].append(face)

                                    
                                     


#                             all_tracked_objs = set(track_history.keys())
#                             disappeared_objs = all_tracked_objs - current_objs
                            
#                             for obj in disappeared_objs:
#                                 history = track_history[obj]

#                                 if history["names"]:
#                                     final_name = Counter(history["names"]).most_common(1)[0][0]


#                                     valid_indices = [i for i, n in enumerate(history["names"]) if n == final_name]
#                                     if valid_indices:
#                                         best_idx = max(valid_indices, key=lambda i: history["scores"][i])
#                                         final_score = history["scores"][best_idx]
#                                         final_face = history["faces"][best_idx]

#                                         start_idx = max(0, best_idx - half_clip)
#                                         end_idx = min(len(history["frames"]), best_idx + half_clip)
#                                         frames_to_save = list(history["frames"])[start_idx:end_idx]
                                        
#                                         log_id = save_detection_with_face(
#                                             person=final_name,
#                                             confidence=float(final_score),
#                                             face_image=final_face,
#                                             frames_to_save= frames_to_save
#                                         )

#                                         if log_id is not None:
#                                             save_database = True  
                                            
                                        
                                           

                                           



#                                 # remove object from history after saving
#                                 del track_history[obj]                            
                                                                                                
                                          
                                        
#                             if frame_count % 1 ==0 and save_database:
#                                 save_database = False
#                                 await save_in_database()
                            
#                             metadata = {
#                                 "type": "video_metadata",
#                                 "persons": persons,
#                                 "scores": [float(s) if s is not None else 0.0 for s in scores],
#                                 "objs": objs,
#                                 'final_persons': final_persons
#                             }
                            
#                             # Broadcast to all clients
#                             disconnected = []
#                             for ws in clients:
#                                 try:
#                                     await ws.send_bytes(frame_bytes)
#                                     await ws.send_json(metadata)
#                                 except Exception:
#                                     disconnected.append(ws)
                            
#                             for ws in disconnected:
#                                 clients.discard(ws)
                                
#                     except Exception as e:
#                         print(f"⚠️ Error processing frame: {e}")
#                         continue
                        
#         except Exception as e:
#             print(f"⚠️ RTSP connection error: {e}")
#             print("🔄 Reconnecting in 5 seconds...")
#             await asyncio.sleep(5)
#         finally:
#             if 'container' in locals():
#                 container.close()

# async def save_in_database():
#     """Broadcast new detection to all connected clients"""
#     offset = 0
#     limit = 20
#     db = next(get_db())
#     logs = db.query(DetectionLog)\
#         .order_by(DetectionLog.detection_time.desc())\
#         .offset(offset)\
#         .limit(limit)\
#         .all()
    
#     logs_data = []
#     base_url = "http://192.168.10.9:8001"  # Your server URL
    
#     for log in logs:
#         face_image_base64 = None
#         if log.face_image_path and os.path.exists(log.face_image_path):
#             try:
#                 img = cv2.imread(log.face_image_path)
#                 if img is not None:
#                     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                     _, buffer = cv2.imencode('.jpg', img_rgb, [cv2.IMWRITE_JPEG_QUALITY, 70])
#                     face_image_base64 = base64.b64encode(buffer).decode('utf-8')
#             except:
#                 pass
        
#         # Generate video URL if video exists
#         video_url = None
#         if log.video_path and os.path.exists(log.video_path):
#             video_url = f"{base_url}/api/v1/detections/{log.id}/video"
        
#         logs_data.append({
#             "id": log.id,
#             "person": log.person,
#             "confidence": float(log.confidence) if log.confidence is not None else 0.0,
#             "detection_time": log.detection_time.isoformat() if log.detection_time else None,
#             "face_image_path": log.face_image_path,
#             "face_image_base64": face_image_base64,
#             "video_url": video_url  # Add video URL
#         })
    
#     detection_info = {
#         "type": "new_detection",
#         "detection": logs_data
#     }
    
#     disconnected = []
#     for i, ws in enumerate(clients):
#         try:
#             await ws.send_json(detection_info)
#         except Exception as e:
#             print(f"   ❌ Failed to send to client #{i}: {e}")
#             disconnected.append(ws)
    
#     for ws in disconnected:
#         clients.discard(ws)

        
    

# @app.on_event("startup")
# async def startup_event():
#     """Initialize on startup"""
#     print("Starting up Face Recognition API...")
    
#     # Start video broadcaster when server starts
#     global processor_task
#     processor_task = asyncio.create_task(video_broadcaster())
#     print("🎬 Video broadcaster started automatically")
    
#     print("Models will be loaded on first WebSocket connection")

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Clean up on shutdown"""
#     global processor_task
#     if processor_task:
#         processor_task.cancel()
#         try:
#             await processor_task
#         except asyncio.CancelledError:
#             pass
#     print("Video broadcaster stopped")






# # Define Pydantic model for request body
# class SaveRequest(BaseModel):
#     id: int
#     name_front: str

# @app.post("/ws-save")
# async def video_ws(
#     request: SaveRequest,
#     db: Session = Depends(get_db)
# ):
#     print('id', request.id)

#     dictionary = {
#         'محمدرضا تیره گری': "Tiregary",
#         'حسین زاده': "hossein_zade",
#         'ناشناس یک': "unk1",

#     }

#     # name_to_ai = dictionary.get(request.name_front)
#     name_to_ai = request.name_front

#     try:
#         # Query the database to get the detection log by ID
#         detection_log = db.query(DetectionLog).filter(DetectionLog.id == request.id).first()
        
#         if not detection_log:
#             return {"error": f"Detection log with ID {request.id} not found"}
        
#         # print(f"Found detection log for: {detection_log.person}")
        
#         # Check if there's a face image path
#         if not detection_log.face_image_path:
#             return {"error": "No face image path found in this log entry"}
        
#         # Check if the file exists
#         if not os.path.exists(detection_log.face_image_path):
#             return {"error": f"Face image file not found at: {detection_log.face_image_path}"}
        
#         # Read image from file path
#         img = cv2.imread(detection_log.face_image_path)
        
#         if img is None:
#             return {"error": f"Failed to read image file from: {detection_log.face_image_path}"}
        
#         print(f'Image loaded successfully. Shape: {img.shape}, Path: {detection_log.face_image_path}')
        
#         # Call vectorDatabase with the image and name
#         # print(f"Calling vectorDatabase with image and name: {name_to_ai}")
        
#         # Option 1: If vectorDatabase(img, name) adds to database
#         result = vectorDatabase(img, name_to_ai)
        
#         # Update the person name in the log if needed
#         detection_log.person = name_to_ai
#         db.commit()
        
#         return {
#             "success": True,
#             "message": f"Image added to vector database for {name_to_ai}",
#             "log_id": request.id,
#             "original_person": detection_log.person,
#             "corrected_person": name_to_ai,
#             "image_path": detection_log.face_image_path,
#             "image_shape": img.shape,
#             "vector_database_result": result if result else "Success",
#             "detection_time": detection_log.detection_time.isoformat() if detection_log.detection_time else None
#         }
            
#     except Exception as e:
#         print(f"Error processing log ID {request.id}: {e}")
#         import traceback
#         traceback.print_exc()
#         return {"error": str(e)}
#     finally:
#         print('Processing complete')


# from pathlib import Path
# @app.get("/qs-save")
# async def video_ws1(
#     # request: SaveRequest,
#     # db: Session = Depends(get_db)
# ):
#     IMAGE_ROOT = "./Detection/TensorRT/qdrant_images"


#     root = Path(IMAGE_ROOT)

#     for person_dir in root.iterdir():
#         if not person_dir.is_dir():
#             print('empttyyyyyyyyyyyyyyyyyyy')
#             continue

#         person_name = person_dir.name
#         print(f"\n📁 Person: {person_name}")

#         for img_path in person_dir.glob("*.*"):
#             try:
#                 img = cv2.imread(str(img_path))
#                 if img is None:
#                     print("❌ cannot read", img_path.name)
#                     continue
#                 vectorDatabase(img, f'{person_name}', save_mode=True)


                

#                 print("✅ inserted", img_path.name)

#             except Exception as e:
#                 print("❌ error:", img_path.name, e)
        
#     return {
#             "success": True,
            
#         }
            


# @app.get("/qd-remove")
# async def qd_remove(
#     # request: SaveRequest,
#     # db: Session = Depends(get_db)
# ):
#     # print('id', request.id)
#     # # name_to_ai = dictionary.get(request.name_front)
#     # name_to_ai = request.name_front
#     try:
#         name_to_ai = ['3', 'ygj', 'حسین زاده', 'hossein_zade', 'امیر زمانی','amir_zamani']
#         result = removeDatabase(name_to_ai)
   
#         return {
#             "success": True,
#             "message": f"Image added to vector database for {name_to_ai}",
#             # "log_id": request.id,
#             "corrected_person": name_to_ai,
#             "vector_database_result": result if result else "Success",
#         }
            
#     except Exception as e:
#         # print(f"Error processing log ID {request.id}: {e}")
#         import traceback
#         traceback.print_exc()
#         return {"error": str(e)}
#     finally:
#         print('Processing complete')


# @app.get("/reports/daily", response_model=List[DailyAttendanceReport])
# def daily_report(
#     from_date: date = Query(..., description="Start date in YYYY-MM-DD format"),
#     to_date: date = Query(..., description="End date in YYYY-MM-DD format"),
#     db: Session = Depends(get_db)
# ):
#     """Get daily attendance reports between two dates"""
#     try:
#         rows = (
#             db.query(
#                 DetectionLog.person.label("person"),
#                 func.date(DetectionLog.detection_time).label("date"),
#                 func.min(DetectionLog.detection_time).label("first_in"),
#                 func.max(DetectionLog.detection_time).label("last_out"),
#             )
#             .filter(DetectionLog.detection_time.between(from_date, to_date))
#             .group_by(
#                 DetectionLog.person,
#                 func.date(DetectionLog.detection_time)
#             )
#             .order_by("date", DetectionLog.person)
#             .all()
#         )
        
#         reports = []
#         for r in rows:
#             total_minutes = int(
#                 (r.last_out - r.first_in).total_seconds() / 60
#             )
#             reports.append(DailyAttendanceReport(
#                 person=r.person,
#                 date=r.date,
#                 first_in=r.first_in.time(),
#                 last_out=r.last_out.time(),
#                 total_minutes=total_minutes
#             ))
        
#         return reports
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.on_event("shutdown")
# async def shutdown_event():
#     """Clean up on shutdown"""
#     global processor_task
#     if processor_task:
#         processor_task.cancel()
#         try:
#             await processor_task
#         except asyncio.CancelledError:
#             pass
#     print("Video broadcaster stopped")
#     print("Application shutdown complete")
    
    
    
    