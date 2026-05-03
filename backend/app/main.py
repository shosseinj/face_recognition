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
from .models.database import DetectionLog, Personnel, PersonnelImage, FACE_STORAGE_DIR
from .models.db_functions import get_db

from fastapi import FastAPI, UploadFile, File, HTTPException
from io import BytesIO
import cv2
import numpy as np
import tempfile
import os

from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, WebSocket, Depends, HTTPException, WebSocketDisconnect, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from .models.db_functions import save_detection_with_face, get_db
from .models.database import DetectionLog, Personnel
from .api import router as api_router
# from Detection.TensorRT.infer import ws_transfer, vectorDatabase, removeDatabase
from io import BytesIO  # IMPORT THIS!
from Face_ai.main import FrameProcessing, FaceEmbedding, FaceEmbeddingCropping



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
    RTSP_URL = "rtsp://Jafari:Asd@98500@192.168.110.20:554/Streaming/Channels/301" # saloon

    # RTSP_URL = "rtsp://Jafari:Asd@98500@192.168.110.14:554/Streaming/Channels/101"
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
        await send_hossein()
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
    "ref_img_ids": deque(maxlen=VOTE_WINDOW),
    "areas": deque(maxlen=VOTE_WINDOW),
})








async def send_hossein():
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
        all_personnel_image = db.query(PersonnelImage).all()
        personnel_lookup = {p.national_code: p for p in all_personnel}
        
        logs_data = []
        base_url = config.BASE_URL
        
        for log in logs:
            # print('len(logs)', len(logs))
            # face_image_base64 = None
            # if log.face_image_path and os.path.exists(log.face_image_path):
            try:
                ref_img_url = None
                ref_obj = db.query(PersonnelImage).filter(PersonnelImage.id == log.ref_img_id).first() 
                img = cv2.imread(log.face_image_path)
                if ref_obj is not None:
                      ref_img_url = ref_obj.image_url
                if (ref_img_url):
                    ref_img = cv2.imread(ref_img_url)

                    # h1, w1 = ref_img.shape[:2]
                    # h2, w2 = img.shape[:2]
                    img = cv2.hconcat([img,ref_img])


                _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                face_image_base64 = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:

                print('log.face_image_path', log.face_image_path)

                print(f'error in pass ={e}')
                pass
            # else:
            #     face_image_base64 = None
            #     print('\n\n\n\n face_image_base64 = None')
            video_url = None
            if log.video_path and os.path.exists(log.video_path):
                video_url = f"{base_url}/api/v1/detections/{log.id}/video"
            
            # Calculate full name if personnel exists
            full_name = 'Unknown'
            person_value = log.person
            
            if person_value and person_value.isdigit() :
                personnel = personnel_lookup.get(person_value)
                if personnel:
                    full_name = f"{personnel.fname} {personnel.lname}".strip()
            if log.area:
                full_name = full_name + ' - '+ log.area
            
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


def frame_generator(sources):
    """
    sources: list like:
        [
            {"type": "cv2", "src": 0},
            {"type": "rtsp", "src": "rtsp://..."}
        ]
    Yields: (camera_id, frame)
    """

    cameras = []

    # 🔥 Open all cameras properly
    for i, cam in enumerate(sources):

        if cam["type"] == "cv2":
            cap = cv2.VideoCapture(cam["src"])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {cam['src']}")

            cameras.append({
                "type": "cv2",
                "reader": cap
            })

        elif cam["type"] == "rtsp":
            container = av.open(
                cam["src"],
                options={
                    "rtsp_transport": "tcp",
                    "flags": "low_delay",
                    "fflags": "nobuffer"
                }
            )

            cameras.append({
                "type": "rtsp",
                "reader": container.decode(video=0)
            })

            print(f"✅ Connected to RTSP {cam['src']}")

    cam_index = 0
    num_cams = len(cameras)

    # 🔥 Round-robin loop
    while True:
        cam = cameras[cam_index]

        frame = None

        if cam["type"] == "cv2":
            cap = cam["reader"]
            cap.grab()
            ret, frame = cap.read()
            if not ret:
                cam_index = (cam_index + 1) % num_cams
                continue

        elif cam["type"] == "rtsp":
            try:
                frame = next(cam["reader"])
                frame = frame.to_ndarray(format="bgr24")
            except StopIteration:
                cam_index = (cam_index + 1) % num_cams
                continue

        # 🔥 Yield in round-robin order
        yield cam_index, frame

        cam_index = (cam_index + 1) % num_cams

import json
async def video_broadcaster():
    """Background task that streams video and saves detections automatically"""
    print("🎬 Starting video broadcaster...")
    

    with open('./polygon_points.json', 'r') as f:
        loaded_polygon_points = json.load(f)
   
    try_objs = {}
    sources = [
    # {"type": "cv2", "src": 'http://192.168.50.19:8080/video'},
    # {"type": "cv2", "src": './video6.mp4'},
    # {"type": "cv2", "src": 0},
    {"type": "rtsp", "src": config.RTSP_URL}
]
    gen = frame_generator(sources)


    while True:
            gen = frame_generator(sources)

    
            
            clip_length = 100
            half_clip = clip_length // 2

            while True:
                # Yield control to event loop
                    await asyncio.sleep(0.001)


                    
                    cam_id, frame = next(gen)
       
                    data = FrameProcessing(frame, loaded_polygon_points)
                    data['cam_id'] = cam_id


                    persons = data['persons'] 
                    scores = data['scores']
                    
                    faces = data['faces']
                    areas = data['area']
                    objs = data['objs']
                    ref_img_ids = data['ref_img_ids']
            


                    current_objs = set(objs)

                    try:
                        for i, (person, score, obj, face, ref_img_id, area) in enumerate(zip(persons, scores, objs, faces, ref_img_ids, areas)):
                            history = track_history[obj]
                            history["frames"].append(frame.copy())
                            
                            if person:
                                history["ref_img_ids"].append(ref_img_id)
                                history["names"].append(person)
                                history["scores"].append(score)
                                history["faces"].append(face)
                                history["areas"].append(area)
                    except Exception as e:
                        print(f'Error updating history: {e}')

                    disappeared_objs = set(track_history.keys()) - current_objs
                    
                    for obj in current_objs:
                        try_objs.pop(obj, None) 

                    for obj in disappeared_objs:
                        try_objs[obj] = try_objs.get(obj, 0) + 1

                    # Process disappeared objects
                    for obj, count in list(try_objs.items()):
                        if count > 70:

                            if history["names"]:
                                history = track_history[obj]

                                if history["names"]:
                                    name_counts = Counter(history["names"])
                                    final_name, count = name_counts.most_common(1)[0]

                                    valid_indices = [i for i, n in enumerate(history["names"]) if n == final_name]

                                    if valid_indices:
                                        best_idx = max(valid_indices, key=lambda i: history["scores"][i])
                                        final_score = history["scores"][best_idx]
                                        final_face = history["faces"][best_idx]
                                        final_ref_img_id = history["ref_img_ids"][best_idx]

                                        start_idx = max(0, best_idx - half_clip)
                                        end_idx = min(len(history["frames"]), best_idx + half_clip)
                                        frames_to_save = list(history["frames"])[start_idx:end_idx]

                                        valid_entries = [
                                                            (i, name, score, face, ref_id, area, frame) 
                                                            for i, (name, score, face, ref_id, area, frame) in enumerate(zip(
                                                                history["names"], history["scores"], history["faces"], 
                                                                history["ref_img_ids"], history["areas"], history["frames"]
                                                            )) 
                                                                if area != 'OUT'
                                                            ]
                                        unique_areas = set(area for _, _, _, _, _, area, _ in valid_entries)
                                        for area in unique_areas:
                                            log_id = save_detection_with_face(
                                                person=final_name,
                                                confidence=float(final_score),
                                                face_image=final_face,
                                                ref_img_id=final_ref_img_id,
                                                frames_to_save=frames_to_save, 
                                                face_to_save=history['faces'],
                                                area= area,
                                                save_video= False
                                            )
                                            if log_id is not None:
                                                asyncio.create_task(send_hossein())
                            
                            del track_history[obj]
                            del try_objs[obj]
                        
                    # Broadcast video frame to clients IF ANY ARE CONNECTED
                    if manager.count > 0:
                        asyncio.create_task(broadcast_frame(data))
                    
                    # Yield after processing
                    await asyncio.sleep(0)
                    

                        


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
    frame = data['frames']

    success, encoded_frame = cv2.imencode('.jpg', frame, 
                                          [cv2.IMWRITE_JPEG_QUALITY, 85])
           
         

    frame_bytes = encoded_frame.tobytes()
    metadata = {
        "type": "video_metadata",
        "persons": data['persons'],
        "cam_id": data['cam_id'],
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
            # print('len(logs)', len(logs))
            # face_image_base64 = None
            # if log.face_image_path and os.path.exists(log.face_image_path):
                # try:
            img = cv2.imread(log.face_image_path)

            # print('log.face_image_path', log.face_image_path)
        # if img is not None:
            # success, encoded_face = cv2.imencode('.jpg', img, 
            #                   [cv2.IMWRITE_JPEG_QUALITY, 85])

                

            # face_image_base64 = encoded_face.tobytes()

            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            face_image_base64 = base64.b64encode(buffer).decode('utf-8')
                # except:
                #     print('error in pass ')
                #     pass
            # else:
            #     face_image_base64 = None
            #     print('\n\n\n\n face_image_base64 = None')
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
    

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    print(f"Global error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred"}
    )



 


# @app.get("/qd-remove")
# async def qd_remove(
#     # request: SaveRequest,
#     # db: Session = Depends(get_db)
# ):
#     # print('id', request.id)
#     # # name_to_ai = dictionary.get(request.name_front)
#     # name_to_ai = request.name_front
#     try:
#         name_to_ai = ['0311344119']
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
    
    
    
    



# Extracted_frame_path = Path("saved_media")
# @app.post("/extract-video-frames")
# async def extract_video_frames(file: UploadFile = File(...)):
    
#     # Validate file type
#     if not file.filename.endswith(".mp4"):
#         raise HTTPException(status_code=400, detail="Only mp4 files are accepted")

#     try:
#         contents = await file.read()
#         print(f"📦 File size: {len(contents)} bytes")

#         # Save to a temporary file (OpenCV needs a file path)
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
#             tmp.write(contents)
#             tmp_path = tmp.name

#         cap = cv2.VideoCapture(tmp_path)

#         if not cap.isOpened():
#             raise HTTPException(status_code=400, detail="Could not open video")

#         frame_count = 0
#         saved_frames = []

#         daily_dir = Extracted_frame_path / Path(file.filename).stem
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Example: save every 30th frame
#             if frame_count % 1 == 0:
#                 daily_dir.mkdir(exist_ok=True, parents=True)
#                 frame_name = f"frame_{frame_count}.jpg"
#                 frame_name = str(Extracted_frame_path / Path(file.filename).stem ) +'/'+ frame_name
#                 cv2.imwrite(frame_name, frame)
#                 saved_frames.append(frame_name)

#             frame_count += 1

#         cap.release()
#         os.remove(tmp_path)

#         return {
#             "total_frames_processed": frame_count,
#             "frames_saved": len(saved_frames),
#             "frame_files": saved_frames
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    



