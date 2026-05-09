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
from io import BytesIO  # IMPORT THIS!
# from Face_ai.main import FrameProcessing, FaceEmbedding, FaceEmbeddingCropping
from Face_ai.main import ModelManager

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
        self.active_connections: Dict[str, Set[WebSocket]] = {}  # camera_id -> set of connections
        self.all_connections: Set[WebSocket] = set()  # For global broadcast if needed
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, camera_id: str = "all"):
        await websocket.accept()
        async with self._lock:
            if camera_id not in self.active_connections:
                self.active_connections[camera_id] = set()
            self.active_connections[camera_id].add(websocket)
            self.all_connections.add(websocket)
        print(f"✅ Client connected to camera '{camera_id}'. Total: {len(self.all_connections)}")

    async def broadcast_json(self, message: dict):
        """Send JSON to all connected clients"""
        if not self.active_connections:
            return
        
        disconnected = []
        async with self._lock:
            # Collect all WebSocket connections from all camera groups
            connections = []
            for connections_set in self.active_connections.values():
                connections.extend(list(connections_set))
        
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Failed to send to client: {e}")
                disconnected.append(connection)
        
        # Clean up dead connections
        async with self._lock:
            for conn in disconnected:
                # Remove from all camera-specific sets
                for camera_id in list(self.active_connections.keys()):
                    self.active_connections[camera_id].discard(conn)
                    if not self.active_connections[camera_id]:
                        del self.active_connections[camera_id]     
    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.all_connections.discard(websocket)
            for camera_id in list(self.active_connections.keys()):
                self.active_connections[camera_id].discard(websocket)
                if not self.active_connections[camera_id]:
                    del self.active_connections[camera_id]
        print(f"❌ Client disconnected. Total: {len(self.all_connections)}")
    
    async def broadcast_to_camera(self,  datas: dict , batch_camera_id: list):
        """Broadcast to clients subscribed to specific camera"""
        batch_frame = datas['frames']
        batch_persons = datas['persons']
        batch_scores = datas['scores']
        for n, (frame, persons, scores, camera_id) in enumerate(zip(batch_frame, batch_persons, batch_scores, batch_camera_id)):
            camera_id = str(camera_id)
            success, encoded_frame = cv2.imencode('.jpg', frame, 
                                                [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            if not success:
                return
            
            frame_bytes = encoded_frame.tobytes()
            metadata = {
                "type": "video_metadata",
                "cam_id": camera_id,
                "persons": persons,
                "scores": [float(s) if s is not None else 0.0 for s in scores],
                # "objs": datas['objs'],
                "timestamp": time.time()
            }

            if camera_id not in self.active_connections:
                return
            
            disconnected = []
            connections = list(self.active_connections[camera_id])
            
            # Send frame and metadata
            for connection in connections:
                try:
                    await connection.send_bytes(frame_bytes)
                    await connection.send_json(metadata)
                except Exception:
                    disconnected.append(connection)
            
            # Cleanup disconnected clients
            if disconnected:
                async with self._lock:
                    for conn in disconnected:
                        self.active_connections[camera_id].discard(conn)
                        self.all_connections.discard(conn)
    
    @property
    def count(self) -> int:
        return len(self.all_connections)


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

async def handle_camera_switch(websocket: WebSocket, new_camera_id: str):
    """Handle client switching from one camera to another"""
    # Find which camera this client is currently subscribed to
    old_camera_id = None
    async with manager._lock:
        for cam_id, connections in manager.active_connections.items():
            if websocket in connections:
                old_camera_id = cam_id
                break
    
    # If already subscribed to this camera, do nothing
    if old_camera_id == new_camera_id:
        await websocket.send_json({
            "type": "subscription",
            "status": "already_subscribed",
            "camera_id": new_camera_id
        })
        return
    
    # Remove from old camera
    if old_camera_id:
        async with manager._lock:
            if old_camera_id in manager.active_connections:
                manager.active_connections[old_camera_id].discard(websocket)
                if not manager.active_connections[old_camera_id]:
                    del manager.active_connections[old_camera_id]
        
        await websocket.send_json({
            "type": "subscription",
            "status": "unsubscribed",
            "camera_id": old_camera_id
        })
    
    # Add to new camera
    async with manager._lock:
        if new_camera_id not in manager.active_connections:
            manager.active_connections[new_camera_id] = set()
        manager.active_connections[new_camera_id].add(websocket)
    
    # Send confirmation to client
    await websocket.send_json({
        "type": "subscription",
        "status": "subscribed",
        "camera_id": new_camera_id,
        "message": f"Switched from {old_camera_id} to {new_camera_id}" if old_camera_id else f"Subscribed to {new_camera_id}"
    })
    
    print(f"🔄 Client switched camera: {old_camera_id} -> {new_camera_id}")


@app.websocket("/ws/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    """Client connects to specific camera feed"""
    await manager.connect(websocket, camera_id)
    try:
        # Keep connection alive
        while True:
            # Receive subscription updates if needed
            data = await websocket.receive_text()
            # Handle client commands (change camera, etc.)
            if data.startswith("subscribe:"):
                new_camera = data.split(":")[1]
                await handle_camera_switch(websocket, new_camera)
    except WebSocketDisconnect:
        await manager.disconnect(websocket)



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
            
            if person_value :
                personnel = personnel_lookup.get(person_value)
                if personnel:
                    full_name = f"{personnel.fname} {personnel.lname}".strip()
                else:
                    full_name = person_value
            if log.area:
                area =  log.area
            else:
                area = 'بدون ناحیه'
            
            logs_data.append({
                "id": log.id,
                "area": area,
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
            # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
            # cap.grab()
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
def frame_generator_batch(sources, batch_size=None):
    """
    Collects exactly ONE frame from each camera to form a batch.
    
    Args:
        sources: list like [{"type": "cv2", "src": 0}, {"type": "rtsp", "src": "..."}]
        batch_size: ignored (kept for compatibility), always uses number of cameras
    
    Yields:
        (batch_frames, batch_camera_ids, batch_indices)
        - batch_frames: list of frames [frame_cam0, frame_cam1, frame_cam2, ...]
        - batch_camera_ids: list of camera IDs [0, 1, 2, ...]
        - batch_indices: list of indices [0, 1, 2, ...]
    """
    cameras = []
    
    # Open all cameras
    for i, cam in enumerate(sources):
        if cam["type"] == "cv2":
            cap = cv2.VideoCapture(cam["src"])
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {cam['src']}")
            cameras.append({
                "type": "cv2",
                "reader": cap,
                "cam_id": i
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
                "reader": container.decode(video=0),
                "cam_id": i
            })
            print(f"✅ Connected to RTSP {cam['src']}")
    
    num_cams = len(cameras)
    
    while True:
        batch_frames = []
        batch_camera_ids = []
        
        # Collect exactly ONE frame from each camera
        for cam in cameras:
            frame = None
            max_retries = 3
            
            # Try to get a valid frame from this camera
            for attempt in range(max_retries):
                if cam["type"] == "cv2":
                    cap = cam["reader"]
                    cap.grab()
                    ret, frame = cap.read()
                    if not ret:
                        continue
                elif cam["type"] == "rtsp":
                    try:
                        frame = next(cam["reader"])
                        frame = frame.to_ndarray(format="bgr24")
                    except StopIteration:
                        continue
                
                if frame is not None:
                    break
            
            if frame is not None:
                batch_frames.append(frame)
                batch_camera_ids.append(cam["cam_id"])
            else:
                # If camera fails, use a blank frame or skip?
                print(f"⚠️ Warning: Could not read from camera {cam['cam_id']}")
                # Option 1: Use blank frame
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                batch_frames.append(blank_frame)
                batch_camera_ids.append(cam["cam_id"])
        
        # Yield batch (size = number of cameras)
        batch_indices = list(range(len(batch_frames)))
        yield batch_frames, batch_camera_ids, batch_indices
import json
def frame_generator_batch1(sources):
    """
    Use OpenCV for RTSP streams (more stable than PyAV)
    """
    cameras = []
    
    for i, cam in enumerate(sources):
        if cam["type"] == "cv2":
            cap = cv2.VideoCapture(cam["src"])
            if not cap.isOpened():
                print(f"⚠️ Cannot open camera {cam['src']}")
                continue
            cameras.append({
                "type": "cv2",
                "reader": cap,
                "cam_id": i
            })
        elif cam["type"] == "rtsp":
            try:
                # Use OpenCV for RTSP with optimization flags
                cap = cv2.VideoCapture(cam["src"], cv2.CAP_FFMPEG)
                
                # Set RTSP options
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_POS_MSEC, 0)
                
                if not cap.isOpened():
                    print(f"⚠️ Cannot open RTSP {cam['src']}")
                    continue
                    
                cameras.append({
                    "type": "rtsp",
                    "reader": cap,
                    "cam_id": i
                })
                print(f"✅ Connected to RTSP {cam['src']} using OpenCV")
            except Exception as e:
                print(f"❌ Failed to connect to RTSP {cam['src']}: {e}")
                continue
    
    while True:
        batch_frames = []
        batch_camera_ids = []
        
        for cam in cameras:
            frame = None
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    cap = cam["reader"]
                    ret, frame = cap.read()
                    
                    if not ret or frame is None:
                        # Try to reconnect on failure
                        if attempt == max_retries - 1:
                            print(f"⚠️ Reconnecting camera {cam['cam_id']}...")
                            cap.release()
                            cap = cv2.VideoCapture(cam["src"], cv2.CAP_FFMPEG)
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            cam["reader"] = cap
                        continue
                    
                    break
                except Exception as e:
                    print(f"Error reading frame from camera {cam['cam_id']}: {e}")
                    continue
            
            if frame is not None:
                batch_frames.append(frame)
                batch_camera_ids.append(cam["cam_id"])
            else:
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                batch_frames.append(blank_frame)
                batch_camera_ids.append(cam["cam_id"])
        
        yield batch_frames, batch_camera_ids, list(range(len(batch_frames)))
async def process_frame(model,loaded_polygon_points, frame, cam_id, polygon_points, try_objs, track_history):
    try:
            clip_length = 100
            half_clip = clip_length // 2
            # data = {
            #                         "frames": frame,
            #                         "persons": [],
            #                         "scores":[],
            #                         "faces": [],
            #                         "objs": [],
            #                         "ref_img_ids": [],
            #                         "area": [],
            #                     }
            # data['cam_id'] = cam_id
            datas = model.FrameProcessing(frame, loaded_polygon_points)
        

            persons_batch = datas['persons'] 
            scores_batch = datas['scores']
            
            faces_batch = datas['faces']
            areas_batch = datas['area']
            objs_batch = datas['objs']
            ref_img_ids_batch = datas['ref_img_ids']

            for n, (persons, scores, faces, areas, objs, ref_img_ids) in enumerate(zip(persons_batch, scores_batch, faces_batch, areas_batch, objs_batch, ref_img_ids_batch)):

                current_objs = set(objs)


                for i, (person, score, obj, face, ref_img_id, area) in enumerate(zip(persons, scores, objs, faces, ref_img_ids, areas)):
                    history = track_history[obj]
                    history["frames"].append(frame)
                    
                    if person:
                        history["ref_img_ids"].append(ref_img_id)
                        history["names"].append(person)
                        history["scores"].append(score)
                        history["faces"].append(face)
                        history["areas"].append(area)


                disappeared_objs = set(track_history.keys()) - current_objs
                
                for obj in current_objs:
                    try_objs.pop(obj, None) 

                for obj in disappeared_objs:
                    try_objs[obj] = try_objs.get(obj, 0) + 1

                
                for obj, count in list(try_objs.items()):
                    if count > 70:
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
                                    if final_name == 'Unknown':
                                        final_name = final_name +' #' + str(obj)
                                    log_id = await save_detection_with_face(
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
            return datas    
    except Exception as e:
        print('error in process_frame', e)

async def broadcast_frame_to_camera(datas: dict, batch_cam_id: list):
    """Broadcast frame and metadata for specific camera"""
    await manager.broadcast_to_camera(datas, batch_cam_id)




async def video_broadcaster():
    """Background task that streams video and saves detections automatically"""
    print("🎬 Starting video broadcaster...")
    

    with open('./polygon_points.json', 'r') as f:
        loaded_polygon_points = json.load(f)
   
    try_objs = {}
    sources = [
    # {"type": "cv2", "src": 'http://192.168.50.20:8080/video'},
    # {"type": "cv2", "src": './video8.mp4'},
    {"type": "cv2", "src": 0},
    {"type": "cv2", "src": './video7.mp4'},
    {"type": "cv2", "src": './video7.mp4'},
    {"type": "cv2", "src": './video6.mp4'},
    # {"type": "cv2", "src": 0},
    # {"type": "cv2", "src": 0},
    # {"type": "cv2", "src": 0},
    # {"type": "rtsp", "src": "rtsp://Jafari:Asd@98500@192.168.110.14:554/Streaming/Channels/101"},
    # {"type": "rtsp", "src": "rtsp://admin:pMc_897OmId@192.168.110.29:554/Streaming/Channels/101"},
    # {"type": "rtsp", "src": "rtsp://admin:pMc_897OmId@192.168.110.28:554/Streaming/Channels/101"},
    # {"type": "cv2", "src": 0},
    {"type": "rtsp", "src": config.RTSP_URL}
]
    gen = frame_generator_batch(sources)
    model = ModelManager()
    model.initialize()
 
    cap = cv2.VideoCapture('./video6.mp4')
                # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera ")

    while True:
        # Yield control to event loop
            await asyncio.sleep(0.001) 
            batch_frames, batch_camera_ids, batch_indices = next(gen) 
          
            data_list = await process_frame(model, loaded_polygon_points, batch_frames, batch_camera_ids, loaded_polygon_points, try_objs, track_history)
            
            if manager.count > 0:
                # asyncio.create_task(broadcast_frame_to_camera(data, '0'))
                asyncio.create_task(broadcast_frame_to_camera(data_list, batch_camera_ids))


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
        
        asyncio.run(save_detection_with_face(
            person=final_name,
            confidence=float(final_score),
            face_image=final_face,
            frames_to_save=frames_to_save
        ))

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



@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    print(f"Global error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred"}
    )



 