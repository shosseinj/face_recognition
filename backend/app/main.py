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

from fastapi import FastAPI, WebSocket, Depends, HTTPException, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from .models.database import save_detection_with_face, get_db, DetectionLog
from .api import router as api_router
from Detection.TensorRT.infer import ws_transfer, vectorDatabase, removeDatabase
from Detection.TensorRT.source.human_detection import detect_person_pose

processor_task = None

app = FastAPI(title="Face Recognition API", version="1.0.0")
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
    RTSP_URL = "rtsp://Jafari:Asd12345@192.168.110.20:554/Streaming/Channels/301"
    BASE_URL = "http://127.0.0.1:5000"
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
    """Background task that streams video to all connected clients"""
    print("🎬 Starting video broadcaster...")
    
    # Try camera first, fallback to RTSP
    cap = cv2.VideoCapture(0)
    use_camera = cap.isOpened()   and False
    
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
                    # "flags": "low_delay", 
                    # "fflags": "nobuffer"
                })
                print('✅ Connected to RTSP stream')
            
            frame_count = 0
            clip_length = 100
            half_clip = clip_length // 2

            while True:
                # Yield control to event loop
                await asyncio.sleep(0.001)
                
                if manager.count == 0:
                    await asyncio.sleep(0.1)
                    continue
                
                # Send logs periodically (every 5 seconds) as backup
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
                        if person and person != "Unknown":
                            history["names"].append(person)
                            history["scores"].append(score)
                            history["faces"].append(face)
                    
                    # Handle disappeared objects
                    disappeared_objs = set(track_history.keys()) - current_objs
                    
                    for obj in disappeared_objs:
                        history = track_history[obj]
                        if history["names"]:
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
                                    # IMMEDIATELY send the new logs when a detection is saved
                                    asyncio.create_task(save_in_database())
                                    last_log_send = time.time()  # Reset the periodic timer
                        
                        del track_history[obj]
                    
                    # Broadcast video frame to clients
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
    """Broadcast new detection to all connected clients"""
    offset = 0
    limit = 20
    db = next(get_db())
    try:  # ✅ Added try-finally
        logs = db.query(DetectionLog)\
            .order_by(DetectionLog.detection_time.desc())\
            .offset(offset)\
            .limit(limit)\
            .all()
        
        logs_data = []
        base_url = config.BASE_URL
        
        for log in logs:
            face_image_base64 = None
            if log.face_image_path and os.path.exists(log.face_image_path):
                try:
                    img = cv2.imread(log.face_image_path)
                    if img is not None:
                        #img_rgb = img#cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ####
                        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        face_image_base64 = base64.b64encode(buffer).decode('utf-8')
                except:
                    pass
            
            video_url = None
            if log.video_path and os.path.exists(log.video_path):
                video_url = f"{base_url}/api/v1/detections/{log.id}/video"
            
            logs_data.append({
                "id": log.id,
                "person": log.person,
                "confidence": float(log.confidence) if log.confidence is not None else 0.0,
                "detection_time": log.detection_time.isoformat() if log.detection_time else None,
                "face_image_path": log.face_image_path,
                "face_image_base64": face_image_base64,
                "video_url": video_url
            })
        
        detection_info = {
            "type": "new_detection",
            "detection": logs_data
        }
        
        await manager.broadcast_json(detection_info)
        print(f"📊 Broadcast to {manager.count} clients")  # ✅ Using count property
    finally:
        db.close()  # ✅ Close the session
      


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


from pathlib import Path
@app.get("/qs-save")
async def video_ws1(
    # request: SaveRequest,
    # db: Session = Depends(get_db)
):
    IMAGE_ROOT = "./Detection/TensorRT/qdrant_images"


    root = Path(IMAGE_ROOT)

    for person_dir in root.iterdir():
        if not person_dir.is_dir():
            print('empttyyyyyyyyyyyyyyyyyyy')
            continue

        person_name = person_dir.name
        print(f"\n📁 Person: {person_name}")

        for img_path in person_dir.glob("*.*"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    print("❌ cannot read", img_path.name)
                    continue
                vectorDatabase(img, f'{person_name}', save_mode=True)


                

                print("✅ inserted", img_path.name)

            except Exception as e:
                print("❌ error:", img_path.name, e)
        
    return {
            "success": True,
            
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
        name_to_ai = ['3', 'ygj', 'حسین زاده', 'hossein_zade', 'امیر زمانی','amir_zamani']
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    