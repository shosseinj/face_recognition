import cv2
import av
import numpy as np


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


class Config:
    RTSP_URL = "rtsp://Jafari:Asd@98500@192.168.110.20:554/Streaming/Channels/301" # saloon

    # RTSP_URL = "rtsp://Jafari:Asd@98500@192.168.110.14:554/Streaming/Channels/101"
    BASE_URL = "http://192.168.10.9:9000"
    CLIP_LENGTH = 100
    HALF_CLIP = CLIP_LENGTH // 2
    VOTE_WINDOW = 1000

config = Config()




sources = [
# {"type": "cv2", "src": 'http://192.168.50.20:8080/video'},
{"type": "cv2", "src": './video6.mp4'},
{"type": "cv2", "src": 0},
# {"type": "cv2", "src": './video6.mp4'},
# {"type": "cv2", "src": 0},
# {"type": "cv2", "src": 0},
# {"type": "cv2", "src": 0},
# {"type": "rtsp", "src": "rtsp://Jafari:Asd@98500@192.168.110.14:554/Streaming/Channels/101"},
# {"type": "cv2", "src": 0},
# {"type": "rtsp", "src": config.RTSP_URL}
]
gen = frame_generator_batch(sources)

from ultralytics import YOLO

# Load model
model = YOLO("./yolo26s-pose.engine") 

while True:
    
    batch_frames, batch_camera_ids, batch_indices = next(gen) 
    
    results = model(batch_frames, verbose=False)
    
    # Print results
    for batch_frames, batch_camera_ids, batch_indices in gen:
        results = model(batch_frames, verbose=False)
        
        for i, result in enumerate(results):
            if result.keypoints is not None:
                annotated = result.plot()
                cv2.imshow(f"Camera_{batch_camera_ids[i]}", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()