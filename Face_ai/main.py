from source.human_detection import tracking_human_detected
import cv2
import av
from source.face_detection import faceDetection
from source.trt_manager import TensorRTManager
from qdrant_client import QdrantClient
from pathlib import Path
import numpy as np
from qdrant_client.models import PointStruct
import uuid
from ByteTracker.byte_tracker import BYTETracker
from insightface.utils import face_align

frame_id = 0 

def get_color_from_id(track_id):
    np.random.seed(track_id)
    color = tuple(np.random.randint(0, 255, 3).tolist())
    return color

FACE_KPTS = { 3, 4}
FACE_LANDMARK = {0,1,2}
SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (5, 6),
    (11, 12),
    (5, 11), (6, 12)
]



def point_inside_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

client = QdrantClient(url="http://localhost:6333")


def face_decoding(embedding, collocation):
    global client
    search_result = client.query_points(
        collection_name=collocation,
        query=embedding.tolist(),
        limit=1
    )

    person = "Unknown"
    score = 0.0

    if search_result and hasattr(search_result, 'points') and search_result.points:
        best_match = search_result.points[0]

        if hasattr(best_match, 'payload') and best_match.payload:
            person = best_match.payload.get("person", "Unknown")

        if hasattr(best_match, 'score'):
            score = best_match.score

    # Threshold tuning
    if score < 0.35:
        person = "Unknown"
    return person, float(score)



save_face = False
import os
from datetime import datetime
def face_embedding(frame, face_box, landmark): 
    
    face_resized = face_align.norm_crop(frame, landmark.astype(int))
    if save_face:
        SAVE_DIR = "saved_faces"
        os.makedirs(SAVE_DIR, exist_ok=True)
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg" 
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, face_resized)

    if face_resized is None or face_resized.size == 0:
        return None, 0.0

    h, w = face_resized.shape[:2]
    if h < 30 or w < 30:
        return None, 0.0

    # Blur check
    if np.std(face_resized) < 15:
        return None, 0.0 

    # Normalize
    face_norm = (face_resized - 127.5) / 128.0
    input_tensor = face_norm.transpose(2, 0, 1).astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    input_tensor = np.ascontiguousarray(input_tensor)

    embedding = trt_manager.recognize_face(input_tensor)

    if embedding is None:
        return None, 0.0

    emb = embedding[0]
    emb = emb / np.linalg.norm(emb)

    return emb, 1



def box_center(box):
    x1, y1, x2, y2 = box
    return ( (x1 + x2) // 2, (y1 + y2) // 2 )


def box_inside(box_small, box_big):
    cx, cy = box_center(box_small)
    x1, y1, x2, y2 = box_big
    return x1 <= cx <= x2 and y1 <= cy <= y2



face_human_tracker = {}
frame_counter = 0



def draw(frame, face_bbox, face_landmarks, human_keypoints, collocation):
    global face_human_tracker, frame_counter

    frame_counter += 1
    important_points = [0, 1, 2]  # nose, left_eye, right_eye

    # =========================================================
    # STEP 1: Face recognition (run once per face)
    # =========================================================
    face_results = {}

    for face_box, landmark in zip(face_bbox, face_landmarks):
        embedding, rec_score = face_embedding(frame, face_box, landmark)
        x1_f, y1_f, x2_f, y2_f = map(int, face_box[:4])

        if embedding is None:
            person = "Unknown"
            rec_score = 0.0
        else:
            person, rec_score = face_decoding(embedding, collocation)

        face_results[len(face_results)] = {
            "box": [x1_f, y1_f, x2_f, y2_f],
            "person": person,
            "score": rec_score,
        }

    # =========================================================
    # STEP 2: Human centroids (for persistent tracking)
    # =========================================================
    human_centroids = {}

    for human_idx, pose in enumerate(human_keypoints):
        kpts = pose["kpts"]

        valid = [(k[0], k[1]) for k in kpts if k[0] > 0 and k[1] > 0]

        if valid:
            pts = np.array(valid)
            cx, cy = pts.mean(axis=0).astype(int)
            human_centroids[human_idx] = (cx, cy)

    # =========================================================
    # STEP 3: Fast face ↔ human association (NO MASKS)
    # =========================================================
    current_assignments = {}

    for human_idx, pose in enumerate(human_keypoints):
        kpts = pose["kpts"]

        best_face = None
        best_match = 0

        for face_id, data in face_results.items():
            fx1, fy1, fx2, fy2 = data["box"]

            match_count = 0

            for i in important_points:
                x_kpt = int(kpts[i][0])
                y_kpt = int(kpts[i][1])

                if x_kpt > 0 and y_kpt > 0:
                    if fx1 <= x_kpt <= fx2 and fy1 <= y_kpt <= fy2:
                        match_count += 1

            if match_count > best_match:
                best_match = match_count
                best_face = face_id

        if best_face is not None and best_match > 0:
            face_name = face_results[best_face]["person"]
            if face_name != "Unknown":
                current_assignments[human_idx] = face_name
                face_human_tracker[human_idx] = {
                    "name": face_name,
                    "last_seen": frame_counter,
                    "centroid": human_centroids.get(human_idx),
                }

    # =========================================================
    # STEP 4: Persistent tracking propagation
    # =========================================================
    human_face_names = [None] * len(human_keypoints)

    # Apply current detections
    for human_idx, name in current_assignments.items():
        human_face_names[human_idx] = name

    # Recover lost faces via tracker
    for human_idx in range(len(human_keypoints)):
        if (
            human_face_names[human_idx] is None
            and human_idx in face_human_tracker
        ):
            tracker_data = face_human_tracker[human_idx]

            if human_idx in human_centroids:
                cx, cy = human_centroids[human_idx]
                last = tracker_data.get("centroid")

                if last:
                    dx = cx - last[0]
                    dy = cy - last[1]

                    # squared distance (no sqrt → faster)
                    if (dx * dx + dy * dy) < (200 * 200):
                        human_face_names[human_idx] = tracker_data["name"]
                        tracker_data["centroid"] = (cx, cy)
                        tracker_data["last_seen"] = frame_counter
                    else:
                        del face_human_tracker[human_idx]
                else:
                    human_face_names[human_idx] = tracker_data["name"]
                    tracker_data["centroid"] = (cx, cy)
                    tracker_data["last_seen"] = frame_counter

    # =========================================================
    # STEP 5: Cleanup old tracker entries
    # =========================================================
    to_remove = [
        idx
        for idx, data in face_human_tracker.items()
        if frame_counter - data.get("last_seen", 0) > 150
    ]

    for idx in to_remove:
        del face_human_tracker[idx]

    # =========================================================
    # STEP 6: Draw results (no frame.copy for speed)
    # =========================================================
    target_width = 640
    target_height = 540
    h, w = frame.shape[:2]

    scale_x = target_width / w
    scale_y = target_height / h

    output_frame = cv2.resize(frame, (target_width, target_height))

    # Draw faces
    for face_id, data in face_results.items():
        x1, y1, x2, y2 = data["box"]
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        person = data["person"]
        score = data["score"]

        color = get_color_from_id(face_id)

        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

        if y1 > 20:
            cv2.putText(
                output_frame,
                f"{person} ({score:.2f})",
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    # Draw humans
    for human_idx, pose in enumerate(human_keypoints):
        x1, y1, x2, y2 = map(int, pose["bbox"])
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        face_name = human_face_names[human_idx]

        if face_name and face_name != "Unknown":
            color = (0, 255, 0)
            label = face_name
        else:
            color = (0, 0, 255)
            label = "Unknown"

        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

        label_y = y1 - 5 if y1 > 30 else y2 + 20
        cv2.putText(
            output_frame,
            label,
            (x1 + 5, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

        # draw minimal face skeleton
        kpts = pose["kpts"]

        for i in important_points:
            x_kpt = int(kpts[i][0] * scale_x)
            y_kpt = int(kpts[i][1] * scale_y)

       
            if x_kpt > 0 and y_kpt > 0:
                cv2.circle(output_frame, (x_kpt, y_kpt), 2, color, -1)

        if all(kpts[i][0] > 0 for i in important_points):
            # nose = (int(kpts[0][0]), int(kpts[0][1]))
            # left_eye = (int(kpts[1][0]), int(kpts[1][1]))
            # right_eye = (int(kpts[2][0]), int(kpts[2][1]))
            nose = (
                int(kpts[0][0] * scale_x),
                int(kpts[0][1] * scale_y)
            )

            left_eye = (
                int(kpts[1][0] * scale_x),
                int(kpts[1][1] * scale_y)
            )

            right_eye = (
                int(kpts[2][0] * scale_x),
                int(kpts[2][1] * scale_y)
            )
            cv2.line(output_frame, nose, left_eye, color, 1)
            cv2.line(output_frame, nose, right_eye, color, 1)

    return output_frame


def camera_processing(USE_WEBCAM, cap, container,  COLLECTION):
    global frame_id
    cv2.namedWindow('YOLO Pose', cv2.WINDOW_NORMAL)

    while True:
        if USE_WEBCAM and cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            try:
                if container:
                    frame = next(container.decode(video=0))
                    frame = frame.to_ndarray(format="bgr24")
            except StopIteration:
                print("⚠️ Stream ended")
                break
            except Exception as e:
                print("⚠️ RTSP error:", e)
                break
        frame_id += 1 
        original_frame = frame.copy()

        bbox_face, landmarks = faceDetection(frame,  trt_manager)

        # bboxes_np = np.array(bbox_face, dtype=np.float32)
        # face_tracking_bbox = tracker.update(
        #         bboxes_np,
        #         frame.shape[:2],
        #         frame.shape[:2]
        #     )
        
        human_key= tracking_human_detected(frame_id, original_frame)

        frame = draw(original_frame, bbox_face , landmarks,  human_key, COLLECTION)
        cv2.imshow("YOLO Pose", frame)

        if cv2.waitKey(1) & 0xFF ==  ord('q'):
            break
    if USE_WEBCAM:
        cap.release()
    cv2.destroyAllWindows()

def FaceEmbedding(frame, person_name, COLLECTION, img_ref_id= None ):
    bbox_face, landmarks = faceDetection(frame,  trt_manager)
    # cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    # cv2.namedWindow(f'Face {person_name}', cv2.WINDOW_NORMAL)
    # cv2.imshow("Frame", frame)

    for face_box, landmark in zip(bbox_face, landmarks):

        embedding = face_embedding(frame, face_box, landmark)

        client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "person": person_name,
                    "img_ref_id": img_ref_id
                        }
                    )])
                

        # cv2.imshow(f'Face {person_name}', face_crop)
        # cv2.waitKey(10000)

    # cv2.destroyAllWindows()
    

def FaceDecoding(frame, person_name, COLLECTION):
    bbox_face, landmarks = faceDetection(frame,  trt_manager)
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow(f'Face {person_name}', cv2.WINDOW_NORMAL)
    cv2.imshow("Frame", frame)

    while True:
        for face_box, landmark in zip(bbox_face, landmarks):
        
            embedding = face_embedding(frame, face_box, landmark)
            person, rec_score = face_decoding(embedding, COLLECTION)

            
            print('person_name', person_name)
            print('rec name=', person)
            print('rec score=', rec_score)
            print('#' * 60)

            # cv2.imshow(f'Face {person_name}', face_crop)

        if cv2.waitKey(1) & 0xFF ==  ord('q'):
            break
    
    cv2.destroyAllWindows()
    




USE_WEBCAM = False  
RTSP_URL = "rtsp://Jafari:Asd@98500@192.168.110.20:554/Streaming/Channels/301"

trt_manager = TensorRTManager()
current_dir = Path(__file__).parent / 'source/engines'

trt_manager.initialize(
    det_engine_path = current_dir / "retinaface_fp16.engine",
    rec_engine_path = current_dir / "arcface_fp16.engine"
    )

class Args:
    track_thresh = 0.4
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20 = False

tracker = BYTETracker(Args())
cap= None
container= None
CAMERA_PROCESSING = True
FACE_ENCODING = False
FACE_DECODING = False
COLLECTION = 'n12'

if __name__ == "__main__":



    if CAMERA_PROCESSING:
        if USE_WEBCAM:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            if not cap.isOpened():
                print("❌ Cannot open webcam")
                exit()
            print("✅ Webcam started")

        else:
            try:
                container = av.open(
                    RTSP_URL,
                    options={
                        "rtsp_transport": "tcp",
                        "max_delay": "1000000"
                    }
                )
                print("✅ Connected to RTSP stream")
            except Exception as e:
                print("❌ RTSP connection failed:", e)
                exit()
        camera_processing(USE_WEBCAM, cap, container,  COLLECTION)
    if FACE_ENCODING:
        IMAGE_ROOT = "./qdrant_images"
        root = Path(IMAGE_ROOT)

        for person_dir in root.iterdir():
            if not person_dir.is_dir():
                print('empttyyyyyyyyyyyyyyyyyyy')
                continue
            person_name = person_dir.name
            print(f"\n📁 Person: {person_name}")
            for img_path in person_dir.glob("*.*"):
                    img = cv2.imread(str(img_path))
                    FaceEmbedding(img , f'{person_name}', COLLECTION)

    if FACE_DECODING:
        IMAGE_ROOT = "./qdrant_single"
        root = Path(IMAGE_ROOT)

        for person_dir in root.iterdir():
            if not person_dir.is_dir():
                print('empttyyyyyyyyyyyyyyyyyyy')
                continue
            person_name = person_dir.name
            print(f"\n📁 Person: {person_name}")
            for img_path in person_dir.glob("*.*"):
                    img = cv2.imread(str(img_path))
                    FaceDecoding(img , f'{person_name}', COLLECTION)



