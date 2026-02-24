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


frame_id = 0 

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - interArea
    if union == 0:
        return 0
    return interArea / union


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


import cv2
import numpy as np

arcface_template = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


def align_face(face_img, landmarks):
    """
    face_img: cropped face image
    landmarks: 5x2 array (left_eye, right_eye, nose, left_mouth, right_mouth)
    """

    src = np.array(landmarks, dtype=np.float32)

    # Estimate similarity transform
    M, _ = cv2.estimateAffinePartial2D(src, arcface_template, method=cv2.LMEDS)

    aligned = cv2.warpAffine(
        face_img,
        M,
        (112, 112),
        borderValue=0.0
    )

    return aligned







def face_embedding(frame, face_box, landmark):
    x1_f, y1_f, x2_f, y2_f = map(int, face_box[:4])
    face = frame[y1_f:y2_f, x1_f:x2_f]
            
          
    face = face_align.norm_crop(frame, landmark.astype(int))

    if face is None or face.size == 0:
        return "Unknown", 0.0

    h, w = face.shape[:2]
    if h < 30 or w < 30:
        return "Unknown", 0.0

    try:
        face_resized = cv2.resize(face, (112, 112))
    except:
        return "Unknown", 0.0

    # Blur check
    if np.std(face_resized) < 15:
        return "Unknown", 0.0

    # Normalize
    face_norm = (face_resized - 127.5) / 128.0
    input_tensor = face_norm.transpose(2, 0, 1).astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    input_tensor = np.ascontiguousarray(input_tensor)

    embedding = trt_manager.recognize_face(input_tensor)

    if embedding is None:
        return "Unknown", 0.0

    emb = embedding[0]
    emb = emb / np.linalg.norm(emb)

    return emb


from insightface.utils import face_align




def box_center(box):
    x1, y1, x2, y2 = box
    return ( (x1 + x2) // 2, (y1 + y2) // 2 )


def box_inside(box_small, box_big):
    cx, cy = box_center(box_small)
    x1, y1, x2, y2 = box_big
    return x1 <= cx <= x2 and y1 <= cy <= y2


def draw(frame, face_bbox, face_landmarks, humans, human_keypoints, collocation):

   
        
        
 
    # -------- STEP 1: Recognize each face ONCE --------
    face_results = {}
    k=0
    for face_box, landmark in zip(face_bbox, face_landmarks):
        embedding = face_embedding(frame, face_box, landmark)
        x1_f, y1_f, x2_f, y2_f = map(int, face_box[:4])
        

        person, rec_score = face_decoding(embedding, collocation)
        face_results[k] = {
            "box": [x1_f, y1_f, x2_f, y2_f],
            "person": person,
            "score": rec_score
        }
        k += 1

    # -------- STEP 2: Draw humans --------
    for target, pose in zip(humans, human_keypoints):

        x1, y1, w, h = map(int, target.tlwh)
        x2, y2 = x1 + w, y1 + h

        track_id = target.track_id
        score = target.score
        color = get_color_from_id(track_id)

        human_name = "Unknown"

   



        # Draw skeleton
        kpts = pose["kpts"]
        kpts_conf = pose["kpts_conf"]


        if kpts is not None:

            # COCO format:
            # 0 = nose
            # 1 = left eye
            # 2 = right eye

            important_points = [0, 1, 2]  # nose + eyes

            for face_id, data in face_results.items():

                fx1, fy1, fx2, fy2 = data["box"]

                for idx in important_points:
                    if kpts_conf[idx] > 0.3:   # only reliable keypoints
                     
                        x_kpt, y_kpt = map(int, kpts[idx])
                        cv2.circle(frame, (x_kpt, y_kpt), 6, (0, 255, 0), 1)
                        if fx1 <= x_kpt <= fx2 and fy1 <= y_kpt <= fy2:
                            human_name = data["person"]
                            break

                    



        if kpts is not None:
            for j1, j2 in SKELETON:
                if kpts_conf[j1] > 0.3 and kpts_conf[j2] > 0.3:
                    x1_line, y1_line = map(int, kpts[j1])
                    x2_line, y2_line = map(int, kpts[j2])
                    cv2.line(frame, (x1_line, y1_line),
                             (x2_line, y2_line), (255, 0, 0), 2)

        # Draw human box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame,
                    f"ID {track_id} - {score:.2f} - {human_name}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)

    # -------- STEP 3: Draw face boxes --------
    for face_id, data in face_results.items():
        x1_f, y1_f, x2_f, y2_f = data["box"]
        face_color = get_color_from_id(face_id)

        cv2.rectangle(frame,
                      (x1_f, y1_f),
                      (x2_f, y2_f),
                      face_color,
                      2)

        cv2.putText(frame,
                    f"{data['person']} ({data['score']:.2f})",
                    (x1_f, max(0, y1_f - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    face_color,
                    2)

    return frame



def camera_processing(USE_WEBCAM, cap, container, tracker, COLLECTION):
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
        
        bbox_human , human_key= tracking_human_detected(frame_id, frame)

        frame = draw(original_frame, bbox_face , landmarks, bbox_human, human_key, COLLECTION)
        cv2.imshow("YOLO Pose", frame)

        if cv2.waitKey(1) & 0xFF ==  ord('q'):
            break
    if USE_WEBCAM:
        cap.release()
    cv2.destroyAllWindows()

def FaceEmbedding(frame, person_name, COLLECTION):
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
                    "image": "fname",
                    "image_path": "file_path"
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
    




if __name__ == "__main__":

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
        camera_processing(USE_WEBCAM, cap, container, tracker, COLLECTION)
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



