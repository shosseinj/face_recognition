from Face_ai.source.human_detection import tracking_human_detected
import cv2
import av
from Face_ai.source.face_detection import faceDetection
from Face_ai.source.trt_manager import TensorRTManager
from qdrant_client import QdrantClient
from pathlib import Path
import numpy as np
from qdrant_client.models import PointStruct
import uuid
from Face_ai.ByteTracker.byte_tracker import BYTETracker
from insightface.utils import face_align
from qdrant_client.http import models
from collections import defaultdict, deque, Counter
from statistics import mean
from datetime import datetime, timedelta
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

skeleton_edges = [
    # (0,1),(0,2),(1,3),(2,4),            # head
    # (0,5),(0,6),                         # nose to shoulders
    (5,7),(7,9),                          # left arm
    (6,8),(8,10),                         # right arm
    (5,11),(6,12),                        # shoulders to hips
    (11,12),                               # hip line
    (11,13),(13,15),                       # left leg
    (12,14),(14,16)                        # right leg
]

def point_inside_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

# client = QdrantClient(url="http://localhost:7000")
client = QdrantClient(path = './qdrant_storage')
from qdrant_client.http.models import Distance, VectorParams
collection_name = 'n19'

if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=512,
                distance=Distance.COSINE
            )
        )

def face_decoding(embedding, collocation):
    global client
    search_result = client.query_points(
        collection_name=collocation,
        query=embedding.tolist(),
        limit=1
    )

    person = "Unknown"
    score = 0.0
    ref_img_id = None
    if search_result and hasattr(search_result, 'points') and search_result.points:
        best_match = search_result.points[0]

        if hasattr(best_match, 'payload') and best_match.payload:
            person = best_match.payload.get("person", "Unknown")
            ref_img_id = int(best_match.payload.get("ref_img_id" ))

        if hasattr(best_match, 'score'):
            score = best_match.score

            

    # Threshold tuning
    if score < 0.35:
        person = "Unknown"
    return person, float(score) , ref_img_id



save_face = False
import os
from datetime import datetime




def face_embedding_cropping(frame, face_box, landmark): 
    
    face_resized = face_align.norm_crop(frame, landmark.astype(int))
# 
    x1_f, y1_f, x2_f, y2_f = map(int, face_box[:4])
    face_resized = frame[y1_f:y2_f, x1_f:x2_f]
    face_resized = cv2.resize(face_resized, (112,112))
    # 
    if face_resized is None or face_resized.size == 0:
        return None, 0.0

    face_norm = (face_resized - 127.5) / 128.0
    input_tensor = face_norm.transpose(2, 0, 1).astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    input_tensor = np.ascontiguousarray(input_tensor)

    embedding = trt_manager.recognize_face(input_tensor)

    if embedding is None:
        return None, 0.0

    emb = embedding[0]
    emb = emb / np.linalg.norm(emb)

    return emb, 1, face_resized




def face_embedding(frame, face_box, landmark): 
    
    face_resized = face_align.norm_crop(frame, landmark.astype(int))
# 
    x1_f, y1_f, x2_f, y2_f = map(int, face_box[:4])
    face_resized = frame[y1_f:y2_f, x1_f:x2_f]
    face_resized = cv2.resize(face_resized, (112,112))
    # 
    if face_resized is None or face_resized.size == 0:
        return None, 0.0

    # h, w = face_resized.shape[:2]
    # if h < 30 or w < 30:
    #     return None, 0.0

    # # Blur check
    # if np.std(face_resized) < 15:
    #     return None, 0.0 

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




def face_embedding_without_detection(face): 
    # Normalize
    face_resize = cv2.resize(face, (112,112))
    face_norm = (face_resize - 127.5) / 128.0
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

def iou(boxA, boxB):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

def get_head_bbox(kpts):
    head_ids = [0,1,2,3,4]  # nose + eyes + ears
    xs = [kpts[i][0] for i in head_ids if kpts[i][0] > 0]
    ys = [kpts[i][1] for i in head_ids if kpts[i][1] > 0]

    if not xs or not ys:
        return None

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    # Expand a bit for stability
    pad = 10
    return [x1-pad, y1-pad, x2+pad, y2+pad]


VOTE_WINDOW = 1000
track_history = defaultdict(lambda: {
    "name": deque(maxlen=VOTE_WINDOW),
    "score": deque(maxlen=VOTE_WINDOW),
    "ref_img_id": deque(maxlen=VOTE_WINDOW),
    "box": deque(maxlen=VOTE_WINDOW),
    "h_id": deque(maxlen=VOTE_WINDOW),
})

def is_point_in_polygon(point, polygon):
    """Check if a point is inside a polygon"""
    # point should be (x, y) tuple
    # polygon is list of (x, y) points
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

def draw(frame, face_bbox, face_landmarks,human_bbox, human_keypoints, collocation, polygons):
    global face_human_tracker, frame_counter
    persons=[]
    scores=[] 
    ref_img_ids=[] 
    faces=[]
    objs=[]
    human_area=[]
    frame_counter += 1

    # =========================================================
    # STEP 1: Face recognition (run once per face)
    # =========================================================
    face_results = {}

    target_width = 640
    target_height = 540
    h, w = frame.shape[:2]

    scale_x = target_width / w
    scale_y = target_height / h
  
    output_frame = cv2.resize(frame, (target_width, target_height))

    for face_box, landmark in zip(face_bbox, face_landmarks):
        h, w = frame.shape[:2]

        x1 = max(0, min(int(face_box[0]), w - 1))
        y1 = max(0, min(int(face_box[1]), h - 1))
        x2 = max(0, min(int(face_box[2]), w - 1))
        y2 = max(0, min(int(face_box[3]), h - 1))

        fixed_box = [x1, y1, x2, y2]


        nose_x = landmark[2][0]
        nose_y = landmark[2][1]

        # eye (0) and nose (2)
        eye_nose_pts = [landmark[0], landmark[1]]

        xs = [p[0] for p in eye_nose_pts]
        ys = [p[1] for p in eye_nose_pts]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # The correct center condition:
        margin_y = (max_y - min_y)/5
        margin_x = (max_x - min_x)/5
        front_view = (
            (min_x + margin_x<= nose_x <= max_x - margin_x) and 
            (min_y + margin_y < nose_y and  nose_y >= max_y - margin_y)
        )

    
       

        if front_view:
            hcolor= (255, 0, 0)
            x_anchor = int((x1+ x2)/2 *scale_x)
            y_anchor = int(y2 * scale_y)
            if x_anchor > 0 and y_anchor > 0:
                cv2.circle(output_frame, (x_anchor, y_anchor), 3, hcolor, 2)

        else:
            hcolor = (0, 0, 255)
            
        for i, (x, y) in enumerate(landmark):
            if x > 0 and y > 0:
                cv2.circle(output_frame, (int(x * scale_x), int(y*scale_y)), 3, hcolor, -1)
    
        if front_view:
            embedding, rec_score = face_embedding(frame, fixed_box, landmark)
            x1_f, y1_f, x2_f, y2_f = map(int, face_box[:4])
            if embedding is None:
                person = "Unknown"
                rec_score = 0.0
                ref_img_id= None
            else:
                person, rec_score , ref_img_id= face_decoding(embedding, collocation)
            if person  == 'Unknown':
                # rec_score = face_box[4]
                rec_score = 0
                ref_img_id= None
            # if rec_score < 0.55:
            #     person= 'Unknown'
            #     ref_img_id= None
            
            face_results[face_box[5]] = {
                "box": [x1_f, y1_f, x2_f, y2_f],
                "person": person,
                "score": rec_score,
                'ref_img_id': ref_img_id,
                'track_id': face_box[5],
  
            }

        else:
            person= 'NoFace'       


  



    for human_idx, (human_box, pose) in enumerate(zip(human_bbox, human_keypoints)):

        kpts = pose["kpts"]

        head_box = get_head_bbox(kpts)
    
        # cv2.rectangle(output_frame, (int(head_box[0] *scale_x),int(head_box[1] *scale_y)), (int(head_box[2] *scale_x), int(head_box[3]*scale_y)), get_color_from_id(human_idx), 8)
        # if head_box is None:
        #     continue
        
        x1, y1, x2, y2 = map(int, pose['bbox'][:4])
        x_h_anch = int((x1+x2)/2)
        y_h_anch = int(y2)
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        HT = str(pose['track_id'])
        hcolor = get_color_from_id(human_idx)

        if x_h_anch >0 and y2>0:
            cv2.circle(output_frame, (int(x_h_anch* scale_x), int(y_h_anch*scale_y)), 3, hcolor, -1)
            circle_point = (x_h_anch, y_h_anch)

            areas_found = []
            for i, polygon in enumerate(polygons):
                if is_point_in_polygon(circle_point, polygon):
                    areas_found.append(i + 1) 

            if areas_found:
                area_names = [f"Area {a}" for a in areas_found]
                pos_area = f"{', '.join(area_names)}"


            else:
                pos_area = "OUT"
     


        best_face = None
        best_iou = 0
        for face_id, data in face_results.items():
            fx1, fy1, fx2, fy2 = data["box"]

            face_box = [fx1, fy1, fx2, fy2]

            overlap = iou(head_box, face_box)

            if overlap > best_iou :
                best_iou = overlap
                best_face = data
       


            

  
        
        continue_flage = False
        final_bbox_face = None
        history = track_history[HT]
        if best_face is not None and best_iou > 0.2:   # threshold

            if best_face["person"] not in  ['NoFace', 'Unknown']:
                history["name"].append(best_face["person"])
                history["score"].append(best_face["score"])
                history["ref_img_id"].append(best_face["ref_img_id"])
                history["box"].append(best_face["box"])

            hscore = best_face["score"]
            hscore = f'{hscore:.2f}'
            
            FT = str(best_face["track_id"])
          
            face_name = f' - HT:{HT} - FT:{FT} - RS:{hscore}' 

        else:
            face_name = f' - HT:{HT} ' 
            final_name = "Unknown"
            continue_flage = True

        # if continue_flage:
        #     continue
        name_counts = Counter(history["name"])
        known_counts = {name: c for name, c in name_counts.items() if name != "Unknown"}

        if known_counts:
            best_name = max(known_counts, key=known_counts.get)
            indices = [i for i, n in enumerate(history["name"]) if n == best_name]
            score_mean = mean([history["score"][i] for i in indices]) 
            if score_mean < 0.40:
                final_name = 'Unknown'
                final_score = 0
                final_ref_img_id = -1
                final_bbox_face = best_face["box"] if best_face is not None else None
            else:
                final_name = best_name
                best_idx = max(indices, key=lambda i: history["score"][i])
                final_score = history["score"][best_idx]
                final_ref_img_id = history['ref_img_id'][best_idx]
                final_bbox_face = history['box'][best_idx]

        else:
            final_name = "Unknown"
            final_score = 0
            final_ref_img_id = -1
            if best_face is not None:
                final_bbox_face = best_face.get("box") 

        label = pos_area + ' --- '+final_name + face_name
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), hcolor, 2)

        label_y = y1 - 5 if y1 > 30 else y2 + 20
        cv2.putText(
            output_frame,
            label,
            (x1 + 5, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            hcolor,
            1,
        )




    #     # draw skeleton lines
        for start_idx, end_idx in skeleton_edges:
            x1, y1 = kpts[start_idx]
            x2, y2 = kpts[end_idx]

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.line(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)



        objs.append(pose['track_id'])
        human_area.append(pos_area)
        persons.append(final_name)
        scores.append(final_score)
        ref_img_ids.append(final_ref_img_id)

        
            
        if final_bbox_face:
            if final_name !='Unknown':
                margin_x = 20
                margin_y = 20 
            else:
                margin_x = 0
                margin_y = 0 
            x1_face, y1_face, x2_face, y2_face = final_bbox_face   
            y1_m = max(y1_face - margin_y, 0)
            y2_m = min(y2_face + margin_y, frame.shape[0])
            x1_m = max(x1_face - margin_x, 0)
            x2_m = min(x2_face + margin_x, frame.shape[1])
            face_save = frame[y1_m:y2_m, x1_m:x2_m]
            final_face = cv2.resize(face_save, (112,112))

            faces.append(final_face)


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
        if person == 'Unknown':
            id_track = str(data['track_id'])
            person  = 'Unknown - ' + id_track

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

    
    # output_frame = draw_polygons(frame, loaded_polygon_points)
    # frontend_h , frontend_w = (540, 960)
    # output_frame = cv2.resize(output_frame, (frontend_w, frontend_h))
    return output_frame , persons, scores, faces , objs, ref_img_ids, human_area, [scale_x, scale_y]


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





def DeletePointVD(ref_img_id ) :

    try:
        # from Detection.TensorRT.infer import client
        
        # Create filter for ref_img_id
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="ref_img_id",
                    match=models.MatchValue(value=ref_img_id)
                )
            ]
        )
        
        # First, count how many points will be deleted (optional)
        count_result = client.count(
            collection_name=COLLECTION,
            count_filter=filter_condition
        )
        print(f"   Found {count_result.count} points in vector DB for ref_img_id: {ref_img_id}")
        
        # Delete the points
        delete_result = client.delete(
            collection_name=COLLECTION,
            points_selector=models.FilterSelector(
                filter=filter_condition
            )
        )
        
        print(f"✅ Deleted from vector database: {delete_result}")
        return True
        
    except Exception as e:
        print(f"⚠️ Error deleting from vector database: {e}")
        return False
    





def FaceEmbeddingWithoutDetection(face, person_name, ref_img_id=None):

    # cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    # cv2.namedWindow(f'Face {person_name}', cv2.WINDOW_NORMAL)
    # cv2.imshow("Frame", face)
    # cv2.waitKey(10000)

    embedding, rec_score  = face_embedding_without_detection(face)

    client.upsert(
    collection_name=COLLECTION,
    points=[
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={
                "person": person_name,
                "ref_img_id": ref_img_id
                    }
                )])
    print('\n\n\n\n\n\nsaved embed in vector database')



def FaceEmbedding(frame, person_name, ref_img_id=None):
    bbox_face, landmarks = faceDetection(frame,  trt_manager)

    for face_box, landmark in zip(bbox_face, landmarks):

        embedding, rec_score  = face_embedding(frame, face_box, landmark)

        client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "person": person_name,
                    "ref_img_id": ref_img_id
                        }
                    )])
        print('\n\n\n\n\n\nsaved embed in vector database with face detection')
                

        # cv2.imshow(f'Face {person_name}', face_crop)
        # cv2.waitKey(10000)

    # cv2.destroyAllWindows()
    



def FaceEmbeddingCropping(frame, person_name, ref_img_id=None):
    cv2.resize
    bbox_face, landmarks = faceDetection(frame,  trt_manager)

    try:
        embedding, rec_score , cropped_img = face_embedding_cropping(frame, bbox_face[0], landmarks[0])
    except Exception as e:
        print(e)
    client.upsert(
    collection_name=COLLECTION,
    points=[
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={
                "person": person_name,
                "ref_img_id": ref_img_id
                    }
                )])
    print('\n\n\n\n\n\nsaved embed in vector database with face detection')
    return cropped_img


def FaceDecoding(frame, person_name, COLLECTION):
    bbox_face, landmarks = faceDetection(frame,  trt_manager)
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow(f'Face {person_name}', cv2.WINDOW_NORMAL)
    cv2.imshow("Frame", frame)

    while True:
        for face_box, landmark in zip(bbox_face, landmarks):
        
            embedding, rec_score  = face_embedding(frame, face_box, landmark)
            person, rec_score , ref_img_id= face_decoding(embedding, COLLECTION)

            
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
COLLECTION = 'n19'

# if __name__ == "__main__":



#     if CAMERA_PROCESSING:
#         if USE_WEBCAM:
#             cap = cv2.VideoCapture(0)
#             cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#             cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#             if not cap.isOpened():
#                 print("❌ Cannot open webcam")
#                 exit()
#             print("✅ Webcam started")

#         else:
#             try:
#                 container = av.open(
#                     RTSP_URL,
#                     options={
#                         "rtsp_transport": "tcp",
#                         "max_delay": "1000000"
#                     }
#                 )
#                 print("✅ Connected to RTSP stream")
#             except Exception as e:
#                 print("❌ RTSP connection failed:", e)
#                 exit()
#         camera_processing(USE_WEBCAM, cap, container,  COLLECTION)
#     if FACE_ENCODING:
#         IMAGE_ROOT = "./qdrant_images"
#         root = Path(IMAGE_ROOT)

#         for person_dir in root.iterdir():
#             if not person_dir.is_dir():
#                 print('empttyyyyyyyyyyyyyyyyyyy')
#                 continue
#             person_name = person_dir.name
#             print(f"\n📁 Person: {person_name}")
#             for img_path in person_dir.glob("*.*"):
#                     img = cv2.imread(str(img_path))
#                     FaceEmbedding(img , f'{person_name}', COLLECTION)

#     if FACE_DECODING:
#         IMAGE_ROOT = "./qdrant_single"
#         root = Path(IMAGE_ROOT)

#         for person_dir in root.iterdir():
#             if not person_dir.is_dir():
#                 print('empttyyyyyyyyyyyyyyyyyyy')
#                 continue
#             person_name = person_dir.name
#             print(f"\n📁 Person: {person_name}")
#             for img_path in person_dir.glob("*.*"):
#                     img = cv2.imread(str(img_path))
#                     FaceDecoding(img , f'{person_name}', COLLECTION)


def FaceCropping(frame):
    bbox_face, landmarks = faceDetection(frame,  trt_manager)
    frame = face_align.norm_crop(frame, landmarks[0].astype(int))
    x1, y1, x2, y2 = map(int, bbox_face[0][:4])
    img = frame[y1:y2, x1:x2]
    return img



def draw_polygons(frame, polygons,scale):
    """
    Draw saved polygons on frame
    Args:
        frame: input image
        polygons: list of polygons, each polygon is list of (x,y) points
    Returns:
        frame with polygons drawn
    """
    if not polygons:
        return frame
    
    # Define colors for different polygons
    area_colors = [(0, 0, 255),    # Red
              (255, 0, 0),    # Blue  
              (0, 255, 0),    # Green
              (0, 255, 255),  # Yellow
              (255, 0, 255),  # Magenta
              (255, 255, 0)]  # Cyan
    
    # Get current frame dimensions
    frame_h, frame_w = frame.shape[:2]

    # Calculate scale factors
    scale_x = scale[0] 
    scale_y = scale[1] 

    for i, polygon in enumerate(polygons):
        if len(polygon) >= 3:
            # Convert points to numpy array
            pts = np.array(polygon, np.float32)
            
            # Scale points
            pts[:, 0] = pts[:, 0] * scale_x
            pts[:, 1] = pts[:, 1] * scale_y
            pts = pts.astype(np.int32)
            
            pts = pts.reshape((-1, 1, 2))
            
            # Draw polygon outline
            color = area_colors[i % len(area_colors)]
            cv2.polylines(frame, [pts], True, color, 2)


    
    return frame

def FrameProcessing(frame, loaded_polygon_points):
   
    bbox_face, landmarks = faceDetection(frame,  trt_manager)

    
    human_bbox, human_key= tracking_human_detected(frame_id, frame)
    frame , persons, scores, faces, objs, ref_img_ids, human_areas , out_scale= draw(frame, bbox_face , landmarks, human_bbox, human_key, COLLECTION, loaded_polygon_points)
    frame = draw_polygons(frame, loaded_polygon_points, out_scale)

    return {
    "frames": frame,
    "persons": persons,
    "scores":scores,
    "faces": faces,
    "objs": objs,
    "ref_img_ids": ref_img_ids,
    "area": human_areas,
}
