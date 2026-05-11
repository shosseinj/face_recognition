from Face_ai.source.human_detection import tracking_human_detected, tracking_human_detected_batch
import cv2
import av
from Face_ai.source.face_detection import faceDetection, faceDetectionBatch
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

save_face = False




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




from ultralytics import YOLO

face_detector = YOLO(str(Path(__file__).parent / 'source/engines' / "yolov12l-face_batch.engine"), task='detect')
# face_detector = YOLO(str(Path(__file__).parent / 'source/engines' / "yolov12l-face_batch.engine"))

class ModelManager:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self):
        if not self._initialized:
            self.trt_manager = TensorRTManager()
            self.current_dir = Path(__file__).parent / 'source/engines'

            self.trt_manager.initialize(
                # det_engine_path = self.current_dir / "retinaface_mv2_dynamic.engine",
                det_engine_path = self.current_dir / "retinaface_fp16.engine",
                rec_engine_path = self.current_dir / "arcface_fp16.engine"
                )

            class Args:
                track_thresh = 0.4
                track_buffer = 30
                match_thresh = 0.8
                aspect_ratio_thresh = 1.6
                min_box_area = 10
                mot20 = False

            self.tracker = BYTETracker(Args())
            self.cap= None
            self.container= None

            self.COLLECTION = 'n19'


    def FaceCropping(self,frame):
        bbox_face, landmarks = faceDetection(frame,  self.trt_manager)
        frame = face_align.norm_crop(frame, landmarks[0].astype(int))
        x1, y1, x2, y2 = map(int, bbox_face[0][:4])
        img = frame[y1:y2, x1:x2]
        return img
    
        
    def DeletePointVD(self, ref_img_id ) :

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
                collection_name=self.COLLECTION,
                count_filter=filter_condition
            )
            print(f"   Found {count_result.count} points in vector DB for ref_img_id: {ref_img_id}")
            
            # Delete the points
            delete_result = client.delete(
                collection_name=self.COLLECTION,
                points_selector=models.FilterSelector(
                    filter=filter_condition
                )
            )
            
            print(f"✅ Deleted from vector database: {delete_result}")
            return True
            
        except Exception as e:
            print(f"⚠️ Error deleting from vector database: {e}")
            return False
    





    def FaceEmbeddingWithoutDetection(self,face, person_name, ref_img_id=None):
        embedding, rec_score  = self.face_embedding_without_detection(face)
        client.upsert(
        collection_name=self.COLLECTION,
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

    def FaceEmbedding(self, frame, person_name, ref_img_id=None):
        bbox_face, landmarks = self.faceDetection(frame,  self.trt_manager)

        for face_box, landmark in zip(bbox_face, landmarks):

            embedding, rec_score  = self.face_embedding(frame, face_box, landmark)

            client.upsert(
            collection_name=self.COLLECTION,
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
                    
        



    def FaceEmbeddingCropping(self, frame, person_name, ref_img_id=None):
        cv2.resize
        bbox_face, landmarks = self.faceDetection(frame,  self.trt_manager)

        try:
            embedding, rec_score , cropped_img = self.face_embedding_cropping(frame, bbox_face[0], landmarks[0])
        except Exception as e:
            print(e)
        client.upsert(
        collection_name=self.COLLECTION,
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


    def FaceDecoding(self,frame, person_name, COLLECTION):
        bbox_face, landmarks = faceDetection(frame,  self.trt_manager)
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.namedWindow(f'Face {person_name}', cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", frame)

        while True:
            for face_box, landmark in zip(bbox_face, landmarks):
            
                embedding, rec_score  = self.face_embedding(frame, face_box, landmark)
                person, rec_score , ref_img_id= self.face_decoding(embedding, COLLECTION)

                
                print('person_name', person_name)
                print('rec name=', person)
                print('rec score=', rec_score)
                print('#' * 60)

                # cv2.imshow(f'Face {person_name}', face_crop)

            if cv2.waitKey(1) & 0xFF ==  ord('q'):
                break
        
        cv2.destroyAllWindows()
    


    def face_decoding(self, embedding, collocation):
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

        if score < 0.35:
            person = "Unknown"
        return person, float(score) , ref_img_id





    def face_embedding_cropping(self, frame, face_box, landmark): 
        
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

        embedding = self.trt_manager.recognize_face(input_tensor)

        if embedding is None:
            return None, 0.0

        emb = embedding[0]
        emb = emb / np.linalg.norm(emb)

        return emb, 1, face_resized




    def face_embedding(self, frame, face_box, landmark): 
        
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

        embedding = self.trt_manager.recognize_face(input_tensor)

        if embedding is None:
            return None, 0.0

        emb = embedding[0]
        emb = emb / np.linalg.norm(emb)

        return emb, 1
    
    def face_embedding_no_land(self, image): 
        # Normalize
        face_norm = (image - 127.5) / 128.0
        input_tensor = face_norm.transpose(2, 0, 1).astype(np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = np.ascontiguousarray(input_tensor)

        embedding = self.trt_manager.recognize_face(input_tensor)

        if embedding is None:
            return None, 0.0

        emb = embedding[0]
        emb = emb / np.linalg.norm(emb)

        return emb, 1




    def face_embedding_without_detection(self, face): 
        # Normalize
        face_resize = cv2.resize(face, (112,112))
        face_norm = (face_resize - 127.5) / 128.0
        input_tensor = face_norm.transpose(2, 0, 1).astype(np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = np.ascontiguousarray(input_tensor)

        embedding = self.trt_manager.recognize_face(input_tensor)

        if embedding is None:
            return None, 0.0

        emb = embedding[0]
        emb = emb / np.linalg.norm(emb)

        return emb, 1


    def draw(self, frame, face_bbox, face_landmarks,human_bbox, human_keypoints, collocation, polygons):
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
                embedding, rec_score = self.face_embedding(frame, fixed_box, landmark)
                x1_f, y1_f, x2_f, y2_f = map(int, face_box[:4])
                if embedding is None:
                    person = "Unknown"
                    rec_score = 0.0
                    ref_img_id= None
                else:
                    person, rec_score , ref_img_id= self.face_decoding(embedding, collocation)
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

        return output_frame , persons, scores, faces , objs, ref_img_ids, human_area, [scale_x, scale_y]

    def draw_batch(self, frames, face_detection_results, human_detection_results, collocation, polygons):
        """
        Draw batch of frames with face and human detection results
        
        Args:
            frames: List of frames
            face_detection_results: List of face detection results per frame (format: [{'cam_id': idx, 'detections': [...]}, ...])
            human_detection_results: List of human detection results per frame (format: [{'cam_id': idx, 'detections': [...]}, ...])
            collocation: Face collection for recognition
            polygons: Zone polygons
        
        Returns:
            Tuple of (output_frames, all_persons, all_scores, all_faces, all_objs, all_ref_img_ids, all_human_areas, out_scales)
        """
        global face_human_tracker, frame_counter
        draw_face = True
        draw_person = True
        output_frames = []
        all_persons = []
        all_scores = []
        all_faces = []
        all_objs = []
        all_ref_img_ids = []
        all_human_areas = []
        out_scales = []
        processed_frames=[]
        target_width = 640
        target_height = 540
        
        # Process each frame
        for frame_idx, frame in enumerate(frames):
            # Create a copy of the original frame
            annotated_frame = frame
            try:
            # Get face detections for this frame
                face_detections = []
                if face_detection_results and frame_idx < len(face_detection_results):
                    face_detections = face_detection_results[frame_idx].get('detections', [])
                
                # Get human detections for this frame
                human_detections = []
                if human_detection_results and frame_idx < len(human_detection_results):
                    human_detections = human_detection_results[frame_idx].get('humans', [])
                
                # Draw face detections
                if draw_face:
                    for det in face_detections:
                        if len(det) >= 5:
                            x1, y1, x2, y2, score, person_name = det[:6]
                            
                            # Color based on confidence
                            if score > 0.7:
                                color = (0, 255, 0)  # High confidence - green
                            elif score > 0.5:
                                color = (0, 255, 255)  # Medium confidence - yellow
                            else:
                                color = (0, 0, 255)  # Low confidence - red
                            
                            # Draw rectangle
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw label background
                            label = f"{person_name} - {score:.2f}"
                            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(annotated_frame, (x1, y1 - label_h - 5), 
                                        (x1 + label_w, y1), color, -1)
                            
                            # Draw label text
                            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw human detections
                if draw_person:
                    for det in human_detections:
                        if len(det) >= 5:
                            x1, y1, x2, y2 =  map(int, det[:4])  
                            score = det[4]
                            track_id = det[5]
                            
                            # Color based on confidence
                            if score > 0.7:
                                color = (255, 0, 0)  # High confidence - blue (different from faces)
                            elif score > 0.5:
                                color = (255, 255, 0)  # Medium confidence - cyan
                            else:
                                color = (255, 0, 255)  # Low confidence - magenta
                            
                            # Draw rectangle (thicker for humans)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw label background
                            label = f"person {score:.2f} - {track_id}"
                            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            # Position label above the box or below if above is out of frame
                            label_y = y1 - 5 if y1 - label_h > 0 else y2 + label_h + 5
                            cv2.rectangle(annotated_frame, (x1, label_y - label_h - 5), 
                                        (x1 + label_w, label_y), color, -1)
                           
                            # Draw label text
                            cv2.putText(annotated_frame, label, (x1, label_y - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            
                processed_frames.append(annotated_frame)

            except Exception as e:
                print(e)




            # h, w = frame.shape[:2]
            # scale_x = target_width / w
            # scale_y = target_height / h

            # output_frame = cv2.resize(frame, (target_width, target_height))

            # # Get human detection results for this frame
            # human_frame_result = human_detection_results[frame_idx] if frame_idx < len(human_detection_results) else None

            # # Draw persons with their faces
            # if human_frame_result and 'humans' in human_frame_result:
            #     tracked_persons = human_frame_result.get('humans', [])
                
            #     for person_data in tracked_persons:
            #         track_id = person_data.get('track_id', 0)
                    
            #         # Get human detection
            #         human_dets = person_data.get('human', [])
            #         face_dets = person_data.get('face', [])
                    
            #         # Draw human (person)
            #         if human_dets and len(human_dets) > 0:
            #             # human_dets is list of [x1, y1, x2, y2, score]
            #             for human in human_dets:
            #                 if len(human) >= 5:
            #                     x1, y1, x2, y2, score = human[:5]
                                
            #                     color = get_color_from_id(track_id)
                                
            #                     if draw_person:
            #                         # Scale coordinates
            #                         x1_scaled = int(x1 * scale_x)
            #                         y1_scaled = int(y1 * scale_y)
            #                         x2_scaled = int(x2 * scale_x)
            #                         y2_scaled = int(y2 * scale_y)
                                    
            #                         # Draw bounding box
            #                         cv2.rectangle(output_frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 2)
                                    
            #                         # Draw track_id label
            #                         label_y = y1_scaled - 5 if y1_scaled > 20 else y1_scaled + 20
            #                         cv2.putText(output_frame, f"{track_id}",
            #                                 (x1_scaled + 5, label_y),
            #                                 cv2.FONT_HERSHEY_SIMPLEX,
            #                                 0.5, color, 1)
                    
            #         # Draw faces associated with this person
            #         if face_dets and draw_face:
            #             for face in face_dets:
            #                 if len(face) >= 5:
            #                     fx1, fy1, fx2, fy2, face_score = face[:5]
                                
            #                     # Scale coordinates
            #                     fx1_scaled = int(fx1 * scale_x)
            #                     fy1_scaled = int(fy1 * scale_y)
            #                     fx2_scaled = int(fx2 * scale_x)
            #                     fy2_scaled = int(fy2 * scale_y)
                                
            #                     # Draw face with different color or same color with different thickness
            #                     cv2.rectangle(output_frame, (fx1_scaled, fy1_scaled), (fx2_scaled, fy2_scaled), (0, 255, 255), 1)
            #                 # if person == 'Unknown':
            #                 #     id_track = str(data['track_id'])
            #                 #     person = 'Unknown - ' + id_track
                            
            #                 # if y1 > 20:
            #                 #     cv2.putText(output_frame, f"{person} ({score:.2f})",
            #                 #             (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
            #                 #             0.5, color, 1)



            # if human_frame_result and 'detections' in human_frame_result:
            #     detections = human_frame_result.get('detections', [])
            #     cam_id = human_frame_result.get('cam_id', frame_idx)
                
            #     for i, detection in enumerate(detections):
            #         if len(detection) >= 5:  # x1, y1, x2, y2, score (optional track_id)
            #             x1, y1, x2, y2, score, track_id, kpts = detection
                        
            #             color = get_color_from_id(track_id)
                        
            #             if draw_person:
            #                 # Scale coordinates ONCE
            #                 x1_scaled = int(x1 * scale_x)
            #                 y1_scaled = int(y1 * scale_y)
            #                 x2_scaled = int(x2 * scale_x)
            #                 y2_scaled = int(y2 * scale_y)
                            
            #                 # Draw bounding box
            #                 cv2.rectangle(output_frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 2)
                            
            #                 # Draw track_id label (only once)
            #                 label_y = y1_scaled - 5 if y1_scaled > 20 else y1_scaled + 20
            #                 cv2.putText(output_frame, f"{track_id}",
            #                         (x1_scaled + 5, label_y), 
            #                         cv2.FONT_HERSHEY_SIMPLEX,
            #                         0.5, color, 1)
                        
            #             # Draw head box if keypoints exist
            #             if kpts is not None and len(kpts) > 0:
            #                 # Scale keypoints
            #                 kpts_scaled = kpts.copy()
            #                 kpts_scaled[:, 0] = kpts_scaled[:, 0] * scale_x
            #                 kpts_scaled[:, 1] = kpts_scaled[:, 1] * scale_y
                            
            #                 head_box = get_head_bbox(kpts_scaled)
            #                 if head_box is not None:
            #                     cv2.rectangle(output_frame, 
            #                                 (int(head_box[0]), int(head_box[1])), 
            #                                 (int(head_box[2]), int(head_box[3])), 
            #                                 color, 2)
            #     if x_h_anch > 0 and y2 > 0:
            #         cv2.circle(output_frame, (int(x_h_anch * scale_x), int(y_h_anch * scale_y)), 3, hcolor, -1)
                
            #     # Find best face match
            #     best_face = None
            #     best_iou = 0
            #     for face_id, data in face_results.items():
            #         fx1, fy1, fx2, fy2 = data["box"]
            #         face_box_coords = [fx1, fy1, fx2, fy2]
            #         overlap = iou(head_box, face_box_coords)
            #         if overlap > best_iou:
            #             best_iou = overlap
            #             best_face = data





        return processed_frames, all_persons, all_scores, all_faces, all_objs, all_ref_img_ids, all_human_areas, out_scales












            #  hossein jafari 
            
            # Extract human data for drawing
            # human_bbox = [person['bbox'] for person in persons]
            # human_scores = [person['score'] for person in persons]
            # human_keypoints = [person.get('keypoints', None) for person in persons]
            # human_track_ids = [person.get('track_id', -1) for person in persons]
            
            # # Face data for drawing
            # face_bbox = [face['bbox'] for face in faces]
            # face_scores = [face['score'] for face in faces]


            # frame_counter += 1
            
            # # =========================================================
            # # STEP 1: Face recognition (run once per face)
            # # =========================================================
            # face_results = {}
            
            # h, w = frame.shape[:2]
            # scale_x = target_width / w
            # scale_y = target_height / h
            
            # output_frame = cv2.resize(frame, (target_width, target_height))
            
            # # Process faces
            # for face_idx, (face_box, landmark) in enumerate(zip(face_bbox, face_landmarks)):
            #     x1 = max(0, min(int(face_box[0]), w - 1))
            #     y1 = max(0, min(int(face_box[1]), h - 1))
            #     x2 = max(0, min(int(face_box[2]), w - 1))
            #     y2 = max(0, min(int(face_box[3]), h - 1))
                
            #     fixed_box = [x1, y1, x2, y2]
                
            #     # Get track_id if available
            #     track_id = face_box[5] if len(face_box) > 5 else face_idx
                
            #     if len(landmark) >= 3:
            #         nose_x = landmark[2][0]
            #         nose_y = landmark[2][1]
                    
            #         eye_nose_pts = [landmark[0], landmark[1]]
            #         xs = [p[0] for p in eye_nose_pts]
            #         ys = [p[1] for p in eye_nose_pts]
                    
            #         min_x, max_x = min(xs), max(xs)
            #         min_y, max_y = min(ys), max(ys)
                    
            #         margin_y = (max_y - min_y)/5
            #         margin_x = (max_x - min_x)/5
            #         front_view = ((min_x + margin_x <= nose_x <= max_x - margin_x) and 
            #                     (min_y + margin_y < nose_y and nose_y >= max_y - margin_y))
            #     else:
            #         front_view = True
                
            #     if front_view:
            #         hcolor = (255, 0, 0)
            #         x_anchor = int((x1 + x2)/2 * scale_x)
            #         y_anchor = int(y2 * scale_y)
            #         if x_anchor > 0 and y_anchor > 0:
            #             cv2.circle(output_frame, (x_anchor, y_anchor), 3, hcolor, 2)
            #     else:
            #         hcolor = (0, 0, 255)
                
            #     # Draw landmarks
            #     if len(landmark) > 0:
            #         for x, y in landmark:
            #             if x > 0 and y > 0:
            #                 cv2.circle(output_frame, (int(x * scale_x), int(y * scale_y)), 3, hcolor, -1)
                
            #     if front_view:
            #         embedding, rec_score = self.face_embedding(frame, fixed_box, landmark)
            #         x1_f, y1_f, x2_f, y2_f = map(int, face_box[:4])
            #         if embedding is None:
            #             person = "Unknown"
            #             rec_score = 0.0
            #             ref_img_id = None
            #         else:
            #             person, rec_score, ref_img_id = self.face_decoding(embedding, collocation)
                    
            #         if person == 'Unknown':
            #             rec_score = 0
            #             ref_img_id = None
                    
            #         face_results[track_id] = {
            #             "box": [x1_f, y1_f, x2_f, y2_f],
            #             "person": person,
            #             "score": rec_score,
            #             'ref_img_id': ref_img_id,
            #             'track_id': track_id,
            #         }
            #     else:
            #         person = 'NoFace'
            
            # # Process humans
            # persons = []
            # scores = []
            # faces = []
            # objs = []
            # ref_img_ids = []
            # human_area = []
            
            # for human_idx, (human_box, kpts, track_id, human_score) in enumerate(zip(human_bbox, human_keypoints, human_track_ids, human_scores)):
            #     if len(kpts) == 0:
            #         continue
                    
            #     head_box = get_head_bbox(kpts)
                
            #     x1, y1, x2, y2 = map(int, human_box[:4])
            #     x_h_anch = int((x1 + x2)/2)
            #     y_h_anch = int(y2)
                
            #     x1_s = int(x1 * scale_x)
            #     y1_s = int(y1 * scale_y)
            #     x2_s = int(x2 * scale_x)
            #     y2_s = int(y2 * scale_y)
                
            #     HT = str(track_id)
            #     hcolor = get_color_from_id(human_idx)
                
            #     # Check zone
            #     circle_point = (x_h_anch, y_h_anch)
            #     areas_found = []
            #     for i, polygon in enumerate(polygons):
            #         if is_point_in_polygon(circle_point, polygon):
            #             areas_found.append(i + 1)
                
            #     if areas_found:
            #         area_names = [f"Area {a}" for a in areas_found]
            #         pos_area = f"{', '.join(area_names)}"
            #     else:
            #         pos_area = "OUT"
                
            #     if x_h_anch > 0 and y2 > 0:
            #         cv2.circle(output_frame, (int(x_h_anch * scale_x), int(y_h_anch * scale_y)), 3, hcolor, -1)
                
            #     # Find best face match
            #     best_face = None
            #     best_iou = 0
            #     for face_id, data in face_results.items():
            #         fx1, fy1, fx2, fy2 = data["box"]
            #         face_box_coords = [fx1, fy1, fx2, fy2]
            #         overlap = iou(head_box, face_box_coords)
            #         if overlap > best_iou:
            #             best_iou = overlap
            #             best_face = data
                
            #     final_bbox_face = None
            #     history = track_history[HT]
                
            #     if best_face is not None and best_iou > 0.2:
            #         if best_face["person"] not in ['NoFace', 'Unknown']:
            #             history["name"].append(best_face["person"])
            #             history["score"].append(best_face["score"])
            #             history["ref_img_id"].append(best_face["ref_img_id"])
            #             history["box"].append(best_face["box"])
                    
            #         hscore = best_face["score"]
            #         hscore = f'{hscore:.2f}'
            #         FT = str(best_face["track_id"])
            #         face_name = f' - HT:{HT} - FT:{FT} - RS:{hscore}'
            #     else:
            #         face_name = f' - HT:{HT}'
              
                
            #     # Determine final name from history
            #     name_counts = Counter(history["name"])
            #     known_counts = {name: c for name, c in name_counts.items() if name != "Unknown"}
                
            #     if known_counts:
            #         best_name = max(known_counts, key=known_counts.get)
            #         indices = [i for i, n in enumerate(history["name"]) if n == best_name]
            #         score_mean = mean([history["score"][i] for i in indices])
            #         if score_mean < 0.40:
            #             final_name = 'Unknown'
            #             final_score = 0
            #             final_ref_img_id = -1
            #             final_bbox_face = best_face["box"] if best_face is not None else None
            #         else:
            #             final_name = best_name
            #             best_idx = max(indices, key=lambda i: history["score"][i])
            #             final_score = history["score"][best_idx]
            #             final_ref_img_id = history['ref_img_id'][best_idx]
            #             final_bbox_face = history['box'][best_idx]
            #     else:
            #         final_name = "Unknown"
            #         final_score = 0
            #         final_ref_img_id = -1
            #         if best_face is not None:
            #             final_bbox_face = best_face.get("box")
                
            #     label = f'{pos_area} --- {final_name}{face_name}'
            #     cv2.rectangle(output_frame, (x1_s, y1_s), (x2_s, y2_s), hcolor, 2)
                
            #     label_y = y1_s - 5 if y1_s > 30 else y2_s + 20
            #     cv2.putText(output_frame, label, (x1_s + 5, label_y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, hcolor, 1)
                
            #     # Draw skeleton
            #     for start_idx, end_idx in skeleton_edges:
            #         if start_idx < len(kpts) and end_idx < len(kpts):
            #             x1_k, y1_k = kpts[start_idx]
            #             x2_k, y2_k = kpts[end_idx]
            #             if x1_k > 0 and y1_k > 0 and x2_k > 0 and y2_k > 0:
            #                 cv2.line(output_frame, 
            #                         (int(x1_k * scale_x), int(y1_k * scale_y)),
            #                         (int(x2_k * scale_x), int(y2_k * scale_y)),
            #                         (255, 0, 0), 2)
                
            #     objs.append(track_id)
            #     human_area.append(pos_area)
            #     persons.append(final_name)
            #     scores.append(final_score)
            #     ref_img_ids.append(final_ref_img_id)
                
            #     # Save face
            #     if final_bbox_face:
            #         if final_name != 'Unknown':
            #             margin_x = 20
            #             margin_y = 20
            #         else:
            #             margin_x = 0
            #             margin_y = 0
            #         x1_face, y1_face, x2_face, y2_face = final_bbox_face
            #         y1_m = max(y1_face - margin_y, 0)
            #         y2_m = min(y2_face + margin_y, frame.shape[0])
            #         x1_m = max(x1_face - margin_x, 0)
            #         x2_m = min(x2_face + margin_x, frame.shape[1])
            #         face_save = frame[y1_m:y2_m, x1_m:x2_m]
            #         if face_save.size > 0:
            #             final_face = cv2.resize(face_save, (112, 112))
            #             faces.append(final_face)
            
            # # Draw faces
            # for face_id, data in face_results.items():
            #     x1, y1, x2, y2 = data["box"]
            #     x1 = int(x1 * scale_x)
            #     y1 = int(y1 * scale_y)
            #     x2 = int(x2 * scale_x)
            #     y2 = int(y2 * scale_y)
            #     person = data["person"]
            #     score = data["score"]
            #     color = get_color_from_id(face_id)
                
            #     cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            #     if person == 'Unknown':
            #         id_track = str(data['track_id'])
            #         person = 'Unknown - ' + id_track
                
            #     if y1 > 20:
            #         cv2.putText(output_frame, f"{person} ({score:.2f})",
            #                 (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
            #                 0.5, color, 1)
            
            # output_frames.append(output_frame)
            # all_persons.append(persons)
            # all_scores.append(scores)
            # all_faces.append(faces)
            # all_objs.append(objs)
            # all_ref_img_ids.append(ref_img_ids)
            # all_human_areas.append(human_area)
            # out_scales.append([scale_x, scale_y])
        

    def draw_batch2(self, frames,face_detection_results, human_detection_results, collocation, polygons):
        """
        Draw batch of frames with person and face bounding boxes
        
        Args:
            frames: List of frames
            detection_results: List of detection results from process_detections_batch
            polygons: Zone polygons (optional)
        
        Returns:
            Tuple of (output_frames, all_persons, all_scores, all_faces, all_objs, all_ref_img_ids, all_human_areas, out_scales)
        """
        global frame_counter
        
        output_frames = []
        all_persons = []
        all_scores = []
        all_faces = []
        all_objs = []
        all_ref_img_ids = []
        all_human_areas = []
        out_scales = []
        
        target_width = 640
        target_height = 480
        
        for frame_idx, frame in enumerate(frames):
            # Get unified detection results for this frame
            frame_result = detection_results[frame_idx] if frame_idx < len(detection_results) else {'persons': [], 'faces': []}
            
            # Extract persons and faces
            persons = frame_result.get('persons', [])
            faces = frame_result.get('faces', [])
            
            h, w = frame.shape[:2]
            scale_x = target_width / w
            scale_y = target_height / h
            
            # Resize output frame
            output_frame = cv2.resize(frame, (target_width, target_height))
            
            # Lists to store results
            frame_persons = []
            frame_scores = []
            frame_faces = []
            frame_objs = []
            frame_ref_img_ids = []
            frame_human_areas = []
            
            # =========================================================
            # DRAW PERSONS (HUMANS)
            # =========================================================
            for person in persons:
                x1, y1, x2, y2 = person['bbox']
                score = person['score']
                track_id = person['track_id']
                
                # Scale to output size
                x1_s = int(x1 * scale_x)
                y1_s = int(y1 * scale_y)
                x2_s = int(x2 * scale_x)
                y2_s = int(y2 * scale_y)
                
                # Generate color based on track_id
                color = get_color_from_id(track_id)
                
                # Draw person bounding box
                cv2.rectangle(output_frame, (x1_s, y1_s), (x2_s, y2_s), color, 2)
                
                # Draw person label
                label = f"Person {track_id} ({score:.2f})"
                label_y = y1_s - 5 if y1_s > 30 else y2_s + 20
                cv2.putText(output_frame, label, (x1_s + 5, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Check zone (optional)
                x_center = int((x1 + x2) / 2)
                y_bottom = int(y2)
                circle_point = (x_center, y_bottom)
                areas_found = []
                for i, polygon in enumerate(polygons):
                    if is_point_in_polygon(circle_point, polygon):
                        areas_found.append(i + 1)
                
                if areas_found:
                    pos_area = f"Area {', '.join(map(str, areas_found))}"
                else:
                    pos_area = "OUT"
                
                # Store results
                frame_persons.append("Unknown")  # Will be updated by face recognition
                frame_scores.append(score)
                frame_objs.append(track_id)
                frame_human_areas.append(pos_area)
            
            # =========================================================
            # DRAW FACES
            # =========================================================
            for face in faces:
                x1, y1, x2, y2 = face['bbox']
                score = face['score']
                track_id = face['track_id']
                
                # Scale to output size
                x1_s = int(x1 * scale_x)
                y1_s = int(y1 * scale_y)
                x2_s = int(x2 * scale_x)
                y2_s = int(y2 * scale_y)
                
                # Generate color based on track_id
                color = get_color_from_id(track_id)
                
                # Draw face bounding box (thinner line)
                cv2.rectangle(output_frame, (x1_s, y1_s), (x2_s, y2_s), color, 1)
                
                # Draw face label
                label = f"Face {track_id} ({score:.2f})"
                if y1_s > 20:
                    cv2.putText(output_frame, label, (x1_s + 5, y1_s - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Store face for cropping
                face_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if face_crop.size > 0:
                    frame_faces.append(cv2.resize(face_crop, (112, 112)))
                else:
                    frame_faces.append(None)
                
                frame_ref_img_ids.append(None)
            
            # Append results for this frame
            output_frames.append(output_frame)
            all_persons.append(frame_persons)
            all_scores.append(frame_scores)
            all_faces.append(frame_faces)
            all_objs.append(frame_objs)
            all_ref_img_ids.append(frame_ref_img_ids)
            all_human_areas.append(frame_human_areas)
            out_scales.append([scale_x, scale_y])
        
        return output_frames, all_persons, all_scores, all_faces, all_objs, all_ref_img_ids, all_human_areas, out_scales



    def FrameProcessing(self, frames, org_frames, loaded_polygon_points):
        """
        Process frames one by one in a loop
        """
        human_detection_results = []
        # human_detection_results = tracking_human_detected(frames[0])
        face_detection_results = faceDetectionBatch(frames, org_frames, face_detector, self.face_embedding_no_land, self.face_decoding, self.COLLECTION)  # Single frame detection
        human_detection_results = tracking_human_detected_batch(frames)






      

        # face 
        # for frame in frames:
        #     bbox_face, landmarks = faceDetection(frame, self.trt_manager)  # Single frame detection
        #     face_detection_results.append({
        #                                 'bbox_face': bbox_face,
        #                                 'landmarks': landmarks
        #                             })






        # frame_out, persons, scores, faces, objs, ref_img_ids, human_areas, out_scale = self.draw(
        #     frames, face_detection_results,human_deteciton_results , 
        #     self.COLLECTION, loaded_polygon_points
        # )
        # frame_out = draw_polygons(frame_out, loaded_polygon_points, out_scale)
        frames_out, persons, scores, faces, objs, ref_img_ids, human_areas, out_scale = self.draw_batch(
        frames, face_detection_results, human_detection_results,
        self.COLLECTION, loaded_polygon_points
    )
        return {
            "frames": frames_out,
            "persons": persons,
            "scores": scores,
            "faces": faces,
            "objs": objs,
            "ref_img_ids": ref_img_ids,
            "area": human_areas,
        }
        
    
