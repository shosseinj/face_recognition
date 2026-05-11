import cv2
import numpy as np
import av
from ultralytics import YOLO
from pathlib import Path
from Face_ai.ByteTracker.byte_tracker import BYTETracker

# ==============================
# CONFIG
# ==============================

DETECT_EVERY = 1
INPUT_SIZE = 640
current_dir = Path(__file__).parent / 'engines'
# human_detector = YOLO(current_dir / "yolov8x_person_face.engine")
human_detector = YOLO(current_dir / "yolo26s-pose_batch4.engine")
frame_id = 0
all_keypoints = []
person_dets = np.empty((0,5), dtype=np.float32)
FACE_KPTS = {0, 1, 2, 3, 4}
# ByteTrack parameters
class Args:
    track_thresh = 0.4
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20 = False


tracker = BYTETracker(Args())
trackers = [BYTETracker(Args()) , BYTETracker(Args()), BYTETracker(Args()), BYTETracker(Args())]
# COCO Skeleton
SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (5, 6),
    (11, 12),
    (5, 11), (6, 12)
]


# ==============================
# Detection Function
# ==============================

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return interArea / (boxAArea + boxBArea - interArea + 1e-6)



# def detect_person_pose(frames):
#     # small_frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
#     res = []
#     for fr in frames:
#         small_frame = cv2.resize(fr, (INPUT_SIZE, INPUT_SIZE))
#         res.append(small_frame)
   
#     H, W = frame.shape[:2]

#     results = human_detector(res, verbose=False)

#     scale_x = W / INPUT_SIZE
#     scale_y = H / INPUT_SIZE

#     person_dets = []
#     all_keypoints = []
    
#     for result in results:

#         if result.boxes is None or result.keypoints is None:
#             continue

#         boxes = result.boxes
#         keypoints = result.keypoints

#         # Move tensors to CPU once
#         xyxy = boxes.xyxy.cpu().numpy()
#         confs = boxes.conf.cpu().numpy()
#         cls_ids = boxes.cls.cpu().numpy()
#         kpts_xy = keypoints.xy.cpu().numpy()
#         kpts_confs = keypoints.conf.cpu().numpy()

#         xyxy[:, 0] = (xyxy[:, 0] * scale_x)  # x1
#         xyxy[:, 1] = (xyxy[:, 1] * scale_y)  # y1
#         xyxy[:, 2] = (xyxy[:, 2] * scale_x)  # x2
#         xyxy[:, 3] = (xyxy[:, 3] * scale_y)  # y2

#         kpts_xy[:,:, 0] *= scale_x
#         kpts_xy[:,:, 1] *= scale_y

#         for i in range(len(xyxy)):

#             # Person class only
#             if int(cls_ids[i]) != 0:
#                 continue

#             conf = confs[i]
#             if conf < 0.4:
#                 continue

#             x1, y1, x2, y2 = xyxy[i]

#             # person_dets.append([x1, y1, x2, y2, conf])

#             all_keypoints.append({
#                 "bbox": [x1, y1, x2, y2],
#                 "kpts": kpts_xy[i],
#                 "kpts_conf": kpts_confs[i], 
#                 "score":conf
#             })

#     return  all_keypoints




def detect_person_pose(frames):
    """
    Detect person poses in a batch of frames
    
    Args:
        frames: List of frames (numpy arrays) in BGR format
    
    Returns:
        List of detection results for each frame
    """
    if not frames:
        return []
    
    INPUT_SIZE = 640
    
    # Preprocess all frames
    processed_frames = []
    original_sizes = []
    
    for fr in frames:
        # Resize frame
        small_frame = cv2.resize(fr, (INPUT_SIZE, INPUT_SIZE))
        # Convert BGR to RGB (YOLO expects RGB)
        small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        processed_frames.append(small_frame_rgb)
        original_sizes.append(fr.shape[:2])  # Store (H, W)
    
    # Run batch inference
    results = human_detector(processed_frames, verbose=False)
    
    all_results = []
    
    # Process each frame's results
    for idx, result in enumerate(results):
        H, W = original_sizes[idx]
        scale_x = W / INPUT_SIZE
        scale_y = H / INPUT_SIZE
        
        person_dets = []
        all_keypoints = []
        
        if result.boxes is None or result.keypoints is None:
            all_results.append([])
            continue
        
        boxes = result.boxes
        keypoints = result.keypoints
        
        # Move tensors to CPU once
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()
        kpts_xy = keypoints.xy.cpu().numpy()
        kpts_confs = keypoints.conf.cpu().numpy()
        
        # Scale bounding boxes back to original size
        xyxy[:, 0] = xyxy[:, 0] * scale_x  # x1
        xyxy[:, 1] = xyxy[:, 1] * scale_y  # y1
        xyxy[:, 2] = xyxy[:, 2] * scale_x  # x2
        xyxy[:, 3] = xyxy[:, 3] * scale_y  # y2
        
        # Scale keypoints back to original size
        if len(kpts_xy.shape) == 3:
            kpts_xy[:, :, 0] *= scale_x
            kpts_xy[:, :, 1] *= scale_y
        
        for i in range(len(xyxy)):
            # Person class only
            if int(cls_ids[i]) != 0:
                continue
            
            conf = confs[i]
            if conf < 0.4:
                continue
            
            x1, y1, x2, y2 = xyxy[i]
            
            all_keypoints.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "kpts": kpts_xy[i] if len(kpts_xy.shape) == 3 else kpts_xy,
                "kpts_conf": kpts_confs[i],
                "score": float(conf)
            })
        
        all_results.append(all_keypoints)
    
    return all_results


import numpy as np

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    union = areaA + areaB - inter
    return inter / union if union > 0 else 0


def tracking_human_detected( frame):
    global person_dets , all_keypoints, FACE_KPTS
    # if frame_id % DETECT_EVERY == 0:
    all_keypoints = detect_person_pose(frame)


    person_dets = []
    gen=[]

   
    person_dets = []
    gen = []

    for p in all_keypoints:
        temp = {}

        x1, y1, x2, y2 = p['bbox']
        score = float(p['score'])

        person_dets.append([x1, y1, x2, y2, score])

        temp['bbox'] = [x1, y1, x2, y2]
        temp['kpts'] = p['kpts']
        temp['kpts_conf'] = p['kpts_conf']
        temp['score'] = score

        gen.append(temp)

    if len(person_dets) == 0:
        person_dets = np.empty((0,5), dtype=np.float32)
    else:
        person_dets = np.array(person_dets, dtype=np.float32)


    online_targets = tracker.update(
        person_dets,
        frame.shape[:2],
        frame.shape[:2]
    )

    tracked_bbox = []

    for t in online_targets:
        score = float(t.score)

        x1 = t.tlwh[0]
        y1 = t.tlwh[1]
        x2 = t.tlwh[0] + t.tlwh[2]
        y2 = t.tlwh[1] + t.tlwh[3]

        tid = t.track_id

        tracked_bbox.append([x1, y1, x2, y2, score, tid])


    # match detections to tracks
    for g in gen:
        best_iou = 0
        best_track = None

        for tb in tracked_bbox:
            i = iou(g['bbox'], tb[:4])
            if i > best_iou:
                best_iou = i
                best_track = tb

        if best_track is not None:
            g['track_id'] = best_track[5]
        else:
            g['track_id'] = -1


    return tracked_bbox, gen

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


    
def tracking_human_detected_batch(frames):
    """
    Track detected humans and faces in a batch of frames
    Associates faces with persons
    """
    global trackers
    if not frames:
        return []
    
    INPUT_SIZE = 640
    
    # Preprocess all frames
    processed_frames = []
    original_sizes = []
    
    for fr in frames:
        small_frame = cv2.resize(fr, (INPUT_SIZE, INPUT_SIZE))
        small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        processed_frames.append(small_frame_rgb)
        original_sizes.append(fr.shape[:2])
    
    # Run batch inference
    results = human_detector(processed_frames, verbose=False, batch=4)
    # tracker_obj = init_tracker()
    
    all_frame_results = []
    
    for idx, result in enumerate(results):
        H, W = original_sizes[idx]
        scale_x = W / INPUT_SIZE
        scale_y = H / INPUT_SIZE
        
        # Get boxes (both person and face)
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()
        
        # Separate person and face detections
        person_dets = []      # For tracker
        face_dets = []        # Store faces with their person association
        all_keypoints = []
        tracked_persons = []
        

        keypoints = None
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()
        for i in range(len(xyxy)):
            class_id = int(cls_ids[i])
            x1, y1, x2, y2 = xyxy[i]
            conf = float(confs[i])


            person_keypoints = None
            if keypoints is not None and i < len(keypoints):
                person_keypoints = keypoints[i] 
            # Person (class 0)
            if class_id == 0:

                person_dets.append({
                'bbox': [x1, y1, x2, y2, conf],
                'conf': conf,
                'keypoints': person_keypoints  # Include keypoints
            })
                
    

    
        # Update tracker with person detections only
        if len(person_dets) == 0:
            person_dets_np = np.empty((0, 5), dtype=np.float32)
        else:
            person_dets_np = np.array([[p['bbox'][0], p['bbox'][1], 
                                     p['bbox'][2], p['bbox'][3], 
                                     p['conf']] for p in person_dets], dtype=np.float32)
        online_targets = trackers[idx].update(
            person_dets_np,
            (INPUT_SIZE, INPUT_SIZE),
            (INPUT_SIZE, INPUT_SIZE)
        )
        
        # # Associate faces with tracked persons
            
        best_keypoints = None
        best_iou = 0.0
        for t in online_targets:
            score = float(t.score)
            x1 = t.tlwh[0]
            y1 = t.tlwh[1]
            x2 = t.tlwh[0] + t.tlwh[2]
            y2 = t.tlwh[1] + t.tlwh[3]
            track_id = t.track_id
            for person in person_dets:
                px1, py1, px2, py2 = person['bbox'][:4]
                # Calculate IoU between tracked box and detection box
                iou = calculate_iou([x1, y1, x2, y2], [px1, py1, px2, py2])
                
                if iou > best_iou and iou > 0.5:  # IoU threshold
                    best_iou = iou
                    best_keypoints = person['keypoints']

            
            tracked_persons.append( [x1, y1, x2, y2, score, track_id, best_keypoints]
)
        if len(online_targets) ==0:
            tracked_persons.append([])


        


        frame_data = {
            'cam_id': idx,
            'humans': tracked_persons,
        }
        all_frame_results.append(frame_data)
    
    return all_frame_results


    """
    Track detected humans in a batch of frames using YOLO's built-in tracker
    with automatic rescaling to original frame sizes
    
    Args:
        frames: List of frames (numpy arrays)
    
    Returns:
        List of tracking results for each frame
    """
    if not frames:
        return []
    INPUT_SIZE = 640
 
    
    # Preprocess all frames
    processed_frames = []
    original_sizes = []
    
    for fr in frames:
        # Resize frame to model input size
        small_frame = cv2.resize(fr, (INPUT_SIZE, INPUT_SIZE))
        # Convert BGR to RGB (YOLO expects RGB)
        small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        processed_frames.append(small_frame_rgb)
        original_sizes.append(fr.shape[:2])  # Store (H, W)
    
    # Run batch tracking with persist=True to maintain track IDs across frames
    results = human_detector(processed_frames, verbose=False, batch=4)
    
    all_frame_results = []
  # Process each frame's results
    for idx, result in enumerate(results):
        H, W = original_sizes[idx]  # Original frame dimensions
        scale_x = W / INPUT_SIZE
        scale_y = H / INPUT_SIZE
        
        xywh = result.boxes.xywh.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy()
        keypoints = result.keypoints.xy.cpu().numpy()
        
        person_dets = []  # For tracker (INPUT_SIZE scale)
        person_keypoints = []  # For tracker (INPUT_SIZE scale)
        person_dets_original = []  # For output (original scale)
        
        for i in range(len(xywh)):
            if int(cls_ids[i]) != 0:
                continue
            
            # Convert xywh to xyxy (these are in INPUT_SIZE scale)
            x_center, y_center, width, height = xywh[i]
            x1 = x_center - (width / 2)
            y1 = y_center - (height / 2)
            x2 = x_center + (width / 2)
            y2 = y_center + (height / 2)
            score = float(confs[i])
            
            # Store for tracker (INPUT_SIZE scale - no scaling applied)
            person_dets.append([x1, y1, x2, y2, score])
            
            # Store keypoints at INPUT_SIZE scale
            kpts = keypoints[i].copy()
            person_keypoints.append(kpts)
            
            # For output (scale to original size)
            x1_orig = x1 * scale_x
            y1_orig = y1 * scale_y
            x2_orig = x2 * scale_x
            y2_orig = y2 * scale_y
            kpts_orig = kpts.copy()
            kpts_orig[:, 0] = kpts_orig[:, 0] * scale_x
            kpts_orig[:, 1] = kpts_orig[:, 1] * scale_y
            person_dets_original.append([x1_orig, y1_orig, x2_orig, y2_orig, score, kpts_orig])
        
        # Prepare for tracker (using INPUT_SIZE scale)
        if len(person_dets) == 0:
            person_dets_np = np.empty((0, 5), dtype=np.float32)
        else:
            person_dets_np = np.array(person_dets, dtype=np.float32)
        
        # Update tracker with INPUT_SIZE scale
        online_targets = trackers[idx].update(
            person_dets_np,
            (INPUT_SIZE, INPUT_SIZE),  # Use INPUT_SIZE, not original size
            (INPUT_SIZE, INPUT_SIZE)   # Use INPUT_SIZE, not original size
        )
        

        tracked_bbox = []
        for t in online_targets:
            score = float(t.score)
            x1 = t.tlwh[0]
            y1 = t.tlwh[1]
            x2 = t.tlwh[0] + t.tlwh[2]
            y2 = t.tlwh[1] + t.tlwh[3]
            tid = t.track_id
            

            tx1_orig = x1 * scale_x
            ty1_orig = y1 * scale_y
            tx2_orig = x2 * scale_x
            ty2_orig = y2 * scale_y
            # tracked_bbox.append([x1_orig, y1_orig, x2_orig, y2_orig, score, tid, kpts])


        # # Prepare tracked bounding boxes
        # tracked_bbox = []
        # for t in online_targets:
        #     score = float(t.score)
        #     # Tracker outputs at INPUT_SIZE scale
        #     x1 = t.tlwh[0]
        #     y1 = t.tlwh[1]
        #     x2 = t.tlwh[0] + t.tlwh[2]
        #     y2 = t.tlwh[1] + t.tlwh[3]
        #     tid = t.track_id
            
        #     # Find matching keypoints for this track
            matched_kpts = np.array([])
            best_iou = 0
            for det in person_dets_original:
                dx1, dy1, dx2, dy2, dscore, dkpts = det
                # Scale track coordinates to original for IOU calculation
    
                
                # Calculate IOU
                inter_x1 = max(tx1_orig, dx1)
                inter_y1 = max(ty1_orig, dy1)
                inter_x2 = min(tx2_orig, dx2)
                inter_y2 = min(ty2_orig, dy2)
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    box1_area = (tx2_orig - tx1_orig) * (ty2_orig - ty1_orig)
                    box2_area = (dx2 - dx1) * (dy2 - dy1)
                    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
                    
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        matched_kpts = dkpts
            
 
            
            tracked_bbox.append([x1_orig, y1_orig, x2_orig, y2_orig, score, tid, matched_kpts])
        
        frame_data = {
            'cam_id': idx,
            'detections': tracked_bbox
        }
        all_frame_results.append(frame_data)

    return all_frame_results