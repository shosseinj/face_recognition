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
human_detector = YOLO(current_dir / "yolo26s-pose.engine")
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


    
# def tracking_human_detected_batch(frames):
#     """
#     Track detected humans in a batch of frames
    
#     Args:
#         frames: List of frames (numpy arrays)
#         tracker: Tracker object
    
#     Returns:
#         List of tracking results for each frame
#     """
#     if not frames:
#         return []
    
#     INPUT_SIZE = 640
    
#     # Preprocess all frames
#     processed_frames = []
#     original_sizes = []
    
#     for fr in frames:
#         # Resize frame
#         small_frame = cv2.resize(fr, (INPUT_SIZE, INPUT_SIZE))
#         # Convert BGR to RGB (YOLO expects RGB)
#         small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
#         processed_frames.append(small_frame_rgb)
#         original_sizes.append(fr.shape[:2])  # Store (H, W)
    
#     # Run batch inference
#     results = human_detector.track(processed_frames, verbose=False)
    
#     # all_frame_results = []
    
#     # # Process each frame's results
#     # for idx, result in enumerate(results):
#     #     H, W = original_sizes[idx]
#     #     scale_x = W / INPUT_SIZE
#     #     scale_y = H / INPUT_SIZE
        
#     #     person_dets = []
#     #     gen = []
        
#     #     if result.boxes is None or result.keypoints is None:
#     #         all_frame_results.append({
#     #             'tracked_bbox': [],
#     #             'detections': []
#     #         })
#     #         continue
        
#     #     boxes = result.boxes
#     #     keypoints = result.keypoints
        
#     #     # Move tensors to CPU once
#     #     xyxy = boxes.xyxy.cpu().numpy()
#     #     confs = boxes.conf.cpu().numpy()
#     #     cls_ids = boxes.cls.cpu().numpy()
#     #     kpts_xy = keypoints.xy.cpu().numpy()
#     #     kpts_confs = keypoints.conf.cpu().numpy()
        
#     #     # Scale bounding boxes back to original size
#     #     xyxy[:, 0] = xyxy[:, 0] * scale_x  # x1
#     #     xyxy[:, 1] = xyxy[:, 1] * scale_y  # y1
#     #     xyxy[:, 2] = xyxy[:, 2] * scale_x  # x2
#     #     xyxy[:, 3] = xyxy[:, 3] * scale_y  # y2
        
#     #     # Scale keypoints back to original size
#     #     if len(kpts_xy.shape) == 3:
#     #         kpts_xy[:, :, 0] *= scale_x
#     #         kpts_xy[:, :, 1] *= scale_y
        
#     #     for i in range(len(xyxy)):
#     #         # Person class only
#     #         if int(cls_ids[i]) != 0:
#     #             continue
            
#     #         conf = confs[i]
#     #         if conf < 0.4:
#     #             continue
            
#     #         x1, y1, x2, y2 = xyxy[i]
            
#     #         # Add to person detections for tracker
#     #         person_dets.append([x1, y1, x2, y2, conf])
            
#     #         # Create detection dict
#     #         temp = {
#     #             'bbox': [float(x1), float(y1), float(x2), float(y2)],
#     #             'kpts': kpts_xy[i] if len(kpts_xy.shape) == 3 else kpts_xy[i],
#     #             'kpts_conf': kpts_confs[i],
#     #             'score': float(conf)
#     #         }
#     #         gen.append(temp)
        
#     #     # Convert to numpy array for tracker
#     #     if len(person_dets) == 0:
#     #         person_dets_np = np.empty((0, 5), dtype=np.float32)
#     #     else:
#     #         person_dets_np = np.array(person_dets, dtype=np.float32)
        
#     #     # Update tracker for this frame
#     #     online_targets = tracker.update(
#     #         person_dets_np,
#     #         frames[idx].shape[:2],
#     #         frames[idx].shape[:2]
#     #     )
        
#     #     # Prepare tracked bounding boxes
#     #     tracked_bbox = []
#     #     for t in online_targets:
#     #         score = float(t.score)
#     #         x1 = t.tlwh[0]
#     #         y1 = t.tlwh[1]
#     #         x2 = t.tlwh[0] + t.tlwh[2]
#     #         y2 = t.tlwh[1] + t.tlwh[3]
#     #         tid = t.track_id
            
#     #         tracked_bbox.append([x1, y1, x2, y2, score, tid])
        
#     #     # Match detections to tracks
#     #     for g in gen:
#     #         best_iou = 0
#     #         best_track = None
            
#     #         for tb in tracked_bbox:
#     #             iou_score = iou(g['bbox'], tb[:4])
#     #             if iou_score > best_iou:
#     #                 best_iou = iou_score
#     #                 best_track = tb
            
#     #         if best_track is not None and best_iou > 0.3:
#     #             g['track_id'] = best_track[5]
#     #         else:
#     #             g['track_id'] = -1
        
#     #     all_frame_results.append({
#     #         'tracked_bbox': tracked_bbox,
#     #         'detections': gen
#     #     })
    
#     return results





def tracking_human_detected_batch(frames):
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
    results = human_detector.track(processed_frames, persist=True, verbose=False, batch=4, tracker="bytetrack.yaml")
    
    all_frame_results = []
    
    # Process each frame's results
    for idx, result in enumerate(results):
        H, W = original_sizes[idx]  # Original frame dimensions
        scale_x = W / INPUT_SIZE
        scale_y = H / INPUT_SIZE
        
        # Initialize empty list for this frame
        tracked_bbox = []
        
        # Check if there are any detections
            # Get all data
        try:   
            track_ids = result.boxes.id
            if track_ids is not None:
                track_ids = track_ids.cpu().numpy()
                xywh = result.boxes.xywh.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy()
                keypoints = result.keypoints.xy.cpu().numpy()

                # keypoints[:, 0] = keypoints[:, 0] * scale_x  # Scale x coordinates
                # keypoints[:, 1] = keypoints[:, 1] * scale_y
                # xywh[:, 0] = xywh[:, 0] * scale_x
                # xywh[:, 1] = xywh[:, 1] * scale_y
                # Process each detection
                for i in range(len(xywh)):
                    # Check if it's a person (class 0)
                    if int(cls_ids[i]) != 0:
                        continue
                    
                    # Convert xywh to xyxy
                    x_center, y_center, width, height = xywh[i]
                    x1 = x_center - (width / 2)
                    y1 = y_center - (height / 2)
                    x2 = x_center + (width / 2)
                    y2 = y_center + (height / 2)
                            # Rescale to original frame size
                    x1 = x1 * scale_x
                    y1 = y1 * scale_y
                    x2 = x2 * scale_x
                    y2 = y2 * scale_y
                    


                    # Convert to integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    kpts = keypoints[i]
                    kpts[:,0]  = kpts[:,0]  * scale_x
                    kpts[:,1]  = kpts[:,1]  * scale_y

                    score = float(confs[i])
                    track_id = int(track_ids[i]) if track_ids is not None else -1
                    
                    # Append to list: [x1, y1, x2, y2, score, track_id]
                    tracked_bbox.append([x1, y1, x2, y2, score, track_id, kpts ])
    
            all_frame_results.append(tracked_bbox)
        except Exception as e:
            print(e)
    return all_frame_results


