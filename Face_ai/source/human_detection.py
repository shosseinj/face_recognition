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



def detect_person_pose(frame):
    small_frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    H, W = frame.shape[:2]
    results = human_detector(small_frame, verbose=False)

    scale_x = W / INPUT_SIZE
    scale_y = H / INPUT_SIZE

    person_dets = []
    all_keypoints = []

    for result in results:

        if result.boxes is None or result.keypoints is None:
            continue

        boxes = result.boxes
        keypoints = result.keypoints

        # Move tensors to CPU once
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()
        kpts_xy = keypoints.xy.cpu().numpy()
        kpts_confs = keypoints.conf.cpu().numpy()

        xyxy[:, 0] = (xyxy[:, 0] * scale_x)  # x1
        xyxy[:, 1] = (xyxy[:, 1] * scale_y)  # y1
        xyxy[:, 2] = (xyxy[:, 2] * scale_x)  # x2
        xyxy[:, 3] = (xyxy[:, 3] * scale_y)  # y2

        kpts_xy[:,:, 0] *= scale_x
        kpts_xy[:,:, 1] *= scale_y

        for i in range(len(xyxy)):

            # Person class only
            if int(cls_ids[i]) != 0:
                continue

            conf = confs[i]
            if conf < 0.4:
                continue

            x1, y1, x2, y2 = xyxy[i]

            # person_dets.append([x1, y1, x2, y2, conf])

            all_keypoints.append({
                "bbox": [x1, y1, x2, y2],
                "kpts": kpts_xy[i],
                "kpts_conf": kpts_confs[i], 
                "score":conf
            })

    # if len(person_dets) > 0:
    #     person_dets = np.array(person_dets, dtype=np.float32)
    # else:
    #     person_dets = np.empty((0, 5), dtype=np.float32)

    return  all_keypoints


def tracking_human_detected(frame_id, frame):
    global person_dets , all_keypoints, FACE_KPTS
    # if frame_id % DETECT_EVERY == 0:
    all_keypoints = detect_person_pose(frame)

    # if person_dets is None or len(person_dets) == 0:
    #     person_dets = np.empty((0,5), dtype=np.float32)
    
    # online_targets = tracker.update(
    #     person_dets,
    #     frame.shape[:2],
    #     frame.shape[:2]
    # )
    # return online_targets , all_keypoints
    return  all_keypoints
   