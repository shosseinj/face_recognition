from .retina_face import RetinaFace
import numpy as np
import cv2
import os
from datetime import datetime
from Face_ai.ByteTracker.byte_tracker import BYTETracker

INPUT_SIZE = 640  # RetinaFace expected input
CONF_THRESHOLD = 0.5

detector = RetinaFace()

save_face = False

class Args:
    track_thresh = 0.4
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20 = False

tracker = BYTETracker(Args())

def faceDetection(frame, trt_manager):
    if frame is None or frame.size == 0:
        print("⚠️ Empty frame received")
        return frame

    H, W = frame.shape[:2]

    # Preprocess: resize to model input
    img_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img_resized.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img, dtype=np.float32)

    # Detect faces
    outputs = trt_manager.detect_faces(img)

    # Postprocess: get bboxes & landmarks
    bboxes, landmarks = detector.detect(outputs, input_size=(INPUT_SIZE, INPUT_SIZE))

    # Scale bboxes and landmarks back to original frame size
    scale_x = W / INPUT_SIZE
    scale_y = H / INPUT_SIZE

    bboxes[:, 0] = (bboxes[:, 0] * scale_x)  # x1
    bboxes[:, 1] = (bboxes[:, 1] * scale_y)  # y1
    bboxes[:, 2] = (bboxes[:, 2] * scale_x)  # x2
    bboxes[:, 3] = (bboxes[:, 3] * scale_y)  # y2

    landmarks[:, :, 0] *= scale_x
    landmarks[:, :, 1] *= scale_y


    if len(bboxes) > 0:
        bboxes_np = np.array(bboxes, dtype=np.float32)
    else:
        bboxes_np = np.empty((0, 5), dtype=np.float32)


    online_targets = tracker.update(
        bboxes_np,
        frame.shape[:2],
        frame.shape[:2]
    )
    tracked_bbox=[]
    for t in online_targets:


        score = float(t.score)

        x1 = t.tlwh[0]
        y1 = t.tlwh[1]
        x2 = t.tlwh[0] + t.tlwh[2]
        y2 = t.tlwh[1] + t.tlwh[3] 
        id = t.track_id
        tracked_bbox.append([x1, y1, x2, y2, score, id])


    return  tracked_bbox , landmarks
    # return  bboxes , landmarks