from .retina_face import RetinaFace
import numpy as np
import cv2
import os
from datetime import datetime

INPUT_SIZE = 640  # RetinaFace expected input
CONF_THRESHOLD = 0.5

detector = RetinaFace()

save_face = False


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
    
    if save_face:
        SAVE_DIR = "saved_faces"
        os.makedirs(SAVE_DIR, exist_ok=True)

        for face_box, landmark in zip(bboxes, landmarks):
            x1_f, y1_f, x2_f, y2_f = map(int, face_box[:4])
            face = frame[y1_f:y2_f, x1_f:x2_f]

            if face.size == 0:
                continue

            # Unique filename using timestamp
            filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
            

            filepath = os.path.join(SAVE_DIR, filename)

            cv2.imwrite(filepath, face)



    return  bboxes , landmarks