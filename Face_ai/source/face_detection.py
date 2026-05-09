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


def faceDetectionBatch(frames, trt_manager):
    results=[]
    for frame in frames:
        bbox_face, landmarks = faceDetection(frame, trt_manager)
        results.append( [bbox_face, landmarks])
    return results





def faceDetectionBatch_old(frames, trt_manager):
    """
    Process multiple frames in batch
    
    Args:
        frames: List of frames (numpy arrays)
        trt_manager: TensorRT manager for batch inference
        detector: Face detector instance (for post-processing)
    
    Returns:
        List of results for each frame
    """
    INPUT_SIZE = 640  # Make sure this matches your model input size
    
    batch_size = len(frames)
    if batch_size == 0:
        return []
    
    # Store original dimensions for each frame
    original_dims = []
    preprocessed_frames = []
    
    for frame in frames:
        if frame is None or frame.size == 0:
            preprocessed_frames.append(None)
            original_dims.append(None)
            continue
        
        # Store original dimensions
        H, W = frame.shape[:2]
        original_dims.append((H, W))
        
        # Resize and normalize
        img_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = img_resized.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        preprocessed_frames.append(img)
    
    # Batch detection
    valid_indices = [i for i, f in enumerate(preprocessed_frames) if f is not None]
    valid_frames = [preprocessed_frames[i] for i in valid_indices]
    
    if len(valid_frames) == 0:
        return [([], []) for _ in range(batch_size)]
    
    batch_input = np.stack(valid_frames, axis=0)
    print(f"📦 Processing batch: {len(valid_frames)} frames, shape={batch_input.shape}")
    
    # Run batch detection
    batch_outputs = trt_manager.detect_faces_batch(batch_input)
    
    batch_size_valid = batch_outputs[0].shape[0]
    
    # Define anchor counts for each feature level (stride 8, 16, 32)
    # For 640x640 input: 
    # stride 8: (80x80) = 6400 anchors × 2 (aspect ratios) = 12800
    # stride 16: (40x40) = 1600 anchors × 2 = 3200
    # stride 32: (20x20) = 400 anchors × 2 = 800
    anchor_counts = [12800, 3200, 800]
    split_indices = np.cumsum(anchor_counts[:-1])  # [12800, 16000]
    
    results = [([], []) for _ in range(batch_size)]
    
    for batch_idx in range(batch_size_valid):
        loc = batch_outputs[0][batch_idx]      # Shape: (16800, 4)
        conf = batch_outputs[1][batch_idx]     # Shape: (16800, 2)
        landmarks = batch_outputs[2][batch_idx] # Shape: (16800, 10)
        
        # Debug: Check confidence scores
        face_scores = conf[:, 1]
        print(f"Frame {batch_idx}: Max confidence = {face_scores.max():.4f}, "
              f"Mean = {face_scores.mean():.4f}, "
              f"Detections > 0.5 = {(face_scores > 0.5).sum()}, "
              f"Detections > 0.3 = {(face_scores > 0.3).sum()}")
        
        # Split into 9 outputs as expected by detector.detect()
        # Format: [score_stride8, score_stride16, score_stride32,
        #          bbox_stride8, bbox_stride16, bbox_stride32,
        #          landmark_stride8, landmark_stride16, landmark_stride32]
        
        split_outputs = []
        
        # 1. Split scores (face confidence) for each stride
        scores_full = conf[:, 1]  # Get face confidence (class 1)
        scores_levels = np.split(scores_full, split_indices)
        for i in range(3):
            # Reshape to (num_anchors, 1) as expected by detector
            split_outputs.append(scores_levels[i].reshape(-1, 1))
        
        # 2. Split bounding box predictions for each stride
        loc_levels = np.split(loc, split_indices)
        for i in range(3):
            split_outputs.append(loc_levels[i])
        
        # 3. Split landmark predictions for each stride
        landmarks_levels = np.split(landmarks, split_indices)
        for i in range(3):
            split_outputs.append(landmarks_levels[i])
        
        # Debug: Print shapes of split_outputs
        print(f"Split outputs shapes for frame {batch_idx}:")
        for i, out in enumerate(split_outputs):
            print(f"  [{i}]: {out.shape}")
        
        # Now call detector.detect() with the properly formatted 9 outputs
        try:
            bboxes, landmarks_points = detector.detect(
                split_outputs, 
                input_size=(INPUT_SIZE, INPUT_SIZE),
                confidence_threshold=0.5  # Adjust as needed
            )
            
            print(f"Detection result: {len(bboxes)} faces found")
            
            # Scale bounding boxes back to original frame size
            if len(bboxes) > 0:
                orig_h, orig_w = original_dims[valid_indices[batch_idx]]
                scale_x = orig_w / INPUT_SIZE
                scale_y = orig_h / INPUT_SIZE
                
                # Scale boxes (assuming bboxes format: [x1, y1, x2, y2, confidence])
                bboxes[:, [0, 2]] *= scale_x  # x1, x2
                bboxes[:, [1, 3]] *= scale_y  # y1, y2
                
                # Scale landmarks if present
                if len(landmarks_points) > 0 and landmarks_points.shape[1] == 5:
                    landmarks_points[:, :, 0] *= scale_x
                    landmarks_points[:, :, 1] *= scale_y
            
        except Exception as e:
            print(f"Error in detection: {e}")
            import traceback
            traceback.print_exc()
            bboxes = np.empty((0, 5), dtype=np.float32)
            landmarks_points = np.empty((0, 5, 2), dtype=np.float32)
        
        original_idx = valid_indices[batch_idx]
        results[original_idx] = (bboxes, landmarks_points)
    
    return results

def postprocess_detections(loc, conf, landmarks, input_size, original_size, 
                          confidence_threshold=0.5, nms_threshold=0.4):
    """
    Direct post-processing of model outputs
    
    Args:
        loc: (N, 4) bounding box offsets
        conf: (N, 2) confidence scores [background, face]
        landmarks: (N, 10) landmark coordinates
        input_size: (width, height) of model input
        original_size: (height, width) of original frame
    """
    import numpy as np
    
    # Get face confidence scores
    scores = conf[:, 1]
    
    # Filter by confidence threshold
    mask = scores > confidence_threshold
    if not np.any(mask):
        return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)
    
    loc = loc[mask]
    scores = scores[mask]
    landmarks = landmarks[mask]
    
    # Decode bounding boxes (this depends on your anchor configuration)
    # You'll need to implement anchor generation and decoding here
    
    # Apply NMS
    # keep = nms(bboxes, scores, nms_threshold)
    
    # Format bboxes as [x1, y1, x2, y2, confidence]
    bboxes_result = np.zeros((len(scores), 5), dtype=np.float32)
    # bboxes_result[:, :4] = decoded_bboxes
    # bboxes_result[:, 4] = scores
    
    # Format landmarks as (N, 5, 2)
    landmarks_result = landmarks.reshape(-1, 5, 2)
    
    return bboxes_result, landmarks_result