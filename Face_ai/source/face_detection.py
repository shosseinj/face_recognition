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


def estimate_face_landmarks(bbox, frame_shape=None):
    """
    Estimate facial landmarks based on bounding box proportions.
    Returns 5 key points: [left_eye, right_eye, nose, left_mouth, right_mouth]
    
    Args:
        bbox: [x1, y1, x2, y2] face bounding box
        frame_shape: (h, w) optional for bounds checking
    
    Returns:
        list of 5 landmarks, each as [x, y]
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # Face ratios based on anthropological measurements
    # Eyes are typically at 1/3 of face height from top
    eye_y = y1 + height * 0.35
    
    # Eye positions (left and right eyes)
    left_eye_x = x1 + width * 0.35
    right_eye_x = x1 + width * 0.65
    
    # Nose is typically at 1/2 of face height
    nose_x = x1 + width * 0.5
    nose_y = y1 + height * 0.55
    
    # Mouth corners are typically at 2/3 of face height
    mouth_y = y1 + height * 0.75
    left_mouth_x = x1 + width * 0.35
    right_mouth_x = x1 + width * 0.65
    
    landmarks = [
        [int(left_eye_x), int(eye_y)],      # Left eye
        [int(right_eye_x), int(eye_y)],     # Right eye
        [int(nose_x), int(nose_y)],         # Nose
        [int(left_mouth_x), int(mouth_y)],  # Left mouth corner
        [int(right_mouth_x), int(mouth_y)]  # Right mouth corner
    ]
    

    
    return landmarks


import numpy as np

def assess_face_quality(face_image):
    """
    Assess if face is frontal and good quality
    Returns: quality_score, is_valid
    """
    if face_image is None or face_image.size == 0:
        return 0.0, False
    
    try:
        h, w = face_image.shape[:2]
        
        # Check minimum size
        if h < 30 or w < 30:
            return 0.0, False
        
        # Convert to grayscale for analysis
        if len(face_image.shape) == 3:
            if face_image.shape[2] == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image[:,:,0]  # Already single channel
        else:
            gray = face_image
        
        # Ensure we have valid image data
        if gray is None or gray.size == 0:
            return 0.0, False
        
        # 1. Symmetry check (frontal faces are more symmetric)
        try:
            mid_point = w // 2
            left_half = gray[:, :mid_point]
            right_half = cv2.flip(gray[:, mid_point:], 1)
            
            # Handle width mismatch
            min_width = min(left_half.shape[1], right_half.shape[1])
            if min_width > 0:
                left_half = left_half[:, :min_width]
                right_half = right_half[:, :min_width]
                symmetry_score = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
            else:
                symmetry_score = 0.0
        except Exception:
            symmetry_score = 0.0
        
        # 2. Contrast/sharpness check
        try:
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 100.0, 1.0)
        except Exception:
            sharpness_score = 0.0
        
        # 3. Image variance check
        try:
            std_val = np.std(gray.astype(float))
            variance_score = min(std_val / 50.0, 1.0)
        except Exception:
            variance_score = 0.0
        
        # 4. Brightness check (avoid too dark or too bright images)
        try:
            mean_brightness = np.mean(gray)
            # Optimal brightness range: 60-200
            if mean_brightness < 40 or mean_brightness > 230:
                brightness_score = 0.3
            elif mean_brightness < 60 or mean_brightness > 200:
                brightness_score = 0.7
            else:
                brightness_score = 1.0
        except Exception:
            brightness_score = 0.5
        
        # 5. Histogram spread check (well-exposed images have better histogram spread)
        try:
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            # Calculate entropy-like measure
            non_zero_hist = hist[hist > 0]
            if len(non_zero_hist) > 0:
                hist_spread_score = min(len(non_zero_hist) / 100.0, 1.0)
            else:
                hist_spread_score = 0.0
        except Exception:
            hist_spread_score = 0.0
        
        # 6. Edge density check (too few edges might indicate blur/noise)
        try:
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)
            edge_score = min(edge_density * 20, 1.0)  # Normalize
        except Exception:
            edge_score = 0.5
        
        # Combined quality score with updated weights
        quality_score = (
            symmetry_score * 0.25 +    # Symmetry is important for frontal faces
            sharpness_score * 0.25 +   # Sharpness for clarity
            variance_score * 0.15 +    # Overall image variance
            brightness_score * 0.15 +  # Proper exposure
            hist_spread_score * 0.1 +  # Histogram distribution
            edge_score * 0.1           # Edge information
        )
        
        # Updated thresholds with more nuanced validation
        is_valid = (
            quality_score > 0.35 and           # Overall quality threshold
            symmetry_score > 0.2 and           # Minimum symmetry
            sharpness_score > 0.15 and         # Minimum sharpness
            variance_score > 0.1 and           # Minimum variance
            brightness_score > 0.5 and         # Acceptable brightness
            quality_score > 0.4                # Final quality check
        )
        
        return quality_score, is_valid
        
    except Exception as e:
        print(f"Error in assess_face_quality: {e}")
        return 0.0, False


def faceDetectionBatch(frames, org_frames, face_detector, face_embedding, face_recognition, collocation):
    if not frames:
        return []
     
    results = face_detector(frames, verbose=False, batch=4)
    all_frame_results = []
        
    for idx, result in enumerate(results):
        tracked_bbox = []
        try:   
  
            xywh = result.boxes.xywh.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy()
            # keypoints = result.keypoints.xy.cpu().numpy()
     
            for i in range(len(xywh)):
                if int(cls_ids[i]) != 0:
                    continue

                x_center, y_center, width, height = xywh[i]
                x1 = x_center - (width / 2)
                y1 = y_center - (height / 2)
                x2 = x_center + (width / 2)
                y2 = y_center + (height / 2)
        
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                score = float(confs[i])

                # face_resized = face_align.norm_crop(frame, landmark.astype(int))
                orig_h, orig_w = org_frames[idx]['frame'].shape[:2]
                det_h, det_w = frames[idx].shape[:2]

                scale_x = orig_w / det_w
                scale_y = orig_h / det_h

                face_resized = org_frames[idx]['frame'][int(y1*scale_y):int(y2*scale_y), int(x1*scale_x):int(x2*scale_x)]

                # face_resized = frames[idx][y1:y2, x1:x2]
                face_resized = cv2.resize(face_resized, (112,112))
                if face_resized is None or face_resized.size == 0:
                    return None, 0.0

                h, w = face_resized.shape[:2] # Blur check
                if (h < 30 or w < 30) or (np.std(face_resized) < 15):
                    continue

                quality_score, img_valid = assess_face_quality(face_resized)

                if img_valid:
                    embedding, rec_score = face_embedding(face_resized)
                    if embedding is None:
                        person = "Unknown"
                        rec_score = 0.0
                        ref_img_id = None
                    else:
                        person, rec_score, ref_img_id = face_recognition(embedding, collocation)
                else:
                    person = 'No quality ------------------------'
                tracked_bbox.append([x1, y1, x2, y2, score, person, rec_score, ref_img_id])
            frame_data = {
                'cam_id': idx,
                'detections': tracked_bbox
            }
            all_frame_results.append(frame_data)
        except Exception as e:
            print(e)
    return all_frame_results




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


