import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Load the exported engine
model_path = "yolov8x_person_face.engine"
model = YOLO(model_path)

def detect_person_face_batch(frames, conf_threshold=0.5):
    """
    Detect persons and faces in a batch of frames
    
    Args:
        frames: List of frames (numpy arrays in BGR format)
        conf_threshold: Confidence threshold for detections
    
    Returns:
        List of results for each frame with person and face detections
    """
    if not frames:
        return []
    
    INPUT_SIZE = 640
    
    # Preprocess frames (resize and convert to RGB)
    processed_frames = []
    original_sizes = []
    
    for frame in frames:
        # Resize to model input size
        resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        processed_frames.append(rgb)
        original_sizes.append(frame.shape[:2])  # (H, W)
    
    # Run inference
    results = model(processed_frames, verbose=False)
    
    all_results = []
    
    for idx, result in enumerate(results):
        H, W = original_sizes[idx]
        scale_x = W / INPUT_SIZE
        scale_y = H / INPUT_SIZE
        
        persons = []
        faces = []
        
        if result.boxes is not None:
            # Get detection data
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy()
            
            for i in range(len(xyxy)):
                # Scale coordinates to original frame size
                x1 = int(xyxy[i][0] * scale_x)
                y1 = int(xyxy[i][1] * scale_y)
                x2 = int(xyxy[i][2] * scale_x)
                y2 = int(xyxy[i][3] * scale_y)
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(confs[i]),
                    'class_id': int(cls_ids[i])
                }
                
                # Class 0 = person, Class 1 = face
                if int(cls_ids[i]) == 0:
                    detection['class'] = 'person'
                    persons.append(detection)
                elif int(cls_ids[i]) == 1:
                    detection['class'] = 'face'
                    faces.append(detection)
        
        all_results.append({
            'persons': persons,
            'faces': faces,
            'frame_id': idx
        })
    
    return all_results


def draw_detections(frame, detection_result):
    """
    Draw bounding boxes on frame
    
    Args:
        frame: Original frame
        detection_result: Result from detect_person_face_batch for this frame
    
    Returns:
        Frame with drawn bounding boxes
    """
    output = frame.copy()
    
    # Draw persons (blue)
    for person in detection_result['persons']:
        x1, y1, x2, y2 = person['bbox']
        conf = person['confidence']
        
        # Draw rectangle
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw label
        label = f"Person {conf:.2f}"
        cv2.putText(output, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Draw faces (green)
    for face in detection_result['faces']:
        x1, y1, x2, y2 = face['bbox']
        conf = face['confidence']
        
        # Draw rectangle
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"Face {conf:.2f}"
        cv2.putText(output, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return output


# ============================================
# USAGE EXAMPLE
# ============================================

# Load your frames


frame1 = cv2.imread('./img/1.png')
frame2 = cv2.imread('./img/2.png')
frame3 = cv2.imread('./img/3.png')
frame4 = cv2.imread('./img/4.png')
frames = [frame1, frame2, frame3, frame4]  # Your list of frames

# Run detection
results = detect_person_face_batch(frames, conf_threshold=0.5)

# Display results
for idx, result in enumerate(results):
    print(f"\nFrame {idx}: {len(result['persons'])} persons, {len(result['faces'])} faces")
    
    # Draw on frame
    output_frame = draw_detections(frames[idx], result)
    
    # Show or save
    cv2.imshow(f'Frame {idx}', output_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================
# FOR BATCH PROCESSING WITH CONCATENATED DISPLAY
# =