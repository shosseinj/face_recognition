from ultralytics import YOLO

# Load a pretrained YOLO26n model
# model = YOLO("./yolo26s-pose.pt")
# path = model.export(
#     format="engine",
#     # batch=4,           # Fixed batch size of 4
#     # imgsz=640,         # Image size
#     #   dynamic=True, 
#     # half=True,         # FP16 for better performance
#     # workspace=8        # 8GB workspace
# )


model = YOLO("yolov12l-face.pt")
model.export(format="engine", 
             
             batch=4,           # Fixed batch size of 4
    imgsz=640,         # Image size
      dynamic=True, 

            #  dynamic=False, 
             nms=True, device="cuda:0")


# from ultralytics import YOLO
# import cv2
# import numpy as np
# import time

# # ============================================
# # Step 1: Export the model to TensorRT engine
# # ============================================
# print("=" * 50)
# print("Step 1: Exporting YOLO model to TensorRT engine")
# print("=" * 50)

# # Load a pretrained model
# model = YOLO("./yolo26s-pose.pt")

# # Export with dynamic batch support
# path = model.export(
#     format="engine",
#     batch=4,              # Maximum batch size (supports 1-4 with dynamic=True)
#     imgsz=640,            # Image size
#     dynamic=True,         # Enable dynamic batch (CRITICAL for variable batch sizes)
#     half=True,            # FP16 for better performance
#     workspace=4,          # 4GB workspace (adjust based on your GPU memory)
#     device=0,             # Use GPU 0
#     verbose=True          # Show export progress
# )

# print(f"\n✓ Model exported successfully to: {path}")
# print(f"✓ Export configuration: batch=4 (dynamic), imgsz=640, half=True")

# # ============================================
# # Step 2: Load the exported engine
# # ============================================
# print("\n" + "=" * 50)
# print("Step 2: Loading TensorRT engine")
# print("=" * 50)

# # Load the exported engine
# pose_model = YOLO(path)
# print(f"✓ TensorRT engine loaded successfully")

# # ============================================
# # Step 3: Test with different batch sizes
# # ============================================
# print("\n" + "=" * 50)
# print("Step 3: Testing with different batch sizes")
# print("=" * 50)

# # Create test images (random or dummy images)
# def create_test_image(size=640):
#     """Create a dummy test image"""
#     # Create a gradient image for testing
#     img = np.zeros((size, size, 3), dtype=np.uint8)
#     # Add some color gradient
#     for i in range(size):
#         img[i, :, 0] = int(255 * i / size)  # Red channel gradient
#         img[:, i, 1] = int(255 * i / size)  # Green channel gradient
#     img[:, :, 2] = 128  # Blue channel constant
#     return img

# # Test with different batch sizes
# batch_sizes = [1, 2, 3, 4]

# for batch_size in batch_sizes:
#     print(f"\n--- Testing batch size: {batch_size} ---")
    
#     # Create list of test images
#     test_images = [create_test_image() for _ in range(batch_size)]
    
#     # Time the inference
#     start_time = time.time()
    
#     try:
#         # Run inference
#         results = pose_model(test_images, verbose=False)
        
#         inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
#         print(f"✓ Success! Processed {batch_size} images")
#         print(f"  Inference time: {inference_time:.2f} ms")
#         print(f"  Average per image: {inference_time/batch_size:.2f} ms")
        
#         # Check results
#         for i, result in enumerate(results):
#             if result.keypoints is not None:
#                 keypoints = result.keypoints.data.cpu().numpy()
#                 print(f"  Image {i+1}: {keypoints.shape[1]} keypoints detected")
#             else:
#                 print(f"  Image {i+1}: No keypoints detected")
                
#     except Exception as e:
#         print(f"✗ Failed with batch size {batch_size}: {e}")

# # ============================================
# # Step 4: Test with actual frame processing
# # ============================================
# print("\n" + "=" * 50)
# print("Step 4: Testing with frame processing simulation")
# print("=" * 50)

# def process_frames(frames, model, input_size=640):
#     """
#     Process multiple frames with batch inference
    
#     Args:
#         frames: List of numpy arrays (H, W, 3) in BGR format
#         model: YOLO model
#         input_size: Target input size
#     """
#     # Preprocess frames
#     processed_frames = []
#     original_sizes = []
    
#     for frame in frames:
#         # Store original size
#         H, W = frame.shape[:2]
#         original_sizes.append((H, W))
        
#         # Resize
#         resized = cv2.resize(frame, (input_size, input_size))
        
#         # Convert BGR to RGB
#         resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
#         processed_frames.append(resized_rgb)
    
#     # Run batch inference
#     start_time = time.time()
#     results = model(processed_frames, verbose=False)
#     inference_time = (time.time() - start_time) * 1000
    
#     print(f"Processed {len(frames)} frames in {inference_time:.2f} ms")
#     print(f"Average: {inference_time/len(frames):.2f} ms per frame")
    
#     # Scale keypoints back to original sizes
#     for i, result in enumerate(results):
#         if result.keypoints is not None:
#             H, W = original_sizes[i]
#             keypoints = result.keypoints.data.cpu().numpy()
            
#             # Scale coordinates back
#             scale_x = W / input_size
#             scale_y = H / input_size
#             keypoints[:, :, 0] *= scale_x
#             keypoints[:, :, 1] *= scale_y
            
#             print(f"Frame {i+1}: {keypoints.shape[1]} keypoints (scaled to {W}x{H})")
    
#     return results

# # Create dummy frames
# dummy_frames = [
#     create_test_image(1920),  # 1080p
#     create_test_image(1280),  # 720p
#     create_test_image(640),   # SD
# ]

# print("Testing with 3 frames of different sizes...")
# results = process_frames(dummy_frames, pose_model)

# # ============================================
# # Step 5: Benchmark performance
# # ============================================
# print("\n" + "=" * 50)
# print("Step 5: Performance Benchmark")
# print("=" * 50)

# def benchmark_model(model, batch_sizes=[1, 2, 4], num_runs=10):
#     """Benchmark model performance"""
    
#     print(f"\nBenchmarking with {num_runs} runs per batch size:")
#     print("-" * 50)
    
#     for batch_size in batch_sizes:
#         # Create test batch
#         test_batch = [create_test_image() for _ in range(batch_size)]
        
#         # Warmup run
#         _ = model(test_batch, verbose=False)
        
#         # Benchmark
#         times = []
#         for _ in range(num_runs):
#             start = time.time()
#             _ = model(test_batch, verbose=False)
#             times.append((time.time() - start) * 1000)
        
#         avg_time = np.mean(times)
#         std_time = np.std(times)
#         fps = (batch_size / avg_time) * 1000
        
#         print(f"Batch size {batch_size}:")
#         print(f"  Avg: {avg_time:.2f} ms (±{std_time:.2f})")
#         print(f"  FPS: {fps:.1f} frames/second")
#         print(f"  Throughput: {batch_size * fps:.1f} items/second")

# benchmark_model(pose_model)

# # ============================================
# # Step 6: Save model info for future reference
# # ============================================
# print("\n" + "=" * 50)
# print("Step 6: Model Information Summary")
# print("=" * 50)

# # Try to get model input details
# print("\nModel Input Requirements:")
# print(f"  - Format: RGB images")
# print(f"  - Input size: 640x640 pixels")
# print(f"  - Batch support: Dynamic (1 to 4)")
# print(f"  - Data type: uint8 (0-255) or normalized float32")
# print(f"  - Channel order: RGB (not BGR)")

# print(f"\nModel Output:")
# print(f"  - Keypoints: 17 COCO keypoints")
# print(f"  - Bounding boxes: Available")
# print(f"  - Confidence scores: Included")

# print("\n" + "=" * 50)
# print("✓✓ All tests completed successfully! ✓✓")
# print("=" * 50)

# # ============================================
# # Bonus: Example usage with your frames
# # ============================================
# print("\n" + "=" * 50)
# print("Bonus: Example usage pattern")
# print("=" * 50)

# print("""
# # To use the model with your frames:

# from ultralytics import YOLO
# import cv2

# # Load model
# model = YOLO('./yolo26s-pose.engine')

# # For single frame
# frame = cv2.imread('image.jpg')
# results = model(frame, verbose=False)

# # For multiple frames (batch processing)
# frames = [cv2.imread(f'frame{i}.jpg') for i in range(4)]
# results = model(frames, verbose=False)

# # For video streaming
# cap = cv2.VideoCapture(0)
# frames_buffer = []
# while True:
#     ret, frame = cap.read()
#     frames_buffer.append(frame)
    
#     if len(frames_buffer) == 4:
#         results = model(frames_buffer, verbose=False)
#         # Process results
#         frames_buffer = []
# """)
