





import json



import cv2
import asyncio
import numpy as np 
import av

def select_polygon(frame):
    """
    Interactive tool to define multiple polygons
    Returns: list of polygons, each polygon is a list of (x, y)
    """
    window_name = "Define Restricted Areas - ENTER: save | N: new | ESC: finish"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Get actual frame dimensions
    # h, w = frame.shape[:2]
    # cv2.resizeWindow(window_name, w, h)


    h, w = frame.shape[:2]
 
    cv2.resizeWindow(window_name, w, h)



    polygons = []
    current_points = []
    
    # Use a copy for drawing to avoid modifying original
    base_frame = frame.copy()

    def update_display():
        # Start fresh from base frame each time
        display = base_frame.copy()

        instructions = [
            "Left Click: add point",
            f"Current: {len(current_points)} points | Polygons: {len(polygons)}",
            "ENTER: save (need 3+)",
            "N: new polygon",
            "D: delete last",
            "C: clear",
            "ESC: finish"
        ]

        for i, text in enumerate(instructions):
            cv2.putText(display, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw completed polygons with distinct colors
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)]
        for i, poly in enumerate(polygons):
            if len(poly) >= 3:
                color = colors[i % len(colors)]
                pts = np.array(poly, np.int32)
                cv2.polylines(display, [pts], True, color, 3)
                cv2.putText(display, f"Area {i+1}", (poly[0][0], poly[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw current polygon
        for i, (x, y) in enumerate(current_points):
            cv2.circle(display, (x, y), 6, (0, 255, 255), -1)
            cv2.putText(display, str(i+1), (x+8, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if len(current_points) > 1:
            pts = np.array(current_points, np.int32)
            cv2.polylines(display, [pts], False, (0, 255, 0), 2)

        cv2.imshow(window_name, display)

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_points
        if event == cv2.EVENT_LBUTTONDOWN:
            # Clamp coordinates to frame bounds
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            current_points.append((x, y))
            print(f"Point {len(current_points)}: ({x}, {y})")
            update_display()

    cv2.setMouseCallback(window_name, mouse_callback)
    update_display()

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key in (13, 10):  # ENTER
            if len(current_points) >= 3:
                polygons.append(current_points.copy())
                print(f"✓ Saved Area {len(polygons)}: {len(current_points)} points")
                current_points.clear()
                update_display()
            else:
                print(f"⚠ Need 3+ points (have {len(current_points)})")

        elif key == ord('n'):  # New polygon
            if len(current_points) >= 3:
                polygons.append(current_points.copy())
                print(f"✓ Saved Area {len(polygons)}: {len(current_points)} points")
            current_points.clear()
            print("Starting new area...")
            update_display()

        elif key == ord('d') and current_points:  # Delete
            current_points.pop()
            update_display()

        elif key == ord('c'):  # Clear
            current_points.clear()
            update_display()

        elif key == 27:  # ESC
            if len(current_points) >= 3:
                polygons.append(current_points.copy())
                print(f"✓ Saved final Area {len(polygons)}")
            break

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)  # Flush
    print(f"\nTotal areas defined: {len(polygons)}")
    return polygons


class Config:
    RTSP_URL = "rtsp://Jafari:Asd@98500@192.168.110.20:554/Streaming/Channels/301" # saloon

    # RTSP_URL = "rtsp://Jafari:Asd@98500@192.168.110.14:554/Streaming/Channels/101"
    BASE_URL = "http://192.168.10.9:9000"
    CLIP_LENGTH = 100
    HALF_CLIP = CLIP_LENGTH // 2
    VOTE_WINDOW = 1000

config = Config()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0) 
    # cap = cv2.VideoCapture('http://192.168.50.19:8080/video') 
    cap = cv2.VideoCapture('./video6.mp4') 
    use_camera = cap.isOpened() and False #or True
    
    first_frame = True
    if not use_camera:
        cap.release()
        print("📹 Camera not available, trying RTSP stream...")
    
    while True:
        # cap = cv2.VideoCapture('./video1.mp4') 

        if not use_camera:
            # Add more robust RTSP options
            container = av.open(config.RTSP_URL, options={
                "rtsp_transport": "tcp",
                "stimeout": "5000000",  # 5 second timeout
                "max_delay": "500000",
                "buffer_size": "1024000",
                "fflags": "nobuffer+discardcorrupt",  # Discard corrupt frames
                "flags": "low_delay",
                "reorder_queue_size": "0"  # Disable reordering to avoid POC errors
            })
            
            # Get video stream
            video_stream = container.streams.video[0]
            print('✅ Connected to RTSP stream')
        



        asyncio.sleep(0.001)
    

        # Get frame from either camera or RTSP
        if use_camera:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Camera read failed")
                break
        else:
            try:
                # Add timeout for frame reading
                frame_packet = next(container.decode(video_stream))
                frame = frame_packet.to_ndarray(format="bgr24")
                
                # Validate frame
                if frame is None or frame.size == 0:
                    print("⚠️ Received empty frame, skipping...")
                    continue
                    
                # Reset error counter on successful frame
                consecutive_errors = 0
                
            except StopIteration:
                print("⚠️ Stream ended, attempting to reconnect...")
                break
            except av.AVError as e:
                consecutive_errors += 1
                print(f"⚠️ AV Error: {e} (attempt {consecutive_errors}/10)")
                
                if consecutive_errors >= 10:
                    print("❌ Too many consecutive errors, reconnecting...")
                    break
                
                # Skip bad frame and continue
                asyncio.sleep(0.1)
                continue
        

# Process the frame
        if first_frame:
            polygon_points = select_polygon(frame)
            print('polygon_points', polygon_points)
            with open('polygon_points.json', 'w') as f:
            
                json.dump(polygon_points, f)

            break
