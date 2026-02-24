

import cv2
import numpy as np
import tensorrt as trt
from qdrant_client import QdrantClient
import pycuda.driver as cuda
import time
import gc
import os
from datetime import datetime
import requests
import av
import traceback
import atexit
from pathlib import Path
from qdrant_client.models import PointStruct


from Detection.TensorRT.yolox.tracker.byte_tracker import BYTETracker

INPUT_SIZE = 640
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

def force_context_cleanup():
    import pycuda.driver as cuda
    try:
        while True:
            cuda.Context.pop()
    except:
        pass

import atexit
atexit.register(force_context_cleanup)



def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div

def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)




import cv2
import numpy as np


def select_polygon(frame, max_display_width=1280):
    """
    Interactive tool to define multiple polygons.

    The window is scaled down for easier usage, but returned
    coordinates are in ORIGINAL frame resolution.

    Returns:
        list of polygons -> [[(x1,y1), (x2,y2), ...], ...]
    """

    window_name = "Define Restricted Areas - ENTER: save | N: new | ESC: finish"

    orig_h, orig_w = frame.shape[:2]

    # ---------- scale ----------
    scale = min(1.0, max_display_width / orig_w)
    preview_w = int(orig_w * scale)
    preview_h = int(orig_h * scale)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, preview_w, preview_h)

    # Resize only for visualization
    base_frame = cv2.resize(frame, (preview_w, preview_h))

    polygons = []          # stored in ORIGINAL resolution
    current_points = []    # stored in ORIGINAL resolution

    # ---------- helpers ----------
    def to_preview(points):
        """convert original -> preview"""
        return (np.array(points) * scale).astype(np.int32)

    # ---------- draw ----------
    def update_display():
        display = base_frame.copy()

        instructions = [
            "Left Click: add point",
            f"Current: {len(current_points)} | Polygons: {len(polygons)}",
            "ENTER: save (need 3+)",
            "N: new polygon",
            "D: delete last",
            "C: clear",
            "ESC: finish",
        ]

        for i, text in enumerate(instructions):
            cv2.putText(display, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0),
                  (0, 255, 255), (255, 0, 255)]

        # Draw saved polygons
        for i, poly in enumerate(polygons):
            if len(poly) >= 3:
                color = colors[i % len(colors)]
                pts = to_preview(poly)
                cv2.polylines(display, [pts], True, color, 3)
                cv2.putText(display, f"Area {i+1}",
                            (pts[0][0], max(20, pts[0][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw current polygon
        if current_points:
            pts = to_preview(current_points)

            for i, (x, y) in enumerate(pts):
                cv2.circle(display, (x, y), 6, (0, 255, 255), -1)
                cv2.putText(display, str(i + 1), (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if len(pts) > 1:
                cv2.polylines(display, [pts], False, (0, 255, 0), 2)

        cv2.imshow(window_name, display)

    # ---------- mouse ----------
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_points

        if event == cv2.EVENT_LBUTTONDOWN:
            # clamp inside preview
            x = max(0, min(x, preview_w - 1))
            y = max(0, min(y, preview_h - 1))

            # convert → original
            orig_x = int(x / scale)
            orig_y = int(y / scale)

            current_points.append((orig_x, orig_y))

            print(
                f"Point {len(current_points)}: preview=({x},{y}) -> original=({orig_x},{orig_y})"
            )

            update_display()

    cv2.setMouseCallback(window_name, mouse_callback)
    update_display()

    # ---------- keyboard ----------
    while True:
        key = cv2.waitKey(30) & 0xFF

        if key in (13, 10):  # ENTER
            if len(current_points) >= 3:
                polygons.append(current_points.copy())
                print(f"✓ Saved Area {len(polygons)}")
                current_points.clear()
                update_display()
            else:
                print("⚠ Need at least 3 points")

        elif key == ord("n"):  # new polygon
            if len(current_points) >= 3:
                polygons.append(current_points.copy())
                print(f"✓ Saved Area {len(polygons)}")
            current_points.clear()
            update_display()

        elif key == ord("d") and current_points:
            current_points.pop()
            update_display()

        elif key == ord("c"):
            current_points.clear()
            update_display()

        elif key == 27:  # ESC
            if len(current_points) >= 3:
                polygons.append(current_points.copy())
                print(f"✓ Saved final Area {len(polygons)}")
            break

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)

    print(f"\nTotal areas defined: {len(polygons)}")
    return polygons

import numpy as np
import matplotlib.pyplot as plt 




def distance2kps(points, distance, max_shape=None):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class RetinaFaceHossein:
    def __init__(self, model_file=None, session=None):
        self.taskname = 'detection'
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self._init_vars()

    def _init_vars(self):
        input_shape = [1, 3, '?', '?']
        self.input_shape = input_shape
        
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        
        self.input_name = 'input.1'
        self.output_names = ['448', '471', '494', '451', '474', '497', '454', '477', '500']
        
        self.input_mean = 127.5
        self.input_std = 128.0
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            print('Warning: CPU execution not supported in TRT version')
        nms_thresh = kwargs.get('nms_thresh', None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        det_thresh = kwargs.get('det_thresh', None)
        if det_thresh is not None:
            self.det_thresh = det_thresh
        input_size = kwargs.get('input_size', None)
        if input_size is not None:
            if self.input_size is not None:
                print('warning: det_size is already set in detection model, ignore')
            else:
                self.input_size = input_size

    def detect(self, net_outs, input_size, max_num=0, metric='default'):
        det_scale = 1
        scores_list = []
        bboxes_list = []
        kpss_list = []
        
        input_height = input_size[0]
        input_width = input_size[1]
        fmc = self.fmc
        
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx+fmc]
            bbox_preds = bbox_preds * stride
            
            if self.use_kps:
                kps_preds = net_outs[idx+fmc*2] * stride
            
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape((-1, 2))
                
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= self.det_thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        if len(scores_list) == 0:
            return np.zeros((0, 5)), None
            
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        
        if self.use_kps and len(kpss_list) > 0:
            kpss = np.vstack(kpss_list) / det_scale
        else:
            kpss = None
        
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        
        if self.use_kps and kpss is not None:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = 640 // 2, 640 // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0
            
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            
            if kpss is not None:
                kpss = kpss[bindex, :]
        
        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


def resize_to_fit(frame, max_width, max_height):
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    
    if scale == 1.0:
        return frame
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def frame_generator(container, webCamUsage=True, max_width=3000, max_height=3000):
    if webCamUsage:
        while True:
            ret, frame = container.read()
            if not ret:
                break
            frame = resize_to_fit(frame, max_width, max_height)
            yield frame
    else:
        for packet in container:
            frame = packet.to_ndarray(format="bgr24")
            frame = resize_to_fit(frame, max_width, max_height)
            yield frame




import cv2
import numpy as np



def recognize_face(embedding, client):
    search_result = client.query_points(
        collection_name='n5', 
        query=embedding.tolist(), 
        limit=1
    )
    
    person = "Unknown"
    score = 0.0
    
    if search_result and hasattr(search_result, 'points') and search_result.points:
        best_match = search_result.points[0]
        
        if hasattr(best_match, 'payload') and best_match.payload:
            person = best_match.payload.get("person", "Unknown")
        
        if hasattr(best_match, 'score'):
            score = best_match.score

    if score < 0.3:
        person = "Unknown"
    
    return person, score


class TensorRTManager:
    """Singleton manager for TensorRT resources with proper lifecycle"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TensorRTManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = False
            self.det_engine = None
            self.det_context = None
            self.rec_engine = None
            self.rec_context = None
            self.cuda_ctx = None
            self.stream = None
            self.det_bindings = []
            self.rec_bindings = []
            self.det_outputs = []
            self.rec_output = None
            
            # Register cleanup on exit
            # atexit.register(self.cleanup)
    
    def initialize(self, det_engine_path, rec_engine_path, det_input_shape=(1,3,640,640), rec_input_shape=(1,3,112,112)):
        """Initialize TensorRT engines with proper context"""
        try:
            print("Initializing TensorRT...")
            
            # Initialize CUDA context
            cuda.init()
            device = cuda.Device(0)
            self.cuda_ctx = device.make_context()
            print(f"CUDA Device: {device.name()}")
            
            # Create stream for async operations
            self.stream = cuda.Stream()
            
            logger = trt.Logger(trt.Logger.WARNING)
            
            # Load detection engine
            print(f"Loading detection engine: {det_engine_path}")
            with open(det_engine_path, "rb") as f:
                runtime = trt.Runtime(logger)
                self.det_engine = runtime.deserialize_cuda_engine(f.read())
            self.det_context = self.det_engine.create_execution_context()
            
            # Set input shape for detection
            det_input_name = self.det_engine.get_tensor_name(0)
            self.det_context.set_input_shape(det_input_name, det_input_shape)
            assert self.det_context.all_binding_shapes_specified, "Detection input shape not set correctly"
            
            # Load recognition engine
            print(f"Loading recognition engine: {rec_engine_path}")
            with open(rec_engine_path, "rb") as f:
                runtime = trt.Runtime(logger)
                self.rec_engine = runtime.deserialize_cuda_engine(f.read())
            self.rec_context = self.rec_engine.create_execution_context()
            
            # Set input shape for recognition
            rec_input_name = self.rec_engine.get_tensor_name(0)
            self.rec_context.set_input_shape(rec_input_name, rec_input_shape)
            assert self.rec_context.all_binding_shapes_specified, "Recognition input shape not set correctly"
            
            print("Engines loaded, allocating buffers...")
            
            # Allocate buffers
            self._allocate_detection_buffers()
            self._allocate_recognition_buffers()
            
            self._initialized = True
            print("TensorRT initialization complete")
            
        except Exception as e:
            print(f"Error initializing TensorRT: {e}")
            traceback.print_exc()
            self.cleanup()
            raise


    def _allocate_detection_buffers(self):
        """Allocate buffers for detection engine"""
        self.det_bindings = []  # Device buffers only
        self.det_outputs = []   # (host, device, shape, dtype) tuples
        
        # Input buffer
        input_name = self.det_engine.get_tensor_name(0)
        input_shape = self.det_context.get_tensor_shape(input_name)
        input_dtype = trt.nptype(self.det_engine.get_tensor_dtype(input_name))
        input_size = np.prod(input_shape) * np.dtype(input_dtype).itemsize
        d_input = cuda.mem_alloc(int(input_size))
        self.det_bindings.append(d_input)
        
        # Output buffers
        for i in range(1, self.det_engine.num_io_tensors):
            output_name = self.det_engine.get_tensor_name(i)
            output_shape = self.det_context.get_tensor_shape(output_name)
            output_dtype = trt.nptype(self.det_engine.get_tensor_dtype(output_name))
            # Handle dynamic shapes - use max size or actual size
            actual_shape = tuple(max(1, s) if s > 0 else input_shape[0] for s in output_shape)
            h_output = np.empty(actual_shape, dtype=output_dtype)
            d_output = cuda.mem_alloc(int(h_output.nbytes))
            self.det_outputs.append((h_output, d_output, actual_shape, output_dtype))

    def _allocate_recognition_buffers(self):
        """Allocate buffers for recognition engine"""
        self.rec_bindings = []
        input_name = self.rec_engine.get_tensor_name(0)
        input_shape = self.rec_context.get_tensor_shape(input_name)
        input_dtype = trt.nptype(self.rec_engine.get_tensor_dtype(input_name))
        input_size = np.prod(input_shape) * np.dtype(input_dtype).itemsize
        print(f"Rec input buffer size: {input_size} bytes, shape: {input_shape}")
        d_input = cuda.mem_alloc(int(input_size))
        self.rec_bindings.append(d_input)
        
        output_name = self.rec_engine.get_tensor_name(1)
        output_shape = self.rec_context.get_tensor_shape(output_name)
        output_dtype = trt.nptype(self.rec_engine.get_tensor_dtype(output_name))
        print(f"Rec output buffer size: {np.prod(output_shape) * np.dtype(output_dtype).itemsize} bytes, shape: {output_shape}")
        h_output = np.empty(output_shape, dtype=output_dtype)
        d_output = cuda.mem_alloc(int(h_output.nbytes))
        self.rec_output = (h_output, d_output)
        self.rec_bindings.append(d_output)

    def detect_faces(self, input_tensor):
        self.cuda_ctx.push()
        try:
            input_buffer = np.ascontiguousarray(input_tensor)
            
            # Get tensor names
            input_name = self.det_engine.get_tensor_name(0)
            
            # Set input shape
            self.det_context.set_input_shape(input_name, input_buffer.shape)
            
            # Copy to device buffer
            d_input = self.det_bindings[0]
            cuda.memcpy_htod_async(d_input, input_buffer, self.stream)
            
            # Set input tensor address (REQUIRED for enqueue_v3)
            self.det_context.set_tensor_address(input_name, int(d_input))
            
            # Set output tensor addresses
            output_buffers = []
            for i, (h_buf, d_buf, shape, dtype) in enumerate(self.det_outputs):
                output_name = self.det_engine.get_tensor_name(i + 1)
                self.det_context.set_tensor_address(output_name, int(d_buf))
                output_buffers.append(h_buf)
            
            # Execute
            self.det_context.execute_async_v3(stream_handle=self.stream.handle)
            self.stream.synchronize()
            
            # Copy outputs back
            for h_buf, d_buf, _, _ in self.det_outputs:
                cuda.memcpy_dtoh_async(h_buf, d_buf, self.stream)
            
            self.stream.synchronize()
            
            return [buf.copy() for buf in output_buffers]
        except Exception as e:
                print(f"Fatal error in dec: {e}")
                traceback.print_exc()
        finally:
            self.cuda_ctx.pop()

            


    def recognize_face(self, face_tensor):
            """Perform face recognition"""
            if not self._initialized:
                raise RuntimeError("TensorRT not initialized")
            self.cuda_ctx.push()
            try:
                face_data = np.ascontiguousarray(face_tensor)
                
                # Verify input size matches buffer
                d_input = self.rec_bindings[0]
                d_output = self.rec_bindings[1]
                
                # Use synchronous copy to avoid async issues
                cuda.memcpy_htod(d_input, face_data)
                
                self.rec_context.set_tensor_address("input.1", int(d_input))
                self.rec_context.set_tensor_address("683", int(d_output))
                self.rec_context.set_input_shape("input.1", face_tensor.shape)
                
                # Use sync execution for stability
                self.rec_context.execute_v2([int(d_input), int(d_output)])
                
                output = self.rec_output[0]
                cuda.memcpy_dtoh(output, d_output)
                
                return output.copy()
            except Exception as e:
                print(f"Fatal error in rec: {e}")
                traceback.print_exc()
    
            finally:
                
                self.cuda_ctx.pop()


    def save_face(self, face_tensor):
            """Perform face recognition"""
            if not self._initialized:
                raise RuntimeError("TensorRT not initialized")
            self.cuda_ctx.push()
            try:
                face_data = np.ascontiguousarray(face_tensor)
                
                # Verify input size matches buffer
                d_input = self.rec_bindings[0]
                d_output = self.rec_bindings[1]
                
                # Use synchronous copy to avoid async issues
                cuda.memcpy_htod(d_input, face_data)
                
                self.rec_context.set_tensor_address("input.1", int(d_input))
                self.rec_context.set_tensor_address("683", int(d_output))
                self.rec_context.set_input_shape("input.1", face_tensor.shape)
                
                # Use sync execution for stability
                self.rec_context.execute_v2([int(d_input), int(d_output)])
                
                output = self.rec_output[0]
                cuda.memcpy_dtoh(output, d_output)
                
                return output.copy()
            except Exception as e:
                print(f"Fatal error in rec: {e}")
                traceback.print_exc()
    
            finally:
                
                self.cuda_ctx.pop()


    
    def cleanup(self):
        """Proper cleanup of all resources"""
        print("\nCleaning up TensorRT resources...")
   
   
        try:
            for binding in self.rec_bindings + [b for _, b, _, _ in self.det_outputs] + self.det_bindings:
                if binding:
                    try:
                        cuda.mem_free(binding)
                    except:
                        pass
            
            self.det_context = None
            self.rec_context = None
            self.det_engine = None
            self.rec_engine = None
            self.stream = None
        finally:
            if self.cuda_ctx:
                try:
                    self.cuda_ctx.pop()
                    del self.cuda_ctx
                    self.cuda_ctx = None
                except:
                    pass
            self.det_bindings = []
            self.rec_bindings = []
            self.det_outputs = []
            self.rec_output = None   
            






import uuid



                

def SaveFace(frame, name, trt_manager,  client, COLLECTION):
    # print('********************************')
    if frame is None or frame.size == 0:
        print("⚠️ Empty frame received")
        return [], None
    
    output_detection = []
    
    # Get original frame dimensions
    orig_h, orig_w = frame.shape[:2]
    
    # Define minimum face size (e.g., 20x20 pixels)
    MIN_FACE_WIDTH = 20
    MIN_FACE_HEIGHT = 20
    
    # Preprocess for detection
    img_small = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    
    # Normalize image for detection
    img = img_small.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img, dtype=np.float32)
    
    # Detect faces
    outputs = trt_manager.detect_faces(img)
    
    # Process outputs using detector
    bboxes, landmarks = detector.detect(outputs, input_size=(INPUT_SIZE, INPUT_SIZE))
    
    if bboxes is None or len(bboxes) == 0:
        return output_detection, img_small
    
    # print('#################')
    # print(f'Detected {len(bboxes)} faces**********************')
    
    # Calculate scaling factors
    scale_x = orig_w / INPUT_SIZE
    scale_y = orig_h / INPUT_SIZE
    
    for i, bbox in enumerate(bboxes):
        if len(bbox) < 5:
            continue
        
        x1, y1, x2, y2, score = bbox[:5]
        
        # Filter by confidence score
        if score < CONF_THRESHOLD:
            continue
        
        # Scale coordinates back to original frame size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)
        
        # Calculate face dimensions
        face_width = x2_scaled - x1_scaled
        face_height = y2_scaled - y1_scaled
        
        # Filter out faces that are too small
        if face_width < MIN_FACE_WIDTH or face_height < MIN_FACE_HEIGHT:
            print(f"Skipping small face: {face_width}x{face_height}px")
            continue
        
        # Ensure bounds
        x1_scaled = max(0, x1_scaled)
        y1_scaled = max(0, y1_scaled)
        x2_scaled = min(orig_w, x2_scaled)
        y2_scaled = min(orig_h, y2_scaled)
        
        # Check if face region is valid
        if x2_scaled <= x1_scaled or y2_scaled <= y1_scaled:
            continue
        
        # Extract face with margin (improves recognition)
        margin_percent = 0.1
        margin_x = int(face_width * margin_percent)
        margin_y = int(face_height * margin_percent)
        
        x1_margin = max(0, x1_scaled - margin_x)
        y1_margin = max(0, y1_scaled - margin_y)
        x2_margin = min(orig_w, x2_scaled + margin_x)
        y2_margin = min(orig_h, y2_scaled + margin_y)
        
        # Extract face
        aligned = frame[y1_margin:y2_margin, x1_margin:x2_margin]
        
        # Check if extracted face is valid
        if aligned.size == 0:
            continue
        
        # Resize for recognition model
        aligned_resized = cv2.resize(aligned, (112, 112))
        
        # Quality check: ensure the face has reasonable content
        if np.std(aligned_resized) < 10:  # Too blurry/low contrast
            print("Face too blurry, skipping recognition")
            person = "Unknown"
            score = 0.0
        else:
            # Normalize for recognition
            aligned_face = (aligned_resized - 127.5) / 128.0
            input_tensor = aligned_face.transpose(2, 0, 1).astype(np.float32)
            input_tensor = np.expand_dims(input_tensor, axis=0)
            input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)
            
            # Recognize face
            embedding = trt_manager.recognize_face(input_tensor)
            if embedding is not None:
                emb = embedding[0] / np.linalg.norm(embedding[0])

        client.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb.tolist(),  # VERY IMPORTANT
                    payload={
                        "person": name,
                        "image": "fname",
                        "image_path": "file_path"
                    }
                )
            ]
        )


        # client.upsert(
        #             collection_name=COLLECTION,
        #             points=[{
        #                 "id": str(uuid.uuid4()),
        #                 "vector": emb.tolist(),
        #                 "payload": {
        #                     "person": name,
        #                     "image": 'fname',
        #                     "image_path": 'file_path'
        #                 }
        #             }]
        #         )




from qdrant_client import models



def RemoveRecord( names,  client, COLLECTION):
    """Remove records from the collection by person names"""
    if names:
        # If names is a string, convert to list
        if isinstance(names, str):
            names = [names]
        
        # Delete all matching names at once
        client.delete(
            collection_name=COLLECTION,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="person",
                        match=models.MatchAny(any=names)
                    )
                ]
            )
        )
        print(f"Deleted: {', '.join(names)}")
        return True
    
    return False







global_frame=None
webCam = False


trt_manager = TensorRTManager()
current_dir = Path(__file__).parent  # This is: /.../face_detection/Detection/TensorRT/

trt_manager.initialize(
    det_engine_path = current_dir / "retinaface_fp16.engine",
    rec_engine_path = current_dir / "arcface_fp16.engine"
    )

detector = RetinaFaceHossein()
detector.prepare(ctx_id=0)




from qdrant_client.models import VectorParams, Distance
# client = QdrantClient(path= current_dir /"qdrant_data")
client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "n5"
if not client.collection_exists(COLLECTION_NAME):
    print("Creating collection:", COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=512,
            distance=Distance.COSINE,   # common for face embeddings
        ),
    )
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

last_saved = {}
COOLDOWN = 3
frame_count = 0
CLEANUP_INTERVAL = 100

polygon_points = []
max_width = 3000
max_height = 3000
rec_threshold = 0.47
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4



def get_color_from_id(track_id):
    np.random.seed(track_id)
    color = tuple(np.random.randint(0, 255, 3).tolist())
    return color


from collections import defaultdict, deque, Counter

VOTE_WINDOW = 100
track_history = defaultdict(lambda: {
    "names": deque(maxlen=VOTE_WINDOW),
    "scores": deque(maxlen=VOTE_WINDOW),
})

SKELETON = [
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 6),              # shoulders
    (5, 11), (6, 12),    # torso
    (11, 12),            # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]




INPUT_SIZE = 640
CONF_THRESHOLD = 0.5
FACE_RECOGNITION_INTERVAL = 5  # Process face recognition every N frames
MAX_HISTORY = 30  # Max frames to keep in track history
IOU_THRESHOLD = 0.3  # IoU threshold for face-person matching

# Skeleton connections for pose estimation
SKELETON = [
    (5, 6),   # shoulders
    (5, 7),   # left arm
    (7, 9),   # left forearm
    (6, 8),   # right arm
    (8, 10),  # right forearm
    (5, 11),  # left hip
    (6, 12),  # right hip
    (11, 12), # hips
    (11, 13), # left thigh
    (13, 15), # left leg
    (12, 14), # right thigh
    (14, 16)  # right leg
]

# Global track history (consider moving to a class for better organization)
track_history = {}
frame_counter = 0

def get_color_from_id(track_id):
    """Generate consistent color for track ID"""
    np.random.seed(track_id)
    return tuple(map(int, np.random.randint(0, 255, 3).tolist()))

def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0





# Global smoothing buffers (simple approach)
bbox_buffer = {}  # {track_id: deque of bboxes}
MAX_BUFFER_SIZE = 50  # Number of frames to smooth over
SMOOTHING_FACTOR = 0.7  # 0.0-1.0, higher = more smoothing

def smooth_detections(person_dets, track_id, current_bbox):
    """
    Simple exponential smoothing for real-time bbox stabilization
    
    Args:
        person_dets: list of detections to append to
        track_id: current track ID
        current_bbox: [x1, y1, x2, y2, score] from YOLO
    
    Returns:
        smoothed bbox
    """
    global bbox_buffer
    
    # Initialize buffer for new track
    if track_id not in bbox_buffer:
        bbox_buffer[track_id] = {
            'smoothed': current_bbox[:4],  # [x1,y1,x2,y2]
            'raw': deque(maxlen=MAX_BUFFER_SIZE)
        }
    
    # Add current detection to raw buffer
    bbox_buffer[track_id]['raw'].append(current_bbox[:4])
    
    # Apply exponential smoothing
    if len(bbox_buffer[track_id]['raw']) > 1:
        prev_smoothed = bbox_buffer[track_id]['smoothed']
        current_raw = current_bbox[:4]
        
        # Exponential moving average
        smoothed = []
        for i in range(4):
            # Weighted average: more weight to previous smoothed value
            val = SMOOTHING_FACTOR * prev_smoothed[i] + (1 - SMOOTHING_FACTOR) * current_raw[i]
            smoothed.append(int(val))
        
        bbox_buffer[track_id]['smoothed'] = smoothed
    else:
        smoothed = list(map(int, current_bbox[:4]))
        bbox_buffer[track_id]['smoothed'] = smoothed
    
    # Append smoothed detection
    person_dets.append(smoothed + [current_bbox[4]])
    
    return smoothed




def DecodeFace(frame, trt_manager, client, COLLECTION,  tracker= None):
    if frame is None or frame.size == 0:
        print("⚠️ Empty frame received")
        return {
            'tracked_persons': [],
            'recognized_faces': [],
            'frame': None,
            'error': 'Empty frame'
        }
    
    
    output_detection = []
    frame_copy = frame.copy()


    print('Face Detection Started')

    # Preprocess for detection
    img_small = cv2.resize(frame_copy, (INPUT_SIZE, INPUT_SIZE))
    
    # Normalize image for detection
    img = img_small.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img, dtype=np.float32)
    
    # Detect faces
    
    outputs = trt_manager.detect_faces(img)
    
    # Process outputs using detector
    bboxes, landmarks = detector.detect(outputs, input_size=(INPUT_SIZE, INPUT_SIZE))
    
    # if bboxes is None or len(bboxes) == 0:
    #     return output_detection, img_small
    
    print('Face Detection Completed')


  

    # ==================== FACE RECOGNITION ====================
    recognized_faces = []
    if bboxes is not None and len(bboxes) > 0:
        for fb in bboxes:
            fx1, fy1, fx2, fy2 = map(int, fb[:4])
            score = fb[4]

            if score < CONF_THRESHOLD:
                continue

            aligned = img_small[fy1:fy2, fx1:fx2]
            if aligned.size == 0:
                continue

            aligned_resized = cv2.resize(aligned, (112, 112))
            cv2.rectangle(img_small, (fx1, fy1), (fx2, fy2), (255, 192,203), 2)
            aligned_face = (aligned_resized - 127.5) / 128.0
            input_tensor = aligned_face.transpose(2, 0, 1).astype(np.float32)
            input_tensor = np.expand_dims(input_tensor, axis=0)

            embedding = trt_manager.recognize_face(input_tensor)

            if embedding is not None:
                emb = embedding[0] / np.linalg.norm(embedding[0])
                person, rec_score = recognize_face(emb, client)
            else:
                person, rec_score = "Unknown", 0.0

            recognized_faces.append({
                "bbox": (fx1, fy1, fx2, fy2),
                "name": person,
                'score': rec_score 
            })

    print('Face Recognition Completed')

    # ==================== HUMAN DETECTION ====================
    try:
        human_output = human_detector(img_small)
    except Exception as e:
        print(f"Human detection error: {e}")
        human_output = [] 
    print('Human detection Completed')

    person_dets_list = []
 # Optimized for real-time processing
    for result in human_output:
        boxes = result.boxes
        keypoints = result.keypoints

        if boxes is None or keypoints is None:
            continue

        # Move to CPU once and cache
        kpts = keypoints.xy.cpu().numpy()
        kpts_conf = keypoints.conf.cpu().numpy()
        
        # Pre-allocate color array for this batch
        boxes_np = boxes.xyxy.cpu().numpy()
        scores_np = boxes.conf.cpu().numpy()

        for i in range(len(boxes_np)):

            # ---- Person Box (fast unpacking) ----
            x1, y1, x2, y2 = boxes_np[i]
            score = scores_np[i]
       

            # Fast integer conversion
            x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
      
            person_dets_list.append([x1, y1, x2, y2, score])
            # Get color (fast lookup)
            color = get_color_from_id(i)

            # ===== FAST DRAWING (minimal overhead) =====
            # Draw bounding box (thinner line for speed)
            # cv2.rectangle(img_small, (x1_int, y1_int), (x2_int, y2_int), color, 1)
            
   
            # ---- Fast Keypoint Drawing (skip if low confidence) ----
            if score > 0.5:  # Only draw keypoints for good detections
                # Draw keypoints (vectorized approach)
                kpt_i = kpts[i]
                conf_i = kpts_conf[i]
                
                # Draw keypoints (skip nose + eyes)
                for idx in range(5, 17):  # Start from shoulder joints
                    if conf_i[idx] > 0.3:
                        x, y = int(kpt_i[idx][0]), int(kpt_i[idx][1])
                        cv2.circle(img_small, (x, y), 2, color, -1)  # Smaller circles

                # Draw skeleton (only main connections for speed)
                fast_skeleton = [(5, 6), (5, 7), (6, 8), (11, 12), (11, 13), (12, 14)]
                for joint1, joint2 in fast_skeleton:
                    if (conf_i[joint1] > 0.3 and conf_i[joint2] > 0.3):
                        x1_line, y1_line = int(kpt_i[joint1][0]), int(kpt_i[joint1][1])
                        x2_line, y2_line = int(kpt_i[joint2][0]), int(kpt_i[joint2][1])
                        cv2.line(img_small, (x1_line, y1_line), (x2_line, y2_line), color, 1)
       
        print('Pose Completed')
        if len(person_dets_list)> 0:
            person_dets = np.array(person_dets_list)
            online_targets = tracker.update(
                                    person_dets,
                                    img_small.shape[:2],
                                    img_small.shape[:2]
                                )
        else:
            online_targets = []

        current_objs = set()

        for target in online_targets:

            tlwh = target.tlwh
            track_id = target.track_id
            current_objs.add(track_id)

            x1, y1, w, h = tlwh
            x1, y1, w, h = int(x1), int(y1), int(w), int(h)
            x2, y2 = x1 + w, y1 + h

            human_name = "Unknown"

            # check if any recognized face is inside this tracked person
            # for face in recognized_faces:
            #     fx1, fy1, fx2, fy2 = face["bbox"]

            #     if fx1 >= x1 and fy1 >= y1 and fx2 <= x2 and fy2 <= y2:
            #         human_name = face["name"]
            #         human_score = face["score"]
            #         break

            current_bbox = [x1, y1, x2, y2, score]
            smoothed_bbox = smooth_detections(person_dets_list, track_id, current_bbox)
        
        # Use smoothed bbox for drawing
            x1, y1, x2, y2 = map(int, smoothed_bbox)


            color = get_color_from_id(track_id)
            # history = track_history[track_id]

            # if human_name and human_name != "Unknown":
            #     history["names"].append(human_name)
            #     # history["scores"].append(human_score)

            display_name = "Unknown"
            # if history["names"]:
            #     # majority vote
            #     voted_name = Counter(history["names"]).most_common(1)[0][0]

            #     # get best score for voted name
            #     valid_indices = [
            #         i for i, n in enumerate(history["names"])
            #         if n == voted_name
            #     ]

            #     best_idx = max(valid_indices, key=lambda i: history["scores"][i])
            #     best_score = history["scores"][best_idx]
            #     display_name = voted_name

            cv2.rectangle(img_small, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_small,
                        f'ID {track_id} - {display_name} - {human_name}',
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

            cx = (x1 + x2) // 2
            cy = y2
            cv2.circle(img_small, (cx, cy), 6, color, -1)


        return output_detection, img_small


from ultralytics import YOLO

class Config:
    # Demo type
    demo = "webcam"          # "image", "video", or "webcam"
    path = "./videos/palace.mp4"
    camid = 0
    save_result = True

    # Experiment / model
    exp_file = "exps/example/mot/yolox_m_mix_det.py"
    ckpt = None
    device = "gpu"
    conf = None
    nms = None
    tsize = None
    fps = 30
    fp16 = False
    fuse = False
    trt = True

    # Tracking params
    track_thresh = 0.5
    track_buffer = 10
    match_thresh = 0.8
    aspect_ratio_thresh = 1.6
    min_box_area = 0
    mot20 = False


args = Config()
tracker = BYTETracker(args, frame_rate=30)
tracker_human = BYTETracker(args, frame_rate=30)



first_frame = True
static_overlay = 0

# human_detector = YOLO(current_dir /"yolo26l.engine")
human_detector = YOLO(current_dir /"yolo26s-pose.engine")
# human_detector = YOLO(current_dir /"yolov8n.engine")
# human_detector = YOLO(current_dir /"yolov8n.pt")


import numpy as np
import cv2

import numpy as np
import cv2

import numpy as np
import cv2


def ws_transfer(frame_batch):
    global frame_count, rec_threshold, first_frame, static_overlay

    list_person=[]
    list_score=[]
    list_face = []
    list_obj = []

         
    if first_frame and False: 
        frame_with_overlay = frame.copy()  
        polygon_points = select_polygon(frame)
        # if not polygon_points:
        #     print("No areas defined. Exiting.")
        #     exit()
        

        static_overlay = np.zeros_like(frame)
        import colorsys
        def unique_color(idx, total):
            hue = idx / total
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            return int(b*255), int(g*255), int(r*255)  

        for idx, polygon in enumerate(polygon_points):
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            color = unique_color(idx, len(polygon_points))

            cv2.polylines(frame_with_overlay, [pts], True, color, 2)
            cv2.putText(
                frame_with_overlay,
                f"Area {idx+1}",
                (polygon[0][0], max(20, polygon[0][1]-10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
        first_frame = False


    frame_count += 1
    if frame_count % CLEANUP_INTERVAL == 0:
        gc.collect()
    

    frame = frame_batch
    # face_detection , frame = DecodeFace(frame,  trt_manager,  client, 'n5', tracker)
 
 
    # for idx, (frame) in enumerate(zip(frame_batch)):
    # human_output = human_detector(frame) 
    # human_output = human_detector(frame, classes=[0])[0] 
    # print('human_output', human_output)
    # results = human_detector(frame, stream=False)

    person_count = 0
    # for r in human_output:
    #     boxes = r.boxes
    #     for box in boxes:
    #         cls = int(box.cls[0])
    #         conf = float(box.conf[0])
    #         # Only keep class 'person' (COCO class id 0)
    #         if cls == 0 and conf > 0.5:
    #             person_count += 1
    #             x1, y1, x2, y2 = map(int, box.xyxy[0])
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #             face_detection , _ = DecodeFace(frame[y1:y2, x1:x2],  trt_manager,  client, 'n5', tracker)
    #             for bbox, person, score, face, obj in face_detection:
        
    #                 cv2.putText(frame, f'Person {person_count} {person}', (x1, y1 - 10),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)








    face_detection , frame = DecodeFace(frame,  trt_manager,  client, 'n5',  tracker)
















   



















    # cv2.putText(frame, f'Total Persons: {person_count}', (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)



    # boxes_human = human_output.boxes.xyxy
    # scores = torch.ones((boxes_human.shape[0], 1), device=boxes_human.device)  # all 1.0
    # dets = torch.cat([boxes_human, scores], dim=1)

    # boxes_np = dets.cpu().numpy()

    # online_targets = tracker_human.update(
    #         boxes_np,
    #         frame.shape[:2],
    #         frame.shape[:2]
    #     )
            
    online_targets = []
    # for i, t in enumerate(online_targets):
    #     # score = t.score
    #     # x1, y1, w, h = t.tlwh
    #     x1, y1, w, h = t[0] , t[1], t[2] - t[0], t[3] - t[1]
    #     x2 = int(x1 + w)
    #     y2 = int(y1 + h)
    #     x1, y1 = int(x1), int(y1)
    #     # id = t.track_id
    #     id='23'
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    #     cv2.putText(
    #             frame, 
    #             f"ID: {id}",            # text to display
    #             (x1, max(y1-10, 0)),    # position: 10 pixels above box, never < 0
    #             cv2.FONT_HERSHEY_SIMPLEX, 
    #             0.5,                     # font scale
    #             (0, 255, 0),             # text color
    #             2                        # thickness
    #         )


    # for bbox, person, score, face, obj in face_detection:
        
    #     if score < rec_threshold: 
    #             person = 'Unknown'
    #     list_person.append(person)
    #     list_score.append(score)
    #     list_face.append(face)
    #     list_obj.append(obj)



    success, encoded_frame = cv2.imencode('.jpg', frame,  [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    if success:
        data = {
            "persons": list_person,        
            "scores": list_score,          
            'frame': encoded_frame.tobytes(), 
            'faces': list_face,             
            'objs': list_obj,             
        }
        # print('data',data)
        return data
        


def vectorDatabase(frame, name, save_mode= True):
    print(60*'*')
    if save_mode:
        SaveFace(frame, name, trt_manager,  client, 'n5')
    else:
        print('input name=', name)
        face_detection ,frame = DecodeFace(frame,  trt_manager,  client, 'n5')
        for bbox, person, score, face in face_detection:
            print('detect name', person)

        
      





    
def removeDatabase( names):
    RemoveRecord( names,   client, 'n5')

