



import cv2
import numpy as np
import tensorrt as trt

import pycuda.driver as cuda

import traceback




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
            
