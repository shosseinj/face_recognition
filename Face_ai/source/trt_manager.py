import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
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
            
            # Detection attributes
            self.det_input_name = None
            self.det_input_shape = None
            self.det_input_dtype = None
            self.det_output_info = {}
            
            # Recognition attributes
            self.rec_input_name = None
            self.rec_output_name = None
            self.rec_input_shape = None
    
    def initialize(self, det_engine_path, rec_engine_path=None, det_input_shape=(1,3,640,640), rec_input_shape=(1,3,112,112)):
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
            
            # Get input name for detection
            self.det_input_name = self.det_engine.get_tensor_name(0)
            print(f"Detection input name: {self.det_input_name}")
            
            # Set input shape for detection
            self.det_context.set_input_shape(self.det_input_name, det_input_shape)
            assert self.det_context.all_binding_shapes_specified, "Detection input shape not set correctly"
            
            # Load recognition engine if provided
            if rec_engine_path:
                print(f"Loading recognition engine: {rec_engine_path}")
                with open(rec_engine_path, "rb") as f:
                    runtime = trt.Runtime(logger)
                    self.rec_engine = runtime.deserialize_cuda_engine(f.read())
                self.rec_context = self.rec_engine.create_execution_context()
                
                # Get input/output names for recognition
                self.rec_input_name = self.rec_engine.get_tensor_name(0)
                self.rec_output_name = self.rec_engine.get_tensor_name(1)
                print(f"Recognition input name: {self.rec_input_name}")
                print(f"Recognition output name: {self.rec_output_name}")
                
                # Set input shape for recognition
                self.rec_context.set_input_shape(self.rec_input_name, rec_input_shape)
                assert self.rec_context.all_binding_shapes_specified, "Recognition input shape not set correctly"
            
            print("Engines loaded, allocating buffers...")
            
            # Allocate buffers
            self._allocate_detection_buffers()
            if rec_engine_path:
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
        self.det_bindings = []
        self.det_outputs = []
        self.det_output_info = {}
        
        # Input buffer
        input_shape = self.det_context.get_tensor_shape(self.det_input_name)
        input_dtype = trt.nptype(self.det_engine.get_tensor_dtype(self.det_input_name))
        input_size = np.prod(input_shape) * np.dtype(input_dtype).itemsize
        d_input = cuda.mem_alloc(int(input_size))
        self.det_bindings.append(d_input)
        self.det_input_dtype = input_dtype
        self.det_input_shape = input_shape
        
        print(f"Detection input allocated: shape={input_shape}, size={input_size/1024/1024:.2f}MB")
        
        # Output buffers
        for i in range(1, self.det_engine.num_io_tensors):
            output_name = self.det_engine.get_tensor_name(i)
            output_shape = self.det_context.get_tensor_shape(output_name)
            output_dtype = trt.nptype(self.det_engine.get_tensor_dtype(output_name))
            
            # Handle dynamic shapes
            actual_shape = tuple(max(1, s) if s == -1 else s for s in output_shape)
            h_output = np.empty(actual_shape, dtype=output_dtype)
            d_output = cuda.mem_alloc(int(h_output.nbytes))
            
            self.det_bindings.append(d_output)
            self.det_outputs.append((h_output, d_output, actual_shape, output_dtype))
            self.det_output_info[output_name] = (h_output, d_output, actual_shape, output_dtype)
            
            print(f"Detection output: {output_name}, shape={actual_shape}")

    def _allocate_recognition_buffers(self):
        """Allocate buffers for recognition engine"""
        self.rec_bindings = []
        
        # Input buffer
        input_shape = self.rec_context.get_tensor_shape(self.rec_input_name)
        input_dtype = trt.nptype(self.rec_engine.get_tensor_dtype(self.rec_input_name))
        input_size = np.prod(input_shape) * np.dtype(input_dtype).itemsize
        d_input = cuda.mem_alloc(int(input_size))
        self.rec_bindings.append(d_input)
        self.rec_input_shape = input_shape
        
        print(f"Recognition input allocated: shape={input_shape}, size={input_size/1024/1024:.2f}MB")
        
        # Output buffer
        output_shape = self.rec_context.get_tensor_shape(self.rec_output_name)
        output_dtype = trt.nptype(self.rec_engine.get_tensor_dtype(self.rec_output_name))
        h_output = np.empty(output_shape, dtype=output_dtype)
        d_output = cuda.mem_alloc(int(h_output.nbytes))
        self.rec_bindings.append(d_output)
        self.rec_output = (h_output, d_output)
        
        print(f"Recognition output allocated: shape={output_shape}")

    def _allocate_detection_buffers_dynamic(self, input_shape):
        """Reallocate buffers for dynamic batch size"""
        print(f"Reallocating detection buffers for shape: {input_shape}")
        
        # Free existing buffers
        for binding in self.det_bindings:
            if binding:
                try:
                    cuda.mem_free(binding)
                except:
                    pass
        
        self.det_bindings = []
        self.det_outputs = []
        self.det_output_info = {}
        
        # Input buffer
        input_size = np.prod(input_shape) * np.dtype(self.det_input_dtype).itemsize
        d_input = cuda.mem_alloc(int(input_size))
        self.det_bindings.append(d_input)
        
        # Get output shapes for this batch size
        for i in range(1, self.det_engine.num_io_tensors):
            output_name = self.det_engine.get_tensor_name(i)
            # Query the shape after setting input
            output_shape = self.det_context.get_tensor_shape(output_name)
            output_dtype = trt.nptype(self.det_engine.get_tensor_dtype(output_name))
            
            # Make sure shape is valid
            actual_shape = tuple(max(1, s) if s == -1 else s for s in output_shape)
            h_output = np.empty(actual_shape, dtype=output_dtype)
            d_output = cuda.mem_alloc(int(h_output.nbytes))
            
            self.det_bindings.append(d_output)
            self.det_outputs.append((h_output, d_output, actual_shape, output_dtype))
            self.det_output_info[output_name] = (h_output, d_output, actual_shape, output_dtype)
        
        self.det_input_shape = input_shape
        print(f"Buffers reallocated for batch size {input_shape[0]}")

    def detect_faces_batch(self, batch_input):
        """
        Detect faces on multiple frames in batch
        
        Args:
            batch_input: np.array of shape (batch_size, 3, 640, 640) preprocessed
        
        Returns:
            List of outputs: [loc, conf, landmarks] each as numpy arrays
        """
        if not self._initialized:
            raise RuntimeError("TensorRT not initialized")
        
        self.cuda_ctx.push()
        try:
            batch_size = batch_input.shape[0]
            input_shape = batch_input.shape
            
            # Set dynamic batch shape
            self.det_context.set_input_shape(self.det_input_name, input_shape)
            
            # Reallocate buffers if batch size changed
            if self.det_input_shape != input_shape:
                self._allocate_detection_buffers_dynamic(input_shape)
            
            # Copy input to GPU
            d_input = self.det_bindings[0]
            cuda.memcpy_htod_async(d_input, batch_input, self.stream)
            
            # Set tensor addresses
            self.det_context.set_tensor_address(self.det_input_name, int(d_input))
            
            # Set output addresses and collect host buffers
            output_host_buffers = []
            output_names = []
            
            for i, (output_name, (h_buf, d_buf, shape, dtype)) in enumerate(self.det_output_info.items()):
                self.det_context.set_tensor_address(output_name, int(d_buf))
                output_host_buffers.append(h_buf)
                output_names.append(output_name)
            
            # Execute
            self.det_context.execute_async_v3(stream_handle=self.stream.handle)
            self.stream.synchronize()
            
            # Copy outputs back
            for i, (h_buf, d_buf, shape, dtype) in enumerate(self.det_outputs):
                cuda.memcpy_dtoh_async(h_buf, d_buf, self.stream)
            
            self.stream.synchronize()
            
            # Return list of output arrays (not just one)
            # Make a copy of each output to ensure they're separate
            result_outputs = []
            for h_buf in output_host_buffers:
                result_outputs.append(h_buf.copy())
            
            
            return result_outputs  # Return list of arrays
            
        except Exception as e:
            print(f"Error in batch detection: {e}")
            traceback.print_exc()
            return []
        finally:
            self.cuda_ctx.pop()

    def detect_faces(self, input_tensor):
        """Original single frame detection (kept for backward compatibility)"""
        if not self._initialized:
            raise RuntimeError("TensorRT not initialized")
        
        self.cuda_ctx.push()
        try:
            # Ensure input has batch dimension
            if len(input_tensor.shape) == 3:
                input_tensor = np.expand_dims(input_tensor, axis=0)
            
            results = self.detect_faces_batch(input_tensor)
            if results:
                return results[0] if len(results) == 1 else results
            return []
            
        except Exception as e:
            print(f"Error in single detection: {e}")
            traceback.print_exc()
            return []
        finally:
            self.cuda_ctx.pop()

    def recognize_face(self, face_tensor):
        """Perform face recognition"""
        if not self._initialized or self.rec_context is None:
            raise RuntimeError("TensorRT recognition not initialized")
        
        self.cuda_ctx.push()
        try:
            # Ensure correct shape
            if len(face_tensor.shape) == 3:
                face_tensor = np.expand_dims(face_tensor, axis=0)
            
            # Set input shape
            self.rec_context.set_input_shape(self.rec_input_name, face_tensor.shape)
            
            # Copy input
            d_input = self.rec_bindings[0]
            cuda.memcpy_htod(d_input, face_tensor)
            
            # Set tensor addresses
            self.rec_context.set_tensor_address(self.rec_input_name, int(d_input))
            self.rec_context.set_tensor_address(self.rec_output_name, int(self.rec_bindings[1]))
            
            # Execute
            self.rec_context.execute_async_v3(stream_handle=self.stream.handle)
            self.stream.synchronize()
            
            # Get output
            h_output, d_output = self.rec_output
            cuda.memcpy_dtoh(h_output, d_output)
            
            return h_output.copy()
            
        except Exception as e:
            print(f"Error in recognition: {e}")
            traceback.print_exc()
            return None
        finally:
            self.cuda_ctx.pop()

    def cleanup(self):
        """Proper cleanup of all resources"""
        print("\nCleaning up TensorRT resources...")
        
        try:
            # Free detection buffers
            for binding in self.det_bindings:
                if binding:
                    try:
                        cuda.mem_free(binding)
                    except:
                        pass
            
            # Free recognition buffers
            for binding in self.rec_bindings:
                if binding:
                    try:
                        cuda.mem_free(binding)
                    except:
                        pass
            
            # Clear references
            self.det_context = None
            self.rec_context = None
            self.det_engine = None
            self.rec_engine = None
            self.det_bindings = []
            self.rec_bindings = []
            self.det_outputs = []
            self.rec_output = None
            
        finally:
            if self.cuda_ctx:
                try:
                    self.cuda_ctx.pop()
                    del self.cuda_ctx
                    self.cuda_ctx = None
                except:
                    pass
        
        self._initialized = False
        print("Cleanup complete")