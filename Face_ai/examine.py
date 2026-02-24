# import onnx

onnx_model_path = "./weights/buffalo_l/det_10g.onnx"


# model = onnx.load(onnx_path)
# for i, out in enumerate(model.graph.output):
#     print(f"Output {i}: name={out.name} shape={out.type.tensor_type.shape}")



import onnx
import onnxruntime as ort
import numpy as np
import cv2

# Load ONNX model
# onnx_model_path = "retinaface.onnx"  # Change this to your ONNX file path
model = onnx.load(onnx_model_path)

print("=" * 60)
print("MODEL INFORMATION")
print("=" * 60)

# Print model info
print(f"Model IR version: {model.ir_version}")
print(f"Producer name: {model.producer_name}")
print(f"Producer version: {model.producer_version}")

# Print input information
print("\n" + "=" * 60)
print("INPUT INFORMATION")
print("=" * 60)
for input in model.graph.input:
    print(f"Input name: {input.name}")
    print(f"Input type: {input.type.tensor_type.elem_type}")
    print("Input shape:", end=" ")
    shape = []
    for dim in input.type.tensor_type.shape.dim:
        if dim.dim_value:
            shape.append(dim.dim_value)
        elif dim.dim_param:
            shape.append(dim.dim_param)
        else:
            shape.append("?")
    print(shape)
    print("-" * 30)

# Print output information
print("\n" + "=" * 60)
print("OUTPUT INFORMATION")
print("=" * 60)
for output in model.graph.output:
    print(f"Output name: {output.name}")
    print(f"Output type: {output.type.tensor_type.elem_type}")
    print("Output shape:", end=" ")
    shape = []
    for dim in output.type.tensor_type.shape.dim:
        if dim.dim_value:
            shape.append(dim.dim_value)
        elif dim.dim_param:
            shape.append(dim.dim_param)
        else:
            shape.append("?")
    print(shape)
    print("-" * 30)

# Run inference with dummy data to see actual outputs
print("\n" + "=" * 60)
print("RUNNING INFERENCE WITH DUMMY DATA")
print("=" * 60)

# Create ONNX Runtime session
session = ort.InferenceSession(onnx_model_path)

# Get input name and shape
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Input name: {input_name}")
print(f"Input shape: {input_shape}")

# Create dummy input (adjust based on your model's input shape)
if len(input_shape) == 4:  # Batch, Channels, Height, Width
    batch_size, channels, height, width = input_shape
    # Replace dynamic dimensions with 1
    batch_size = 1 if batch_size == -1 else batch_size
    height = 640 if height == -1 else height
    width = 640 if width == -1 else width
    channels = 3 if channels == -1 else channels
    
    # Create random image
    dummy_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    print(f"Created dummy input with shape: {dummy_input.shape}")
else:
    # Create simple dummy input
    dummy_input = np.random.randn(*[1 if dim == -1 else dim for dim in input_shape]).astype(np.float32)
    print(f"Created dummy input with shape: {dummy_input.shape}")

# Run inference
outputs = session.run(None, {input_name: dummy_input})

# Print all outputs
print("\n" + "=" * 60)
print("OUTPUT DETAILS")
print("=" * 60)
for i, output in enumerate(outputs):
    print(f"\nOutput {i}:")
    print(f"  Shape: {output.shape}")
    print(f"  Dtype: {output.dtype}")
    print(f"  Min value: {output.min():.6f}")
    print(f"  Max value: {output.max():.6f}")
    print(f"  Mean value: {output.mean():.6f}")
    print(f"  Std value: {output.std():.6f}")
    
    # Print first few values
    if output.size > 0:
        print(f"  First 10 values (flattened):")
        flat_output = output.flatten()
        for j in range(min(10, len(flat_output))):
            print(f"    [{j}]: {flat_output[j]:.6f}")
    
    # Print statistics for confidence outputs (assuming they're between 0-1)
    if output.min() >= 0 and output.max() <= 1:
        print(f"  Values > 0.5: {(output > 0.5).sum()} / {output.size}")
        print(f"  Values > 0.7: {(output > 0.7).sum()} / {output.size}")
    
    # For location outputs (typically 4 values per box)
    if len(output.shape) >= 2 and output.shape[-1] == 4:
        print(f"  This appears to be location/box output (last dimension = 4)")
        print(f"  Number of boxes: {output.shape[0] if len(output.shape) == 2 else output.shape[1]}")
    
    print("-" * 40)

# Alternative: Print with more details using ONNX graph
print("\n" + "=" * 60)
print("NODE INFORMATION")
print("=" * 60)
print(f"Number of nodes in graph: {len(model.graph.node)}")

# Print the last few nodes to understand output structure
print("\nLast 10 nodes in the graph:")
for i, node in enumerate(model.graph.node[-10:]):
    print(f"\nNode {i}: {node.name}")
    print(f"  Op type: {node.op_type}")
    print(f"  Inputs: {node.input}")
    print(f"  Outputs: {node.output}")