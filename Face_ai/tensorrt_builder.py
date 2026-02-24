import tensorrt as trt

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

onnx_path = "./weights/buffalo_l/det_10g.onnx"
with open(onnx_path, "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
config.set_flag(trt.BuilderFlag.FP16)

# inspect inputs
inputs = []
for i in range(network.num_inputs):
    inp = network.get_input(i)
    print(f"Input {i}: {inp.name} shape={inp.shape}")
    inputs.append(inp)

profile = builder.create_optimization_profile()

# HEIGHT and WIDTH are dynamic
min_hw = (1, 3, 480, 480)
opt_hw = (1, 3, 640, 640)
max_hw = (1, 3, 1280, 1280)

profile.set_shape(inputs[0].name, min_hw, opt_hw, max_hw)

config.add_optimization_profile(profile)

engine = builder.build_serialized_network(network, config)
if engine is None:
    raise RuntimeError("Engine build failed")

with open("retinaface_fp16.engine", "wb") as f:
    f.write(engine)

print("✓ DONE: TensorRT engine created")



# import tensorrt as trt

# logger = trt.Logger(trt.Logger.INFO)
# builder = trt.Builder(logger)
# network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
# parser = trt.OnnxParser(network, logger)

# onnx_path = "./weights/buffalo_l/w600k_r50.onnx"
# with open(onnx_path, "rb") as f:
#     if not parser.parse(f.read()):
#         for i in range(parser.num_errors):
#             print(parser.get_error(i))
#         raise RuntimeError("ONNX parse failed!")

# config = builder.create_builder_config()
# config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
# config.set_flag(trt.BuilderFlag.FP16)

# inp = network.get_input(0)
# print("Input:", inp.name, inp.shape)

# # add optimization profile for dynamic batch
# profile = builder.create_optimization_profile()

# min_shape = (1, 3, 112, 112)
# opt_shape = (4, 3, 112, 112)
# max_shape = (16, 3, 112, 112)

# profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
# config.add_optimization_profile(profile)

# engine = builder.build_serialized_network(network, config)
# if engine is None:
#     raise RuntimeError("Engine build failed")

# with open("arcface_fp16.engine", "wb") as f:
#     f.write(engine)

# print("✓ DONE: arcface_fp16.engine created")