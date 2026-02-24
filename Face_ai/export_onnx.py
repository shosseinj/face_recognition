import torch



def load_models(args):
    """Load RetinaFace and ArcFace ONNX models"""
    # Set up ONNX Runtime
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.gpu >= 0 else ['CPUExecutionProvider']
    
    # Load RetinaFace detector
    print("Loading RetinaFace detector...")
    # Try to find the model
    import os
    model_paths = [
        "./weights/buffalo_l/det_10g.onnx",
        "./weights/buffalo_l/det_500m.onnx",
        "det_10g.onnx"
    ]
    
    det_model_path = None
    for path in model_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            det_model_path = expanded_path
            break
    
    if det_model_path is None:
        raise FileNotFoundError("Could not find RetinaFace model file")
    
    print(f"Using RetinaFace model: {det_model_path}")
    det_session = ort.InferenceSession(det_model_path, providers=providers)
    
    # Load ArcFace recognizer
    print("Loading ArcFace recognizer...")
    rec_model_paths = [
        "./weights/buffalo_l/w600k_r50.onnx",
        "./weights/buffalo_l/glintr100.onnx",
        "w600k_r50.onnx"
    ]
    
    rec_model_path = None
    for path in rec_model_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            rec_model_path = expanded_path
            break
    
    if rec_model_path is None:
        raise FileNotFoundError("Could not find ArcFace model file")


    
    return det_session

def export_to_onnx():

    args = parse_args()
    
    print("Loading models...")
    det_session = load_models(args)

    model = RetinaFace()          # initialize your model
    model = RetinaFace(model_file=None, session=det_session)


    model.eval()

    dummy = torch.randn(1, 3, 640, 640)   # match your input resolution

    torch.onnx.export(
        model,
        dummy,
        "retinaface.onnx",
        input_names=["input"],
        output_names=["boxes", "scores", "landmarks"],
        opset_version=11,
        dynamic_axes={"input": {0: "batch"}}
    )

    print("Exported retinaface.onnx")

if __name__ == "__main__":
    export_to_onnx()