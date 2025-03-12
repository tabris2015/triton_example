import torch
from utils.model import STRModel

def export_to_onnx(model_path: str, onnx_path: str):
    # create Pytorch Model Object
    model = STRModel(input_channels=1, output_channels=512, num_classes=37)

    # load weights
    state = torch.load(model_path)
    state = {key.replace("module.", ""): value for key, value in state.items()}

    model.load_state_dict(state)

    # create ONNX file by tracing model
    trace_input = torch.randn(1, 1, 32, 100)

    torch.onnx.export(model, trace_input, onnx_path, verbose=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the PyTorch model file (.pth)")
    parser.add_argument("--onnx-path", type=str, required=True, help="Path where the ONNX model will be saved")

    args = parser.parse_args()
    export_to_onnx(args.model_path, args.onnx_path)
    print(f"ONNX model saved to {args.onnx_path}")
    