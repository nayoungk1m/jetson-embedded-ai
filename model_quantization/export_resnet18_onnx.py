import torch
import torchvision.models as models

PTH_PATH = "models/resnet18.pth"
ONNX_PATH = "models/resnet18.onnx"

def main():
    model = models.resnet18(weights=None)
    model.load_state_dict(torch.load(PTH_PATH))
    model.eval().cuda()

    dummy = torch.randn(1, 3, 224, 224).cuda()

    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True
    )

    print(f"Saved ONNX to {ONNX_PATH}")

if __name__ == "__main__":
    main()