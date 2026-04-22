import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18HeavyFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        return self.backbone(x)

def main():
    model = ResNet18HeavyFC().eval().cuda()
    dummy = torch.randn(1, 3, 224, 224).cuda()

    torch.onnx.export(
        model,
        dummy,
        "models/resnet18_heavy_fc.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True
    )
    print("Saved ONNX to models/resnet18_heavy_fc.onnx")

if __name__ == "__main__":
    main()