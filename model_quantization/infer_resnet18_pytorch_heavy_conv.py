import os
import torch
import torch.nn as nn
import torchvision.models as models
import time

class ResNet18HeavyConv(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # ResNet18 분해
        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # 여기서 feature map은 대체로 [B, 512, 7, 7]
        self.heavy_block = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.avgpool = base.avgpool
        self.fc = base.fc

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.heavy_block(x)   # profiler에서 여기 확 튐

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def main():
    os.makedirs("models", exist_ok=True)
    model = ResNet18HeavyConv().eval().cuda()
    torch.save(model.state_dict(), "models/resnet18_heavy_conv.pth")
    print("Saved model weights to models/resnet18_heavy_conv.pth")

    x = torch.randn(1, 3, 224, 224).cuda()

    # warmup
    for _ in range(20):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()

    # measure latency
    times = []
    for _ in range(100):
        start = time.time()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)      # s -> ms

    avg_ms = sum(times) / len(times)
    fps = 1000 / avg_ms

    print(f"ResNet18 Heavy Conv PyTorch avg latency: {avg_ms:.3f} ms")
    print(f"ResNet18 Heavy Conv PyTorch FPS: {fps:.2f}")

if __name__ == "__main__":
    main()