import time
import torch
import torchvision.models as models
import os

PTH_PATH = "models/resnet18.pth"

def main():
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(PTH_PATH):
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        torch.save(model.state_dict(), PTH_PATH)
        print(f"Saved ResNet-18 weights to {PTH_PATH}")

    # Load resnet-18 model and move to GPU  
    model = models.resnet18(weights=None)
    model.load_state_dict(torch.load(PTH_PATH))
    model.eval().cuda()
    print("Loaded ResNet-18 model and moved to GPU")

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

    print(f"PyTorch avg latency: {avg_ms:.3f} ms")
    print(f"PyTorch FPS: {fps:.2f}")

if __name__ == "__main__":
    main()