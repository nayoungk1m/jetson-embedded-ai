import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity

def main():
    model = models.resnet18(weights=None).eval().cuda()
    x = torch.randn(1, 3, 224, 224).cuda()

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ],
        record_shapes=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./tensorboard")
    ) as prof:
        
        for _ in range(20):
            with torch.no_grad():
                _ = model(x)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))   # cuda_time_total or cpu_time_total

if __name__ == "__main__":
    main()