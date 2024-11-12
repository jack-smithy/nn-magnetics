import torch
import time
from src.model import Network

torch.set_default_dtype(torch.float32)

x = torch.randn(100000, 6)
model = Network(in_features=6, hidden_dim_factor=12, out_features=3)

cpu_times = []

for epoch in range(100):
    t0 = time.perf_counter()
    output = model(x)
    t1 = time.perf_counter()
    cpu_times.append(t1 - t0)

device = "mps"
model = model.to(device)
x = x.to(device)
torch.mps.synchronize()

gpu_times = []
for epoch in range(100):
    torch.mps.synchronize()
    t0 = time.perf_counter()
    output = model(x)
    torch.mps.synchronize()
    t1 = time.perf_counter()
    gpu_times.append(t1 - t0)

print(
    "CPU {}, GPU {}".format(
        torch.tensor(cpu_times).mean(), torch.tensor(gpu_times).mean()
    )
)
