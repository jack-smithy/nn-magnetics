import torch
import time
from nn_magnetics.model import Network

torch.set_default_dtype(torch.float32)

x = torch.randn(1024, 6)
model = Network(in_features=6, hidden_dim_factor=48, out_features=12)

cpu_times = []

for epoch in range(100):
    t0 = time.perf_counter()
    output = model(x)
    t1 = time.perf_counter()
    cpu_times.append(t1 - t0)

device = "cuda:0"
model = model.to(device)
x = x.to(device)
torch.cuda.synchronize()

gpu_times = []
for epoch in range(100):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    output = model(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    gpu_times.append(t1 - t0)

print(
    "CPU {}, GPU {}".format(
        torch.tensor(cpu_times).mean(), torch.tensor(gpu_times).mean()
    )
)
