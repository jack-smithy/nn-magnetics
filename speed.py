import torch
import time

import torch.optim.adam
from nn_magnetics.model import Network, get_num_params, ComplexNet

torch.set_default_dtype(torch.float32)

BATCH_SIZE = 128
x = torch.randn(BATCH_SIZE, 7)
y = torch.randn(BATCH_SIZE, 3)

model = ComplexNet(
    in_features=7,
    hidden_features=128,
    out_features=3,
    n_residual_blocks=2,
)

loss_fn = torch.nn.L1Loss()
opt = torch.optim.Adam(params=model.parameters())

print(f"Num threads: {torch.get_num_threads()}")
print(f"Number of model params: {get_num_params(model)}")
cpu_times = []

for epoch in range(500):
    t0 = time.perf_counter()

    output = model(x)
    opt.zero_grad()
    loss = loss_fn(output, y)
    loss.backward()
    opt.step()

    t1 = time.perf_counter()
    cpu_times.append(t1 - t0)

device = "mps"
model = model.to(device)
x = x.to(device)
y = y.to(device)
loss_fn = loss_fn.to(device)
opt = torch.optim.Adam(params=model.parameters())

torch.mps.synchronize()
gpu_times = []

for epoch in range(500):
    torch.mps.synchronize()
    t0 = time.perf_counter()

    opt.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    opt.step()

    torch.mps.synchronize()
    t1 = time.perf_counter()
    gpu_times.append(t1 - t0)

print(
    "CPU {}, GPU {}".format(
        torch.tensor(cpu_times).mean(), torch.tensor(gpu_times).mean()
    )
)
