# CPU: python3 nn_pytorch.py --device cpu
# GPU: python3 nn_pytorch.py --device cuda

import argparse
import time

import torch
import torch.nn as nn

# Параметры
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
args = parser.parse_args()

device = torch.device(args.device)

# Размерности
in_dim, hidden_dim, out_dim = 128, 32, 1
batch_size = 8

# Модель и функции
model = nn.Sequential(
    nn.Linear(in_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, out_dim)
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Данные
X = torch.randn(batch_size, in_dim, device=device)
Y = torch.randn(batch_size, out_dim, device=device)

# Разогрев (исключаем из замеров)
for _ in range(2):
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, Y)
    loss.backward()
    optimizer.step()

# Замеры
times = []
for i in range(10):
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.perf_counter()
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, Y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

print(f"PyTorch ({device.type.upper()}) ms per iter:")
print(times)
