# python3 nn_numpy.py

import time

import numpy as np

# Размерности
in_dim = 128
hidden_dim = 32
out_dim = 1
batch_size = 8

# Инициализация весов и смещений
np.random.seed(0)
W1 = np.random.randn(in_dim, hidden_dim).astype(np.float32)
b1 = np.zeros(hidden_dim, dtype=np.float32)
W2 = np.random.randn(hidden_dim, out_dim).astype(np.float32)
b2 = np.zeros(out_dim, dtype=np.float32)


# Прямой и обратный проход

def forward_backward(X, Y_true):
    # Forward
    Z1 = X.dot(W1) + b1  # (batch_size, hidden_dim)
    H = np.maximum(Z1, 0)  # ReLU
    Y_pred = H.dot(W2) + b2  # (batch_size, out_dim)

    # Loss и градиент по выходу
    dY = (Y_pred - Y_true)  # MSE derivative (без 1/2)

    # Backward
    dW2 = H.T.dot(dY)  # (hidden_dim, out_dim)
    db2 = dY.sum(axis=0)
    dH = dY.dot(W2.T)  # (batch_size, hidden_dim)
    dZ1 = dH * (Z1 > 0)  # ReLU'
    dW1 = X.T.dot(dZ1)  # (in_dim, hidden_dim)
    db1 = dZ1.sum(axis=0)

    return dW1, db1, dW2, db2


# Генерация случайных данных
X = np.random.randn(batch_size, in_dim).astype(np.float32)
Y = np.random.randn(batch_size, out_dim).astype(np.float32)

# Разогрев
for _ in range(2):
    forward_backward(X, Y)

# Замеры
times = []
for i in range(10):
    t0 = time.perf_counter()
    grads = forward_backward(X, Y)
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)  # в миллисекундах

print("Низкоуровневый Python (NumPy), ms per iter:")
print(times)
