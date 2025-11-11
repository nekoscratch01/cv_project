import torch, time

print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

# 自动选择设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 构造两个随机矩阵做乘法测试
x = torch.randn(4096, 4096, device=device)
y = torch.randn(4096, 4096, device=device)

# 预热几次（让 GPU/MPS 稳定）
for _ in range(3):
    z = x @ y
    if device.type == "mps":
        torch.mps.synchronize()

# 正式计时
t0 = time.time()
z = x @ y
if device.type == "mps":
    torch.mps.synchronize()
t1 = time.time()

print(f"Matrix multiply done in {t1 - t0:.4f}s")
print(f"Result mean value: {z.mean().item():.6f}")
