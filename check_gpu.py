import torch
print(f"CUDA Device Count: {torch.cuda.device_count()}")
tp_size = max([d for d in [i for i in range(1, 29) if 28 % i == 0 and i % 2 == 0] if d <= torch.cuda.device_count()] + [1])
print(f"Calculated tp_size: {tp_size}")
