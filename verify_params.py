"""Verify TemporalUNet parameter count from actual code."""
import torch
import time
import sys
sys.path.insert(0, '.')
from csdi import CSDI, TemporalUNet

# TemporalUNet alone
unet = TemporalUNet(keypoint_dim=2, num_keypoints=17, num_objects=2, T=64, d_model=64)
unet_params = sum(p.numel() for p in unet.parameters())
print(f'TemporalUNet params: {unet_params:,} ({unet_params/1e6:.2f}M)')

# Full CSDI
model = CSDI(num_keypoints=17, num_objects=2, keypoint_dim=2, T=1000, d_model=64)
total_params = sum(p.numel() for p in model.parameters())
print(f'Full CSDI params:    {total_params:,} ({total_params/1e6:.2f}M)')

# Breakdown
print()
header = f'{"Module":<40} {"Params":>12}'
print(header)
print('=' * 55)
for name, module in unet.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f'{name:<40} {params:>12,}')
print('=' * 55)
print(f'{"TemporalUNet Total":<40} {unet_params:>12,}')

# Forward test
B, T_seq, N = 2, 64, 34
x_noisy = torch.randn(B, T_seq, N, 2)
t = torch.full((B,), 10, dtype=torch.long)
x_cond = torch.randn(B, T_seq, N, 2)
cond_mask = torch.ones(B, T_seq, N)
out = unet(x_noisy, t, x_cond, cond_mask)
print(f'\nForward pass: input {x_noisy.shape} -> output {out.shape}')

# Inference speed on CPU
unet.eval()
x = torch.randn(1, 64, 34, 2)
m = torch.ones(1, 64, 34)
# Warmup
for _ in range(5):
    with torch.no_grad():
        unet(x, torch.tensor([0]), x, m)

# Benchmark
N_runs = 50
start = time.perf_counter()
for _ in range(N_runs):
    with torch.no_grad():
        unet(x, torch.tensor([0]), x, m)
elapsed = time.perf_counter() - start
lat = elapsed / N_runs * 1000
print(f'\nUNet single forward (CPU): {lat:.2f} ms')
print(f'CSDI repair = 50 DDIM steps x UNet forward = ~{lat*50:.0f} ms on CPU')
print(f'(On GPU this would be much faster)')
