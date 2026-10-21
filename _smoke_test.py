# -*- coding: utf-8 -*-
"""端到端一致性冒烟测试：验证修改后的代码与正文/回复信口径一致。"""
import sys, time
import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device = {device}')

# ---------------------------------------------------------------- 1. 编译检查
import py_compile
for f in ['csdi.py', 'hyperedge_generation.py', 'hgcn_model.py',
          'skeleton_topology.py', 'metrics.py', 'pipeline.py']:
    py_compile.compile(f, doraise=True)
print('[1] py_compile: all 6 modules OK')

# ------------------------------------------------- 2. 参数量核对（Sec 3.3/R1-6）
from csdi import CSDI
from hgcn_model import HypergraphActionModel
csdi_p = sum(p.numel() for p in CSDI(num_keypoints=17, num_objects=2,
                                     keypoint_dim=2, T=1000, d_model=64).parameters())
hg = HypergraphActionModel(num_keypoints=34, num_classes=13, in_channels=2,
                           hidden_channels=256, num_hgcn_blocks=9)
hgcn_p = sum(p.numel() for p in hg.parameters())
sel = sum(p.numel() for n, p in hg.named_parameters() if 'kernel_selector' in n)
assert sel == 6939, f'selector params {sel} != 6939'
assert abs(csdi_p / 1e6 - 12.03) < 0.01, f'CSDI {csdi_p} != 12.03M'
print(f'[2] CSDI={csdi_p/1e6:.2f}M (正文12.03M), HGCN={hgcn_p/1e6:.2f}M, '
      f'selector={sel}=771x9 (R1-6) OK')

# --------------------------------------- 3. MDR/FDR 定义核对（Sec 4.2 / R2回复）
from metrics import bidirectional_mdr, fdr_per_class
y_true = np.array([0, 0, 1, 1, 2, 2, 2])
y_pred = np.array([0, 1, 1, 1, 2, 0, 2])   # N=7
mdr = bidirectional_mdr(y_true, y_pred, num_classes=3)
fdr = fdr_per_class(y_true, y_pred, num_classes=3)
# 手算: class0 FN=1,FP=1 -> 2/7=28.57%; class1 FN=1,FP=1 -> 2/7; class2 FN=1,FP=0 -> 1/7=14.29%
assert abs(mdr['per_class'][0] - 200/7) < 1e-6 and abs(mdr['per_class'][2] - 100/7) < 1e-6, mdr
# FDR_c = FP/(TP+FP) = 1 - Precision_c (%): class0 TP=1,FP=1 -> 50; class1 TP=2,FP=1 -> 33.33
assert abs(fdr[0] - 50.0) < 1e-6 and abs(fdr[1] - 100/3) < 1e-6, fdr
print(f"[3] MDR双向定义=(FN_c+FP_c)/N、FDR_c=1-Precision_c OK  "
      f"mdr={np.round(mdr['per_class'],1)}% macro={mdr['macro']:.1f}%")

# -------------------------- 4. Algorithm 1 参数核对（tau_area=50, d_merge=25, sigma=2.5）
from hyperedge_generation import dynamic_hyperedge_generation, build_incidence_matrix
from pipeline import scatter_gaussian_heatmap
import inspect
sig = inspect.signature(dynamic_hyperedge_generation)
assert sig.parameters['d_merge'].default == 25.0, 'd_merge default != 25'
assert sig.parameters['tau_area'].default == 50.0
assert sig.parameters['sigma'].default == 2.5
a = np.ones(34, dtype=np.float32)
pts = np.random.RandomState(0).rand(34, 2).astype(np.float32) * 600 + 20
A = scatter_gaussian_heatmap(a, pts, sigma_h=8.0)   # sigma_h = 8 px (Sec 3.4)
assert A.shape == (640, 640) and A.max() <= 1.0 + 1e-6
edges = dynamic_hyperedge_generation(A, [(float(p[0]), float(p[1])) for p in pts],
                                    tau_area=50.0, d_merge=25.0, sigma=2.5)
H = build_incidence_matrix(edges, num_keypoints=34)
print(f'[4] sigma_h=8px 热图 OK, hyperedges={len(edges)}, H={H.shape} OK')

# ----------------------------------------- 5. λ 网格扫描冒烟测试（微型合成数据）
from pipeline import (lambda_grid_search, LAMBDA_SELECTION_GRID,
                     LAMBDA_ABLATION_GRID, gpu_synchronized_time)
assert LAMBDA_SELECTION_GRID == (0.01, 0.05, 0.1, 0.5, 1.0)
assert LAMBDA_ABLATION_GRID == (0.0, 0.05, 0.1, 0.5, 1.0)
rng = np.random.RandomState(42)
M_tr, M_val, T = 4, 4, 32
def make_complete(m):
    x = rng.rand(m, T, 34, 2).astype(np.float32) * 500 + 50
    return torch.from_numpy(x)
def make_action(m):
    x0 = make_complete(m).clone()
    mask = np.ones((m, T, 34), dtype=np.float32)
    y = torch.randint(0, 13, (m,))
    mask[rng.rand(m, T, 34) < 0.3] = 0.0
    x0_np = x0.numpy()
    x0_np[mask == 0] = 0.0
    return torch.from_numpy(x0_np), torch.from_numpy(mask), y
class SeqDataset(torch.utils.data.Dataset):
    """__getitem__ 返回裸张量 -> default_collate 自动堆叠为 (B, T, 34, 2) 批张量。"""
    def __init__(self, x): self.x = x
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i): return self.x[i]
csd_train = torch.utils.data.DataLoader(SeqDataset(make_complete(M_tr)), batch_size=2)
x0_a, mk_a, y_a = make_action(M_tr)
x0_v, mk_v, y_v = make_action(M_val)
act_train = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x0_a, mk_a, y_a), batch_size=2)
val = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x0_v, mk_v, y_v), batch_size=2)
joint_pairs = [(i, (i + 3) % 34) for i in range(0, 34, 2)]
angle_triplets = [(i, (i + 1) % 34, (i + 2) % 34) for i in range(0, 34, 3)]
mean_lengths = np.linalg.norm(rng.rand(17, 2) * 50 + 10, axis=1).astype(np.float32)
t0 = time.time()
res = lambda_grid_search(
    csd_train, act_train, val, joint_pairs, angle_triplets, mean_lengths,
    grid=(0.0, 0.1), num_csdi_epochs=1, num_hgcn_epochs=1,
    num_ddim_steps=2, d_model=16, device=device, verbose=False)
assert set(res['per_lambda'].keys()) == {0.0, 0.1} and 'best_lambda' in res
r = res['per_lambda'][0.1]
assert all(0.0 <= r[k] <= 100.0 for k in ('Macro-Recall', 'Macro-Precision', 'Macro-F1'))
print(f"[5] lambda_grid_search 冒烟 OK ({time.time()-t0:.1f}s): "
      f"per_lambda keys={list(res['per_lambda'])}, best={res['best_lambda']}, "
      f"λ=0.1 Macro-F1={r['Macro-F1']:.1f}%")

# ------------------------------------------------------- 6. GPU 同步计时核对
t1 = gpu_synchronized_time(device)
assert isinstance(t1, float)
print('[6] gpu_synchronized_time OK (CPU直通, CUDA时先synchronize)')

print('\nALL CHECKS PASSED - 代码与正文/回复信口径一致')
