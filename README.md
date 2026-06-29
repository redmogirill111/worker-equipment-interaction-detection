# YOLO11n-pose + CSDI (Temporal U-Net) + hypergraph learning

PyTorch implementation of the framework described in:

**"Simulating and detecting keypoint-based active worker-equipment interaction actions under partial occlusion via conditional diffusion model and hypergraph learning"**

---

## File Structure

| File | Description |
|------|-------------|
| `csdi.py` | Conditional diffusion model (CSDI) for keypoint repair: Temporal U-Net denoiser, noise schedule, DDIM sampling, physical plausibility loss, training mask generation |
| `hyperedge_generation.py` | Dynamic hyperedge generation algorithm (5 stages) + incidence matrix construction + hypergraph Laplacian |
| `hgcn_model.py` | Hypergraph learning model: SGC + DTC + HGC blocks, action classifier (GAP + FC + Softmax) |
| `skeleton_topology.py` | Keypoint definitions and adjacency matrices for 16 construction entity categories |
| `metrics.py` | Evaluation metrics for keypoint repair (AJC, MPJPE, PCK) and action recognition (Recall, Precision, F1, Macro-F1, MDR, FDR) |
| `pipeline.py` | End-to-end inference pipeline integrating all 5 phases |
| `verify_params.py` | Parameter count verification and CPU inference benchmark |

---

## Dependencies

```
python >= 3.11
torch >= 2.3.0
numpy
opencv-python
ultralytics  # optional, for YOLO11 inference
```

---

## Quick Start

```python
from csdi import CSDI, generate_training_mask, repair_keypoints
from hyperedge_generation import dynamic_hyperedge_generation, build_incidence_matrix
from hgcn_model import HypergraphActionModel

# 1. Keypoint repair (CSDI, DDIM 1000→50 steps)
model = CSDI(num_keypoints=17, num_objects=2, keypoint_dim=2, T=1000, d_model=64)
x_repaired = repair_keypoints(model, x_obs, mask, num_inference_steps=50)

# 2. Hyperedge generation (Algorithm 1, frozen STA-GCN attention)
hyperedges = dynamic_hyperedge_generation(
    attention_heatmap=heatmap, keypoints=keypoints,
    tau_area=50.0, d_merge=30.0, sigma=2.5,
)
H = build_incidence_matrix(hyperedges, num_keypoints=34)

# 3. Action classification (HGCN)
model = HypergraphActionModel(num_keypoints=34, num_classes=13, hidden_channels=256)
logits = model(x_repaired, H)  # (B, 13)
```

---

## Model Architecture

### Temporal U-Net (CSDI Denoiser) — Figure 5

| Component | Channels | Notes |
|-----------|----------|-------|
| Encoder × 4 | 64 → 128 → 256 → 512 | Residual temporal conv, MaxPool after each |
| Bottleneck | 1024 | Residual conv + temporal self-attention (8 heads, d_k=128) |
| Decoder × 4 | 512 → 256 → 128 → 64 | ConvTranspose + skip connection + residual conv |
| Timestep embedding | 256-dim | Sinusoidal PE → MLP (Linear → GELU → Linear), injected at 8 points |
| Activation | GELU | Throughout |
| Total parameters | ~12.0M | |

### HGCN Model — Figure 4 (lower panel)

| Component | Notes |
|-----------|-------|
| Input block | BN + Linear (2 → 256) |
| HGCN blocks × 9 | Each: SGC (Eq. 7) + DTC (Eq. 8) + HGC (Eq. 9) + residual |
| Classifier | GAP (Eq. 10) → FC (Eq. 11) → Softmax (Eq. 12) |
| Hidden channels | 256 |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA RTX 3090 (24GB) |
| CPU | Intel Xeon Silver 4214 |
| RAM | 64 GB |
| OS | Ubuntu 22.04 LTS |
| Python / PyTorch | 3.11.1 / 2.3.0 |

### YOLO11n / YOLO11n-pose

| Parameter | Value |
|-----------|-------|
| Epochs | 300 |
| Optimizer | AdamW |
| Learning rate | 0.001 (cosine annealing) |
| Batch size | 16 |
| Pre-trained weights | MS COCO |

### CSDI (Temporal U-Net)

| Parameter | Value |
|-----------|-------|
| Diffusion steps (training) | T = 1000 |
| Diffusion steps (inference) | 50 (DDIM sampling) |
| Noise schedule | Linear, β₁ = 1×10⁻⁴, β_T = 0.02 |
| Masking strategy | Hybrid: 70% random + 30% structured limb-part |
| Mask ratio range | 20%–70% per frame |
| Physical constraint | Joint length ±15%, angle 30°–160° |
| Loss | L_total = L_diffusion + λ · L_physical |

### HGCN

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Loss | Cross-entropy |

---

## Dataset

| Attribute | Value |
|-----------|-------|
| Total clips | 6,847 (5–8s each) |
| Sources | Real sites 60.3% + Online 39.7% |
| Resolution | 640 × 640 |
| Sampling rate | 8 FPS |
| Split ratio | ~8:1:1 (train : val : test) |
| Test set | Partially occluded authentic worker videos |

---

## License

Code released for academic reproducibility. Dataset available upon reasonable request after 08/31/2027.
