# Source Code for the Proposed Framework

This directory contains the PyTorch implementation of the framework described in:

**"Simulating and detecting keypoint-based active worker-equipment interaction actions under partial occlusion via conditional diffusion model and hypergraph learning"**

Wenrui Zhu, Junqi Yu, Erhu Liu, Chunyong Feng, Zhengwei Song

Manuscript No.: COENG-19511  
Journal: ASCE Journal of Construction Engineering and Management

> **Note:** Due to project confidentiality agreements with our industry collaborators, we are unable to release the raw dataset. This code provides the complete model architecture and inference pipeline, faithfully following the mathematical formulations (Eqs. 1–13) and Algorithm 1 described in the manuscript.

---

## Framework Overview

The proposed hybrid cascaded framework comprises five core phases (Figure 1):

| Phase | Module | Description |
|-------|--------|-------------|
| I | Data Collection & Preprocessing | Video acquisition (60.3% real sites + 39.7% online), resize to 640×640 |
| II | Object Detection & Pose Estimation | YOLO11n (16 entities) + YOLO11n-pose (keypoint extraction) |
| III | Keypoint Repair | Conditional diffusion model (CSDI) with Temporal U-Net denoiser |
| IV | Hyperedge Generation | Dynamic hyperedge generation algorithm from frozen STA-GCN attention |
| V | Action Classification | Hypergraph learning model (9 HGCN blocks) for 13 interaction actions |

---

## Code Structure

| File | Description | Manuscript Reference |
|------|-------------|---------------------|
| `csdi.py` | Conditional diffusion model (CSDI) for occlusion keypoint repair, including Temporal U-Net denoiser, noise schedule, DDIM sampling, physical plausibility constraints, and training mask generation | Section 3.3, Eqs. 1–6 |
| `hyperedge_generation.py` | Dynamic hyperedge generation algorithm (5 stages: heatmap preprocessing → salient region extraction → keypoint-region matching → hyperedge generation → spatial merging) | Section 3.4, Algorithm 1 |
| `hgcn_model.py` | Hypergraph learning model: Spatial Graph Convolution (SGC), Dynamic Temporal Convolution (DTC), Hypergraph Convolution (HGC), Action Classifier | Section 3.5, Eqs. 7–13 |
| `skeleton_topology.py` | Skeleton topology definitions for all 16 construction entity categories (keypoint names, physical edges, adjacency matrices) | Section 3.2, Table 1 |
| `metrics.py` | Evaluation metrics for keypoint repair (AJC, MPJPE, PCK@0.05×diag, FPS) and action recognition (Recall, Precision, F1, Macro-F1, MDR, FDR, confusion matrix) | Section 4.2 |
| `pipeline.py` | Complete end-to-end inference pipeline integrating all 5 phases | Figure 1 |
| `verify_params.py` | Parameter count verification and inference speed benchmarking for Temporal U-Net | — |

---

## Dependencies

```
python >= 3.11
torch >= 2.3.0
numpy
opencv-python
ultralytics (optional, for YOLO11 inference)
```

---

## Quick Start

### 1. Keypoint Repair (CSDI)

```python
from csdi import CSDI, generate_training_mask, repair_keypoints

model = CSDI(num_keypoints=17, num_objects=2, keypoint_dim=2, T=1000, d_model=64)

# Training with hybrid masking (70% random + 30% structured limb-part)
mask = generate_training_mask((B, T, N), strategy='hybrid')
loss = model.training_loss(x0, mask)  # x0: (B, T, N, 2), mask: (B, T, N)

# Inference (DDIM: 1000 → 50 steps, Eq. 6)
x_obs = x0 * mask.unsqueeze(-1)  # Eq. 1
x_repaired = repair_keypoints(model, x_obs, mask, num_inference_steps=50)
```

### 2. Dynamic Hyperedge Generation (Algorithm 1)

```python
from hyperedge_generation import dynamic_hyperedge_generation, build_incidence_matrix

hyperedges = dynamic_hyperedge_generation(
    attention_heatmap=heatmap,  # STA-GCN attention (H, W), frozen teacher
    keypoints=keypoints,        # repaired keypoint coordinates [(x_i, y_i), ...]
    tau_area=50.0,              # minimum region area (px²)
    d_merge=30.0,               # merge distance for adjacent peaks
    sigma=2.5,                  # Gaussian blur std
)
H = build_incidence_matrix(hyperedges, num_keypoints=len(keypoints))
```

### 3. Action Classification (HGCN)

```python
from hgcn_model import HypergraphActionModel

model = HypergraphActionModel(num_keypoints=34, num_classes=13, hidden_channels=256)

# Forward pass (Eqs. 7-13)
logits = model(x_repaired, H_incidence)  # (B, 13)
predictions = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
```

### 4. Skeleton Topology (Table 1)

```python
from skeleton_topology import build_adjacency_matrix, build_combined_adjacency

# Single entity
adj_worker = build_adjacency_matrix('worker')  # (13, 13)

# Worker-equipment pair
adj_combined = build_combined_adjacency('worker', 'excavator')  # (21, 21)
```

### 5. Evaluation Metrics

```python
from metrics import compute_repair_metrics, compute_action_metrics

# Keypoint repair metrics
repair_results = compute_repair_metrics(predicted, gt, mask, bbox_diagonal=bbox_diag)
# Returns: AJC, MPJPE, PCK@10px, PCK@0.05×diag

# Action recognition metrics
action_results = compute_action_metrics(y_true, y_pred, num_classes=13)
# Returns: accuracy, macro_f1, macro_recall, macro_precision, MDR, FDR, confusion_matrix, per-class metrics
```

### 6. Complete Pipeline

```python
from pipeline import WorkerEquipmentInteractionPipeline

pipeline = WorkerEquipmentInteractionPipeline(device='cuda')
result = pipeline("video.mp4", entity_pair=('worker', 'excavator'))
print(f"Action: {result['action_name']}, Confidence: {result['confidence']:.4f}")
print(f"Latency: {result['latency_ms']['total_ms']:.1f} ms ({result['latency_ms']['FPS']:.1f} FPS)")
```

---

## Architecture Details

### Temporal U-Net (CSDI Denoiser) — Eqs. 1–6, Figure 5

| Component | Specification |
|-----------|--------------|
| Input | Concatenation of [x_noisy (2), x_cond (2), mask (1)] = 5 channels per keypoint |
| Encoder | 4 residual temporal conv blocks: [64, 128, 256, 512] channels |
| Bottleneck | 1024 channels, temporal self-attention (8 heads, d_k=128) |
| Decoder | 4 upsampling blocks: [512, 256, 128, 64] channels, with skip connections |
| Timestep embedding | Sinusoidal positional encoding → MLP (Linear → GELU → Linear), 256-dim, injected at 8 points |
| Activation | GELU throughout |
| Total parameters | ~12.0M |

**Noise Schedule**: Linear, β₁ = 1×10⁻⁴, β_T = 0.02, T = 1000 steps (training)

**Inference**: DDIM sampling accelerates reverse process from T=1000 to 50 steps, achieving 4.1 ms/frame on NVIDIA RTX 3090

**Physical Plausibility Constraints** (L_physical):
- Joint length constraint: repaired distances within ±15% of training set mean
- Articulation angle constraint: joint angles within biomechanically plausible ranges (e.g., elbow: 30°–160°)
- Total loss: L_total = L_diffusion + λ · L_physical

### Training Mask Generation

| Strategy | Ratio | Description |
|----------|-------|-------------|
| Random rectangular | 70% | Random keypoint masking simulating object occlusion |
| Structured limb-part | 30% | Body-part group masking simulating self-occlusion |

Mask ratio range: 20%–70% of keypoints per frame

### Dynamic Hyperedge Generation — Algorithm 1

| Stage | Steps | Operation |
|-------|-------|-----------|
| 1 | 1–3 | Gaussian blur (σ=2.5) + Otsu thresholding + contour detection |
| 2 | 4–10 | Bounding rectangle extraction + area filtering (τ_area ≥ 50 px²) |
| 3 | 11–19 | Keypoint-region matching (point-in-rectangle test) |
| 4 | 20–27 | Initial hyperedge generation (regions with ≥2 keypoints) |
| 5 | 28–29 | Spatial neighborhood merging (Union-Find, d_merge threshold) |

**STA-GCN Teacher**: Pre-trained on training set with full keypoints, **frozen** during hyperedge generation. Serves solely as spatial-temporal attention feature extractor.

### Hypergraph Learning Model — Eqs. 7–13, Figure 4 (lower panel)

| Component | Description |
|-----------|-------------|
| Input block | BatchNorm + linear projection to 256-dim |
| HGCN Block × 9 | Each block: SGC (Eq. 7) + DTC (Eq. 8) + HGC (Eq. 9) + residual connection |
| SGC | Spatial graph convolution with normalized Laplacian (physical skeleton topology) |
| DTC | Dynamic temporal convolution with adaptive kernel sizes [3, 5, 7] |
| HGC | Hypergraph convolution: L = I − D_v⁻¹/² H D_e⁻¹ H^T D_v⁻¹/² |
| Action classifier | Global Average Pooling (Eq. 10) → FC (Eq. 11) → Softmax (Eq. 12) → argmax (Eq. 13) |
| Hidden channels | 256 |
| Output | 13 action classes (Table 2): A01–A13 |

### Skeleton Topology — Table 1

| Entity Type | Keypoints | Examples |
|-------------|-----------|---------|
| Worker | 13 | head, shoulders, elbows, wrists, hips, knees, ankles |
| Large equipment | 6–8 | excavator, crane, forklift, truck, aerial work platform |
| Medium equipment | 4–7 | ladder, scaffold (N), handcart, wheelbarrow, enclosure |
| Small handheld | 3–5 | hammer, pickaxe, shovel, saw, drill |

Total: 16 entity categories with custom-defined keypoint sequences

---

## Dataset

| Attribute | Value |
|-----------|-------|
| Total video clips | 6,847 (5–8s per clip) |
| Data sources | Real construction sites (60.3%) + Online (39.7%) |
| Entity categories | 16 (Table 1) |
| Action classes | 13 (Table 2) |
| Image resolution | 640 × 640 pixels |
| Sampling rate | 8 FPS |

**Dataset Split**:
- Training set: non-occluded videos (stratified random, K-S test p=0.87)
- Validation set: non-occluded videos (~8:1 ratio with training)
- Test set: partially occluded authentic worker operation videos

**Ground Truth for Occluded Keypoints**: Multi-reference inference approach — annotators observed 5–10 adjacent frames, inferred positions via spatiotemporal continuity, 3 independent annotators with IoU > 0.7 consensus.

---

## Implementation Details

### Training Configuration

| Component | Setting |
|-----------|---------|
| YOLO11n / YOLO11n-pose | 300 epochs, AdamW, lr=0.001, cosine annealing, batch=16 |
| Temporal U-Net (CSDI) | T=1000 training steps, linear noise schedule (β₁=1e-4, β_T=0.02) |
| HGCN Model | 50 epochs, AdamW, weight decay=0.01, cross-entropy loss |
| Hardware | Intel Xeon Silver 4214, NVIDIA RTX 3090 (24GB), 64GB RAM |
| Software | Ubuntu 22.04 LTS, Python 3.11.1, PyTorch 2.3.0 |

### Inference Configuration

| Component | Setting |
|-----------|---------|
| CSDI repair | DDIM sampling: 1000 → 50 steps |
| Real-time speed | 54.5 FPS (full pipeline on RTX 3090) |

---

## Performance Benchmarks (Table 3)

### Keypoint Repair

| 2D Estimator | AJC (%) | 2D MPJPE | PCK@0.05×diag (%) | FPS |
|--------------|---------|----------|---------------------|-----|
| OpenPose | 72.5 | 24.0 | 66.1 | 35.3 |
| OpenPose + CSDI | 89.5 | 12.6 | 68.4 | 16.7 |
| YOLOv8-pose | 82.3 | 21.2 | 75.5 | 62.6 |
| YOLOv8-pose + CSDI | 92.5 | 11.3 | 79.8 | 48.3 |
| YOLO11-pose | 84.1 | 19.7 | 77.8 | 71.4 |
| **YOLO11-pose + CSDI** | **94.7** | **10.3** | **83.1** | **54.5** |

### Action Recognition (Table 6, selected)

| Model | Input | Macro-Recall (%) | Macro-Precision (%) | Macro-F1 (%) |
|-------|-------|-------------------|---------------------|---------------|
| ST-GCN | occlusion | 66.84 | 81.42 | 73.33 |
| ST-GCN | repaired | 77.38 | 90.36 | 83.34 |
| SkateFormer | repaired | 92.35 | 94.05 | 93.12 |
| **Ours (Hypergraph)** | repaired | **91.68** | **95.59** | **93.58** |

### Graph Construction Ablation (Table 4)

| Graph Type | Macro-Recall (%) | Macro-F1 (%) |
|------------|-------------------|---------------|
| Static skeleton graph | 76.9 | 79.5 |
| k-NN graph (k=3) | 80.4 | 82.7 |
| Distance-threshold graph (θ=0.3) | 79.1 | 81.4 |
| Learned attention graph | 83.1 | 85.3 |
| Ground-truth interaction hypergraph | 93.5 | 92.5 |
| **STA-GCN attention hypergraph (Ours)** | **91.7** | **93.6** |

---

## 13 Action Classes (Table 2)

| ID | Action | Interaction Type |
|----|--------|-----------------|
| A01 | driving/operating | Worker + Large mechanical equipment |
| A02 | hooking/unhooking | Worker + Large mechanical equipment |
| A03 | inspecting/maintaining tracks or tires | Worker + Large mechanical equipment |
| A04 | pushing/pulling/towing | Worker + Large mechanical equipment |
| A05 | climbing | Worker + Large mechanical equipment |
| A06 | directing/intercepting | Worker + Medium-sized equipment |
| A07 | carrying/transporting | Worker + Medium-sized equipment |
| A08 | crossing/stepping over | Worker + Medium-sized equipment |
| A09 | holding/gripping | Worker + Small handheld equipment |
| A10 | pressing | Worker + Small handheld equipment |
| A11 | hammering/striking | Worker + Small handheld equipment |
| A12 | digging | Worker + Small handheld equipment |
| A13 | cutting | Worker + Small handheld equipment |

---

## Citation

If you use this code, please cite the original paper (upon acceptance).

## License

This code is released for academic reproducibility purposes. The dataset is subject to project confidentiality agreements and will be available upon reasonable request after 04/30/2027.
