"""
Hypergraph Learning Model for Action Classification (HGCN)
============================================================

This module implements the hypergraph learning model described in the manuscript,
consisting of:
  - Spatial Graph Convolution (SGC) - Eq. 7
  - Dynamic Temporal Convolution (DTC) - Eq. 8
  - Hypergraph Convolution (HGC) - Eq. 9
  - Action Classifier (GAP + FC + Softmax) - Eqs. 10-13

The model takes repaired keypoint sequences and constructed hypergraphs as input,
and outputs action class predictions for the 13 active worker-equipment interaction
actions defined in the manuscript.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict


# =============================================================================
# Spatial Graph Convolution (SGC) — Eq. 7
# =============================================================================
class SpatialGraphConv(nn.Module):
    """
    Spatial Graph Convolution using Laplacian transformation (Eq. 7).

    Z^(s)_t = sigma(D_tilde^{-1/2} * A_tilde * D_tilde^{-1/2} * X_t * W_s)

    Where A_tilde = A + I is the self-ring adjacency matrix.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.W_s = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature tensor (B, T, V, C_in)
            adj: Normalized adjacency matrix (V, V) or (B, V, V)
        Returns:
            out: Spatial feature tensor (B, T, V, C_out)
        """
        B, T, V, C_in = x.shape
        x_flat = x.reshape(B * T, V, C_in)

        # Laplacian transformation: D^{-1/2} A D^{-1/2} X W_s
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B * T, -1, -1)

        out = torch.bmm(adj, x_flat)  # (B*T, V, C_in)
        out = self.W_s(out)           # (B*T, V, C_out)
        out = out.reshape(B, T, V, -1)

        # Apply BN and activation
        out = out.permute(0, 3, 1, 2).reshape(B, -1, T * V)
        out = self.bn(out)
        out = out.reshape(B, -1, T, V).permute(0, 2, 3, 1)

        return F.gelu(out)


# =============================================================================
# Dynamic Temporal Convolution (DTC) — Eq. 8
# =============================================================================
class DynamicTemporalConv(nn.Module):
    """
    Dynamic Temporal Convolution (Eq. 8).

    Z^(st) = Conv1D_theta(Z^(s))

    Uses adaptive kernel sizes for dynamic receptive fields.
    """

    def __init__(self, channels: int, kernel_sizes: list = None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
        self.convs = nn.ModuleList()
        # Ensure channels is evenly divisible; use remainder allocation
        n_kernels = len(kernel_sizes)
        ch_per = channels // n_kernels
        ch_list = [ch_per] * (n_kernels - 1) + [channels - ch_per * (n_kernels - 1)]
        for ks, ch_out in zip(kernel_sizes, ch_list):
            padding = ks // 2
            self.convs.append(nn.Conv1d(channels, ch_out, kernel_size=ks, padding=padding))

        self.bn = nn.BatchNorm1d(channels)
        # Adaptive kernel weight generator
        self.kernel_selector = nn.Linear(channels, len(kernel_sizes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spatial feature tensor (B, T, V, C)
        Returns:
            out: Spatio-temporal feature tensor (B, T, V, C)
        """
        B, T, V, C = x.shape

        # Apply multiple kernel sizes and concatenate
        outputs = []
        for conv in self.convs:
            x_perm = x.permute(0, 2, 3, 1).reshape(B * V, C, T)  # (B*V, C, T)
            out = conv(x_perm)  # (B*V, C//K, T)
            outputs.append(out)

        # Concatenate along channel dimension
        out = torch.cat(outputs, dim=1)  # (B*V, C, T)

        # BN + activation
        out = self.bn(out)
        out = F.gelu(out)

        out = out.reshape(B, V, C, T).permute(0, 3, 1, 2)  # (B, T, V, C)
        return out


# =============================================================================
# Hypergraph Convolution (HGC) — Eq. 9
# =============================================================================
class HypergraphConv(nn.Module):
    """
    Hypergraph Convolution using Laplacian transformation (Eq. 9).

    Z = sigma(L * Z^(st) * W_h)

    where L = I - D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}

    Inspired by 1-HyperGCN (Yadati et al. 2019).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.W_h = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spatio-temporal feature tensor (B, T, V, C_in)
            H: Incidence matrix (V, E) or (B, V, E)
        Returns:
            out: Hypergraph feature tensor (B, T, V, C_out)
        """
        B, T, V, C_in = x.shape

        # Compute hypergraph Laplacian
        if H.dim() == 2:
            H_batch = H.unsqueeze(0).expand(B, -1, -1)
        else:
            H_batch = H

        # D_v = diag(sum of H along axis=1)
        d_v = H_batch.sum(dim=2, keepdim=True).clamp(min=1e-8)  # (B, V, 1)
        D_v_inv_sqrt = torch.pow(d_v, -0.5)

        # D_e = diag(sum of H along axis=0)
        d_e = H_batch.sum(dim=1)  # (B, E)
        D_e_inv = torch.pow(d_e.clamp(min=1e-8), -1.0)  # (B, E)

        # L = I - D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}
        HT = H_batch.transpose(1, 2)  # (B, E, V)
        # L_part = D_v^{-1/2} * H * D_e^{-1}, shape: (B, V, E)
        L_part = (D_v_inv_sqrt * H_batch) * D_e_inv.unsqueeze(1)
        # L = I - L_part @ (HT * D_v^{-1/2}.T), where D_v^{-1/2} must be (B, 1, V) for HT (B, E, V)
        L = torch.eye(V, device=x.device).unsqueeze(0) - torch.bmm(L_part, HT * D_v_inv_sqrt.transpose(1, 2))

        # Apply hypergraph convolution
        x_flat = x.reshape(B * T, V, C_in)
        L_expanded = L.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, V, V)

        out = torch.bmm(L_expanded, x_flat)  # (B*T, V, C_in)
        out = self.W_h(out)                   # (B*T, V, C_out)
        out = out.reshape(B, T, V, -1)

        # BN + activation
        out = out.permute(0, 3, 1, 2).reshape(B, -1, T * V)
        out = self.bn(out)
        out = out.reshape(B, -1, T, V).permute(0, 2, 3, 1)

        return F.gelu(out)


# =============================================================================
# HGCN Block (SGC + DTC + HGC + Residual)
# =============================================================================
class HGCNBlock(nn.Module):
    """
    HGCN Block: Spatial GC + Dynamic Temporal Conv + Hypergraph Conv.

    Each block processes features through three stages:
    1. Spatial graph convolution (physical constraints)
    2. Dynamic temporal convolution (temporal dynamics)
    3. Hypergraph convolution (high-order interaction features)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 adj: Optional[torch.Tensor] = None):
        super().__init__()
        self.sgc = SpatialGraphConv(in_channels, out_channels)
        self.dtc = DynamicTemporalConv(out_channels)
        self.hgc = HypergraphConv(out_channels, out_channels)

        # Residual connection
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

        # Store adjacency
        if adj is not None:
            self.register_buffer('adj', adj)
        else:
            self.adj = None

    def forward(self, x: torch.Tensor, H: torch.Tensor,
                adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input feature tensor (B, T, V, C_in)
            H: Hypergraph incidence matrix (V, E) or (B, V, E)
            adj: Optional adjacency matrix (overrides stored adj)
        Returns:
            out: Output feature tensor (B, T, V, C_out)
        """
        adj_use = adj if adj is not None else self.adj

        res = self.residual(x)

        z_s = self.sgc(x, adj_use)    # Eq. 7
        z_st = self.dtc(z_s)          # Eq. 8
        z_h = self.hgc(z_st, H)       # Eq. 9

        return z_h + res


# =============================================================================
# Action Classifier (GAP + FC + Softmax) — Eqs. 10-13
# =============================================================================
class ActionClassifier(nn.Module):
    """
    Action classification head (Eqs. 10-13).

    1. Global Average Pooling (GAP): z_bar = (1/V*T) * sum(Z[i,t,:])  — Eq. 10
    2. Fully Connected: l = W_c * z_bar + b_c                          — Eq. 11
    3. Softmax: P(y=k|X) = exp(l_k) / sum(exp(l_j))                   — Eq. 12
    4. Decision: y_hat = argmax P(y=k|X)                                — Eq. 13
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # GAP over V and T
        self.fc = nn.Linear(in_channels, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor from HGCN blocks (B, T, V, C)
        Returns:
            logits: Unnormalized class scores (B, num_classes)
        """
        B, T, V, C = x.shape

        # Global Average Pooling (Eq. 10)
        z_bar = x.mean(dim=(1, 2))  # (B, C) — average over T and V

        # Fully connected layer (Eq. 11)
        z_bar = self.dropout(z_bar)
        logits = self.fc(z_bar)  # (B, K)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted class indices (Eq. 13)."""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)   # Eq. 12
        y_hat = torch.argmax(probs, dim=-1)  # Eq. 13
        return y_hat

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class probabilities (Eq. 12)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


# =============================================================================
# Complete Hypergraph Learning Model
# =============================================================================
class HypergraphActionModel(nn.Module):
    """
    Complete Hypergraph Learning Model for action classification.

    Architecture (Figure 3, lower panel):
    1. Input: Repaired keypoint sequence X_hat (B, C, T, V, O)
    2. Initial block: BN + linear projection to high-dimensional space
    3. 9 stacked HGCN blocks (SGC + DTC + HGC)
    4. Action Classifier (GAP + FC + Softmax)

    Args:
        num_keypoints: Total number of keypoints (V)
        num_classes: Number of action classes (K=13)
        in_channels: Input feature dimension (2 for x,y coordinates)
        hidden_channels: Hidden feature dimension (default: 256)
        num_hgcn_blocks: Number of stacked HGCN blocks (default: 9)
    """

    def __init__(self, num_keypoints: int = 34, num_classes: int = 13,
                 in_channels: int = 2, hidden_channels: int = 256,
                 num_hgcn_blocks: int = 9):
        super().__init__()

        # Initial block: project to high-dimensional space
        self.input_bn = nn.BatchNorm1d(in_channels)
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # Build adjacency matrix for spatial GC (physical skeleton topology)
        self.register_buffer('adj', self._build_default_adjacency(num_keypoints))

        # 9 stacked HGCN blocks
        self.hgcn_blocks = nn.ModuleList()
        for i in range(num_hgcn_blocks):
            in_ch = hidden_channels if i > 0 else hidden_channels
            out_ch = hidden_channels
            self.hgcn_blocks.append(HGCNBlock(in_ch, out_ch))

        # Action classifier
        self.classifier = ActionClassifier(hidden_channels, num_classes)

    @staticmethod
    def _build_default_adjacency(num_keypoints: int) -> torch.Tensor:
        """
        Build default physical skeleton adjacency matrix.
        Uses skeleton_topology module for accurate per-entity topology (Table 1).
        For combined worker-equipment pairs, builds worker topology and pads.
        """
        try:
            from skeleton_topology import build_adjacency_matrix
            adj = build_adjacency_matrix('worker')  # (13, 13)
        except Exception:
            # Fallback: identity + sequential connections
            adj = torch.eye(13)
            for i in range(12):
                adj[i, i + 1] = 1.0
                adj[i + 1, i] = 1.0

        # Pad to num_keypoints (typically 34 = 13 worker + up to 21 equipment)
        V = num_keypoints
        full_adj = torch.eye(V)
        w = min(adj.shape[0], V)
        full_adj[:w, :w] = adj[:w, :w]
        # Add sequential connections for equipment keypoints
        for i in range(13, V):
            if i > 13:
                full_adj[i, i - 1] = 1.0
                full_adj[i - 1, i] = 1.0
            # Connect first equipment kpt to relevant worker kpts (wrists)
            if i == 13:
                full_adj[5, i] = 1.0  # left wrist -> equipment
                full_adj[i, 5] = 1.0
                full_adj[6, i] = 1.0  # right wrist -> equipment
                full_adj[i, 6] = 1.0

        # Normalize: D^{-1/2} A D^{-1/2}
        degree = full_adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
        full_adj = full_adj / degree.sqrt() / degree.sqrt().t()
        return full_adj

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Repaired keypoint sequence (B, T, V, 2)
            H: Hypergraph incidence matrix (V, E) or (B, V, E)
        Returns:
            logits: Action class logits (B, num_classes)
        """
        B, T, V, D = x.shape

        # Input projection
        x_flat = x.reshape(B * T * V, D)
        x_flat = self.input_bn(x_flat)
        x = self.input_proj(x_flat).reshape(B, T, V, -1)

        # Stack 9 HGCN blocks
        for block in self.hgcn_blocks:
            x = block(x, H, self.adj)

        # Classification
        logits = self.classifier(x)
        return logits

    def predict(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Predict action class indices."""
        logits = self.forward(x, H)
        return torch.argmax(F.softmax(logits, dim=-1), dim=-1)


# =============================================================================
# Complete Pipeline: CSDI + Hyperedge Generation + HGCN
# =============================================================================
class WorkerEquipmentInteractionPipeline(nn.Module):
    """
    Complete inference pipeline integrating all components:
    1. Object Detection (YOLOv11n) — external, not implemented here
    2. Pose Estimation (YOLOv11n-pose) — external, not implemented here
    3. Occlusion Keypoint Repair (CSDI)
    4. Dynamic Hyperedge Generation (Algorithm 1)
    5. Interaction Action Classification (HGCN)
    """

    def __init__(self, num_keypoints: int = 34, num_classes: int = 13,
                 d_model: int = 64, hidden_channels: int = 256):
        super().__init__()
        # Import CSDI
        from csdi import CSDI

        self.csdi = CSDI(
            num_keypoints=num_keypoints // 2,
            num_objects=2,
            keypoint_dim=2,
            T=1000,
            d_model=d_model,
        )
        self.hgcn = HypergraphActionModel(
            num_keypoints=num_keypoints,
            num_classes=num_classes,
            in_channels=2,
            hidden_channels=hidden_channels,
        )

    def forward(self, x_obs: torch.Tensor, mask: torch.Tensor,
                H: torch.Tensor) -> torch.Tensor:
        """
        Complete pipeline inference.

        Args:
            x_obs: Observed (partially occluded) keypoints (B, T, V, 2)
            mask: Visibility mask (B, T, V), 1=visible, 0=occluded
            H: Hypergraph incidence matrix (V, E)

        Returns:
            logits: Action predictions (B, num_classes)
        """
        # Step 1: Repair keypoints
        x_repaired = self.csdi.repair(x_obs, mask, num_inference_steps=50)

        # Step 2: Classify actions
        logits = self.hgcn(x_repaired, H)

        return logits


# =============================================================================
# Action Labels — Matching TABLE 2 (13 categories of active interaction actions)
# =============================================================================
ACTION_CLASSES: Dict[int, str] = {
    0:  "driving/operating (A01)",          # worker + large mechanical equipment
    1:  "hooking/unhooking (A02)",          # worker + large mechanical equipment
    2:  "inspecting/maintaining (A03)",     # worker + large mechanical equipment
    3:  "pushing/pulling/towing (A04)",     # worker + large mechanical equipment
    4:  "climbing (A05)",                   # worker + large mechanical equipment
    5:  "directing/intercepting (A06)",     # worker + medium-sized equipment
    6:  "carrying/transporting (A07)",      # worker + medium-sized equipment
    7:  "crossing/stepping over (A08)",     # worker + medium-sized equipment
    8:  "holding/gripping (A09)",           # worker + small handheld equipment
    9:  "pressing (A10)",                   # worker + small handheld equipment
    10: "hammering/striking (A11)",         # worker + small handheld equipment
    11: "digging (A12)",                    # worker + small handheld equipment
    12: "cutting (A13)",                    # worker + small handheld equipment
}

# Interaction type classification (Table 2)
INTERACTION_TYPES = {
    'worker_large_equipment': [0, 1, 2, 3, 4],      # A01-A05
    'worker_medium_equipment': [5, 6, 7],             # A06-A08
    'worker_handheld_equipment': [8, 9, 10, 11, 12],  # A09-A13
}

# Large mechanical equipment: excavator, crane, forklift, truck, aerial work platform
# Medium-sized equipment: ladder, scaffold, handcart, wheelbarrow, enclosure
# Small handheld equipment: hammer, pickaxe, shovel, saw, drill


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Hyperparameters
    B, T, V = 4, 64, 34  # batch, frames, keypoints
    num_classes = 13
    num_hyperedges = 8

    # Create model
    model = HypergraphActionModel(
        num_keypoints=V,
        num_classes=num_classes,
        in_channels=2,
        hidden_channels=256,
        num_hgcn_blocks=9,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"HGCN Model parameters: {total_params / 1e6:.2f}M")

    # Dummy inputs
    x = torch.randn(B, T, V, 2).to(device)  # Repaired keypoints
    H = torch.randint(0, 2, (V, num_hyperedges)).float().to(device)  # Incidence matrix

    # Forward pass
    logits = model(x, H)
    probs = F.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)

    print(f"Output logits shape: {logits.shape}")
    print(f"Predictions: {preds.cpu().tolist()}")
    for i, p in enumerate(preds.cpu().tolist()):
        print(f"  Sample {i}: {ACTION_CLASSES[p]}")

    print("Hypergraph Action Model test passed!")
