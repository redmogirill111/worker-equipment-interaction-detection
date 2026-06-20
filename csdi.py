"""
Conditional Spatio-Temporal Diffusion Imputation (CSDI) for Keypoint Repair
============================================================================

This module implements the conditional diffusion model (CSDI) for repairing
occluded keypoints in worker-equipment interaction videos, as described in:

"Simulating and detecting keypoint-based active worker-equipment interactions
 under partial occlusion via conditional diffusion model and hypergraph learning"

Reference: Tashiro et al. (2021) - CSDI: Conditional Score-based Diffusion
Model for Probabilistic Time Series Imputation.

Equations (1)-(6) in the manuscript.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# Sinusoidal Position Embeddings for Diffusion Timesteps
# =============================================================================
class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


# =============================================================================
# Noise Schedule (Linear schedule as described in the paper)
# beta_1 = 1e-4, beta_T = 0.02 (Section 4.1 / Implementation Details)
# =============================================================================
class LinearNoiseSchedule:
    """Linear noise schedule with beta_1 = 1e-4 and beta_T = 0.02 (Eq. 2)."""

    def __init__(self, num_timesteps: int = 1000, beta_1: float = 1e-4, beta_T: float = 0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_1, beta_T, num_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For DDPM posterior q(x_{t-1} | x_t, x_0) (Eq. 5)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )


# =============================================================================
# Temporal U-Net Denoising Network (described in manuscript Section 3.3)
#
# Paper specification:
#   Encoder: 4 temporal conv blocks [64, 128, 256, 512]
#   Bottleneck: Residual block + Temporal self-attention (8 heads), 1024ch
#   Decoder: 4 upsampling blocks with skip connections
#   Total: ~5.2M parameters
#
# Input: X in R^{C x T x V x O}
#        C=channel(2 coords), T=64 frames, V=keypoints per entity, O=objects
# The model processes temporal sequences of keypoint coordinates.
# =============================================================================
class TemporalAttention(nn.Module):
    """Multi-head temporal self-attention for keypoint sequences."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C) — temporal sequence for a single keypoint
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.d_k).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        return out


class ResidualTemporalBlock(nn.Module):
    """Residual temporal convolution block with optional attention.
    Processes along the temporal dimension T using Conv1d."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 use_attention: bool = False, n_heads: int = 8, dropout: float = 0.1,
                 time_emb_dim: int = 256):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.attn = TemporalAttention(out_channels, n_heads, dropout) if use_attention else None
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        # x: (B, C, T) — temporal conv along T dimension
        res = self.residual(x)
        x = self.conv(x)
        x = self.bn(x)
        t_emb = self.time_mlp(t_emb).unsqueeze(-1)  # (B, C, 1)
        x = x + t_emb
        x = F.gelu(x)
        x = self.dropout(x)
        x = x + res
        # Optional temporal attention
        if self.attn is not None:
            x_perm = x.permute(0, 2, 1)  # (B, T, C)
            x_perm = self.attn(x_perm)
            x = x_perm.permute(0, 2, 1)  # (B, C, T)
        return x


class TemporalUNet(nn.Module):
    """
    Lightweight Temporal U-Net denoising network (~5.2M parameters).

    Processes each keypoint's temporal sequence independently.
    Input dimension: keypoint_dim * 2 (noisy coords + condition coords)
                     + 1 (condition mask indicator)

    Encoder: 4 temporal conv blocks [64, 128, 256, 512]
    Bottleneck: Residual block + Temporal self-attention (8 heads), 1024ch
    Decoder: 4 upsampling blocks with skip connections [512, 256, 128, 64]
    """

    def __init__(self, keypoint_dim: int = 2, num_keypoints: int = 17, num_objects: int = 2,
                 T: int = 64, d_model: int = 64):
        super().__init__()
        self.T = T
        self.N = num_keypoints * num_objects
        self.d_model = d_model

        # Input: concatenate [x_noisy(2), x_cond(2), mask(1)] = 5 channels per keypoint
        cond_dim = keypoint_dim * 2 + 1
        self.input_proj = nn.Linear(cond_dim, d_model)

        # Time embedding -> d_model * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * 4),
        )
        t_dim = d_model * 4

        # Encoder: 4 levels [64, 128, 256, 512]
        self.enc = nn.ModuleList([
            ResidualTemporalBlock(d_model, d_model, time_emb_dim=t_dim),            # 64->64
            ResidualTemporalBlock(d_model, d_model * 2, time_emb_dim=t_dim),        # 64->128
            ResidualTemporalBlock(d_model * 2, d_model * 4, time_emb_dim=t_dim),    # 128->256
            ResidualTemporalBlock(d_model * 4, d_model * 8, time_emb_dim=t_dim),    # 256->512
        ])
        self.pool = nn.MaxPool1d(2)

        # Bottleneck: 512->1024 with temporal self-attention (8 heads) [Figure 5]
        self.bottleneck = ResidualTemporalBlock(d_model * 8, d_model * 16, use_attention=True, time_emb_dim=t_dim)

        # Decoder: 4 levels (symmetric U-Net, matching Figure 5)
        # dec[0]: up 1024->512, cat skip(512 from enc[3])=1024, conv 1024->512
        # dec[1]: up 512->256,  cat skip(256 from enc[2])=512,  conv 512->256
        # dec[2]: up 256->128,  cat skip(128 from enc[1])=256,  conv 256->128
        # dec[3]: up 128->64,   cat skip(64 from enc[0])=128,   conv 128->64
        self.dec_up = nn.ModuleList([
            nn.ConvTranspose1d(d_model * 16, d_model * 8, 2, stride=2),
            nn.ConvTranspose1d(d_model * 8, d_model * 4, 2, stride=2),
            nn.ConvTranspose1d(d_model * 4, d_model * 2, 2, stride=2),
            nn.ConvTranspose1d(d_model * 2, d_model, 2, stride=2),
        ])
        self.dec = nn.ModuleList([
            ResidualTemporalBlock(d_model * 8 * 2, d_model * 8, time_emb_dim=t_dim),  # 1024->512
            ResidualTemporalBlock(d_model * 4 * 2, d_model * 4, time_emb_dim=t_dim),  # 512->256
            ResidualTemporalBlock(d_model * 2 * 2, d_model * 2, time_emb_dim=t_dim),  # 256->128
            ResidualTemporalBlock(d_model * 2, d_model, time_emb_dim=t_dim),           # 128->64
        ])

        self.output_proj = nn.Linear(d_model, keypoint_dim)

    def forward(self, x_noisy, t, x_cond, cond_mask):
        """
        Args:
            x_noisy: Noisy keypoint sequence (B, T_seq, N, 2)
            t: Timestep (B,)
            x_cond: Observed (condition) keypoint values (B, T_seq, N, 2)
            cond_mask: Condition mask (B, T_seq, N) — 1 for visible, 0 for occluded
        Returns:
            Predicted noise epsilon_theta (B, T_seq, N, 2)
        """
        B, T_seq, N, D = x_noisy.shape

        # Concatenate condition: [x_noisy, x_cond, mask]
        mask_exp = cond_mask.unsqueeze(-1)
        x_input = torch.cat([x_noisy, x_cond, mask_exp], dim=-1)  # (B, T, N, 5)
        h = self.input_proj(x_input)  # (B, T, N, d_model)

        # Reshape: (B*N, d_model, T) — Conv1d along temporal axis
        h = h.permute(0, 2, 3, 1).reshape(B * N, self.d_model, T_seq)

        # Time embedding
        t_emb = self.time_mlp(t)
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)

        # Encoder
        skips = []
        for block in self.enc:
            h = block(h, t_emb)
            skips.append(h)
            h = self.pool(h)

        # Bottleneck
        h = self.bottleneck(h, t_emb)

        # Decoder: 4 levels, skip with enc[3], enc[2], enc[1], enc[0]
        for up, conv, skip in zip(self.dec_up, self.dec, reversed(skips)):
            h = up(h)
            if h.shape[-1] < skip.shape[-1]:
                h = F.pad(h, [0, skip.shape[-1] - h.shape[-1]])
            elif h.shape[-1] > skip.shape[-1]:
                skip = F.pad(skip, [0, h.shape[-1] - skip.shape[-1]])
            h = torch.cat([h, skip], dim=1)
            h = conv(h, t_emb)

        # Output: (B*N, d_model, T) -> (B, T, N, 2)
        h = h.permute(0, 2, 1).reshape(B, N, T_seq, self.d_model).permute(0, 2, 1, 3)
        return self.output_proj(h)


# =============================================================================
# Physical Plausibility Constraints (described in R5-C4f response)
# Soft constraints: joint length ±15%, articulation angle within plausible ranges
# =============================================================================
class PhysicalPlausibilityLoss(nn.Module):
    """
    Soft physical constraints on repaired keypoints:
    - Joint length constraint: Repaired keypoint-to-keypoint distances within +/-15%
      of the statistical mean from the training set.
    - Articulation angle constraint: Joint angles within biomechanically plausible ranges
      (e.g., elbow: 0-160 deg, knee: 0-150 deg).
    Implemented as soft loss terms (L_physical) added to L_diffusion.
    """

    def __init__(self, joint_pairs: list, angle_triplets: list, mean_lengths: torch.Tensor):
        """
        Args:
            joint_pairs: List of (idx_i, idx_j) pairs for length constraint
            angle_triplets: List of (idx_a, idx_b, idx_c) triplets for angle constraint
                            where angle is at vertex b
            mean_lengths: Mean distances for each joint pair (len(joint_pairs),)
        """
        super().__init__()
        self.joint_pairs = joint_pairs
        self.angle_triplets = angle_triplets
        self.register_buffer('mean_lengths', mean_lengths)

    def forward(self, repaired_kpts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            repaired_kpts: (B, T, N, 2) repaired keypoint coordinates
        Returns:
            L_physical: scalar loss
        """
        loss_length = 0.0
        for i, (j1, j2) in enumerate(self.joint_pairs):
            dist = torch.norm(repaired_kpts[:, :, j1] - repaired_kpts[:, :, j2], dim=-1)
            ratio = dist / (self.mean_lengths[i] + 1e-6)
            # Penalize deviation beyond ±15%
            violation = torch.clamp(torch.abs(ratio - 1.0) - 0.15, min=0.0)
            loss_length += violation.mean()

        loss_angle = 0.0
        for (j1, j2, j3) in self.angle_triplets:
            v1 = repaired_kpts[:, :, j1] - repaired_kpts[:, :, j2]
            v2 = repaired_kpts[:, :, j3] - repaired_kpts[:, :, j2]
            cos_sim = F.cosine_similarity(v1, v2, dim=-1)
            # Penalize extreme angles outside biomechanically plausible ranges
            # Paper: elbow 30°–160°; cos(30°)≈0.866, cos(160°)≈-0.940
            loss_angle += torch.mean(torch.clamp(cos_sim - 0.866, min=0.0))
            loss_angle += torch.mean(torch.clamp(-0.94 - cos_sim, min=0.0))

        n_pairs = max(len(self.joint_pairs), 1)
        n_triplets = max(len(self.angle_triplets), 1)
        return loss_length / n_pairs + loss_angle / n_triplets


# =============================================================================
# CSDI Model
# =============================================================================
class CSDI(nn.Module):
    """
    Conditional Score-based Diffusion model for keypoint Imputation (CSDI).

    Implements the forward process (Eqs. 1-2) and reverse process (Eqs. 3-6)
    from the manuscript, with DDIM sampling for accelerated inference.

    Key equations:
      Eq. 1: X_obs = X_0 * M + eps * (1 - M)   — observed signal
      Eq. 2: q(x_t | x_{t-1}) = N(sqrt(1-beta_t)*x_{t-1}, beta_t*I)  — forward
      Eq. 3: p_theta(x_{t-1}^ta | x_t^ta, x_t^co)  — reverse (conditional)
      Eq. 4: mu_theta = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_t))*eps_theta)  — DDPM
      Eq. 5: sigma_theta  — variance schedule
      Eq. 6: DDIM deterministic sampling (Song et al. 2020)

    Args:
        num_keypoints: Number of keypoints per entity (17 for worker)
        num_objects: Number of entities (2 for worker + equipment)
        keypoint_dim: Dimension of keypoint coordinates (2 for x,y)
        T: Number of diffusion timesteps (default: 1000)
        d_model: Base channel dimension (default: 64)
    """

    def __init__(self, num_keypoints: int = 17, num_objects: int = 2,
                 keypoint_dim: int = 2, T: int = 1000, d_model: int = 64):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_objects = num_objects
        self.keypoint_dim = keypoint_dim
        self.num_timesteps = T

        self.schedule = LinearNoiseSchedule(num_timesteps=T)
        self.denoiser = TemporalUNet(
            keypoint_dim=keypoint_dim,
            num_keypoints=num_keypoints,
            num_objects=num_objects,
            T=64,
            d_model=d_model,
        )

    def _get_observed_signal(self, x0, mask, noise):
        """Eq. 1: X_obs = X_0 * M + eps * (1 - M)"""
        mask_exp = mask.unsqueeze(-1)  # (B, T, N, 1)
        return x0 * mask_exp + noise * (1 - mask_exp)

    def forward_process(self, x0: torch.Tensor, mask: torch.Tensor, t: torch.Tensor):
        """
        Forward diffusion process (Eqs. 1-2):
        Add Gaussian noise, compute observed signal.

        Args:
            x0: Original keypoint sequence (B, T_seq, N, 2)
            mask: Binary mask, 1=visible, 0=occluded (B, T_seq, N)
            t: Timestep indices (B,)
        Returns:
            x_noisy: Noisy keypoint sequence (B, T_seq, N, 2)
            noise: The noise added (B, T_seq, N, 2)
            x_obs: Observed signal per Eq.1 (B, T_seq, N, 2)
        """
        noise = torch.randn_like(x0)

        sqrt_alpha = self.schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(x0.device)
        sqrt_one_minus_alpha = self.schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(x0.device)

        # Forward process: x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * eps
        x_noisy = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

        # Observed signal (Eq. 1): X_obs = X_0 * M + eps * (1 - M)
        x_obs = self._get_observed_signal(x0, mask, noise)

        return x_noisy, noise, x_obs

    def training_loss(self, x0: torch.Tensor, mask: torch.Tensor,
                      phys_loss_fn=None, lambda_phys: float = 0.1):
        """
        Compute training loss: L_diffusion + lambda * L_physical.

        L_diffusion = MSE(eps_theta(x_t^ta, t | x_t^co), eps) on occluded positions

        Args:
            x0: Ground truth keypoint sequence (B, T_seq, N, 2)
            mask: Binary mask, 1=visible, 0=occluded (B, T_seq, N)
            phys_loss_fn: Optional PhysicalPlausibilityLoss instance
            lambda_phys: Weight for physical plausibility loss (default: 0.1)
        Returns:
            loss: Scalar training loss
        """
        B = x0.shape[0]
        device = x0.device

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # Forward process (Eqs. 1-2)
        x_noisy, noise, x_obs = self.forward_process(x0, mask, t)

        # Predict noise conditioned on observed keypoints
        # eps_theta(x_t^ta, t | x_t^co) — Eq. 4
        predicted_noise = self.denoiser(x_noisy, t, x_obs, mask.float())

        # Loss: MSE on predicted noise for occluded keypoints only
        occ_mask = (1 - mask).unsqueeze(-1)  # (B, T, N, 1)
        loss_diffusion = F.mse_loss(predicted_noise * occ_mask, noise * occ_mask)

        # Total loss
        loss = loss_diffusion

        # Add physical plausibility loss if provided
        if phys_loss_fn is not None:
            # Reconstruct x_0 prediction from noise prediction for physical check
            alpha_t = self.schedule.alphas_cumprod[t].view(-1, 1, 1, 1).to(device)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            x0_pred = (x_noisy - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
            loss_physical = phys_loss_fn(x0_pred)
            loss = loss + lambda_phys * loss_physical

        return loss

    @torch.no_grad()
    def repair(self, x_obs: torch.Tensor, mask: torch.Tensor,
               num_inference_steps: int = 50, device: str = 'cuda') -> torch.Tensor:
        """
        Reverse diffusion process for keypoint repair using DDIM sampling (Eq. 6).
        Reduces inference from 1000 to `num_inference_steps` steps.

        Args:
            x_obs: Observed (partially occluded) keypoint sequence (B, T_seq, N, 2)
                   Eq. 1: X_obs = X_0 * M + eps * (1 - M)
            mask: Binary mask, 1=visible, 0=occluded (B, T_seq, N)
            num_inference_steps: Number of DDIM sampling steps (default: 50)
            device: Device for computation
        Returns:
            x_repaired: Repaired keypoint sequence (B, T_seq, N, 2)
        """
        B, T_seq, N, D = x_obs.shape

        # DDIM timestep schedule (evenly spaced)
        step_size = self.num_timesteps // num_inference_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        # Initialize from pure noise for occluded positions
        x_t = torch.randn_like(x_obs)
        # Fill observed positions with observed values
        mask_expanded = mask.unsqueeze(-1)
        x_t = x_t * (1 - mask_expanded) + x_obs * mask_expanded

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)

            # Predict noise (conditioned on observed keypoints)
            predicted_noise = self.denoiser(x_t, t, x_obs, mask.float())

            # DDIM update (Eq. 6)
            alpha_t = self.schedule.alphas_cumprod[t_val]
            alpha_t_prev = self.schedule.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else 1.0

            # Predict x_0: x_0 = (x_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
            x0_pred = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)

            # Keep observed keypoints fixed (they are the condition)
            x0_pred = x0_pred * (1 - mask_expanded) + x_obs * mask_expanded

            if i < len(timesteps) - 1:
                # DDIM deterministic step:
                # x_{t-1} = sqrt(alpha_{t-1}) * x_0_pred + sqrt(1-alpha_{t-1}) * eps_pred
                x_t = (
                    torch.sqrt(alpha_t_prev) * x0_pred
                    + torch.sqrt(1 - alpha_t_prev) * predicted_noise
                )
            else:
                x_t = x0_pred

        return x_t


# =============================================================================
# Mask Generation Strategies (described in manuscript Section 3.3)
# Hybrid: 70% random rectangular + 30% structured limb-part
# Mask ratio: 20%–70% of keypoints per frame
# =============================================================================
def generate_training_mask(shape: tuple, strategy: str = 'hybrid',
                           mask_ratio_range: tuple = (0.2, 0.7)) -> torch.Tensor:
    """
    Generate occlusion masks for training the CSDI model.

    Hybrid strategy: 70% random rectangular + 30% structured limb-part

    Args:
        shape: (B, T, N) mask shape
        strategy: 'random', 'structured', 'full_entity', or 'hybrid'
        mask_ratio_range: (min_ratio, max_ratio) of keypoints to mask per frame
    Returns:
        mask: Binary mask, 1=visible, 0=occluded (B, T, N)
    """
    B, T, N = shape
    mask = torch.ones(shape, dtype=torch.float32)
    min_r, max_r = mask_ratio_range

    if strategy == 'hybrid':
        # Paper: 70% random rectangular + 30% structured limb-part (2 strategies)
        rand = torch.rand(B)
        for b in range(B):
            if rand[b] < 0.7:
                m = generate_training_mask((1, T, N), 'random', mask_ratio_range)
            else:
                m = generate_training_mask((1, T, N), 'structured', mask_ratio_range)
            mask[b] = m.squeeze(0)
        return mask

    target_ratio = torch.rand(B) * (max_r - min_r) + min_r

    if strategy == 'random':
        # Random rectangular occlusion on keypoint sequence
        for b in range(B):
            num_mask = int(N * target_ratio[b])
            idx = torch.randperm(N)[:num_mask]
            mask[b, :, idx] = 0

    elif strategy == 'structured':
        # Structured limb-part masking (simulating self-occlusion)
        # Worker keypoints (13): head(0), l_shoulder(1), r_shoulder(2),
        #   l_elbow(3), r_elbow(4), l_wrist(5), r_wrist(6),
        #   l_hip(7), r_hip(8), l_knee(9), r_knee(10), l_ankle(11), r_ankle(12)
        limb_groups = {
            'left_arm': [3, 5],       # left elbow, left wrist
            'right_arm': [4, 6],      # right elbow, right wrist
            'left_leg': [9, 11],      # left knee, left ankle
            'right_leg': [10, 12],    # right knee, right ankle
            'torso': [1, 2, 7, 8],    # shoulders + hips
        }
        for b in range(B):
            num_groups = max(1, int(len(limb_groups) * target_ratio[b]))
            selected = np.random.choice(list(limb_groups.keys()), size=num_groups, replace=False)
            for group in selected:
                for t in range(T):
                    mask[b, t, limb_groups[group]] = 0

    elif strategy == 'full_entity':
        # Full entity masking (simulating complete occlusion)
        for b in range(B):
            kpts_per_entity = N // 2  # Assuming 2 entities (worker + equipment)
            if target_ratio[b] > 0.5:
                mask[b, :, :kpts_per_entity] = 0
            else:
                mask[b, :, kpts_per_entity:] = 0

    return mask


# =============================================================================
# Training Script
# =============================================================================
def train_csdi(model: CSDI, dataloader, num_epochs: int = 100,
               lr: float = 1e-4, lambda_phys: float = 0.1,
               phys_loss_fn=None, device: str = 'cuda'):
    """
    Train the CSDI model on complete keypoint sequences.

    The model learns to repair occluded keypoints by training with
    artificially generated masks on complete sequences.

    Total loss: L_total = L_diffusion + lambda * L_physical
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            # batch: (B, T, N, 2) complete keypoint sequences
            x0 = batch.to(device)
            B, T, N, D = x0.shape

            # Generate random training masks (hybrid strategy)
            mask = generate_training_mask((B, T, N), strategy='hybrid').to(device)

            # Compute loss (L_diffusion + lambda * L_physical)
            loss = model.training_loss(x0, mask, phys_loss_fn=phys_loss_fn, lambda_phys=lambda_phys)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

    return model


# =============================================================================
# Inference Script
# =============================================================================
def repair_keypoints(model: CSDI, x_obs: torch.Tensor, mask: torch.Tensor,
                     num_inference_steps: int = 50, device: str = 'cuda') -> torch.Tensor:
    """
    Repair occluded keypoints using trained CSDI model with DDIM sampling.

    Args:
        model: Trained CSDI model
        x_obs: Observed keypoint sequence (B, T, N, 2)
               Observed positions have actual coords, occluded positions have 0
        mask: Binary mask, 1=visible, 0=occluded (B, T, N)
        num_inference_steps: DDIM sampling steps (default: 50)
        device: Computation device
    Returns:
        x_repaired: Complete keypoint sequence (B, T, N, 2)
    """
    model.eval()
    x_obs = x_obs.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        x_repaired = model.repair(x_obs, mask, num_inference_steps, device)

    return x_repaired


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Model configuration (matching paper: Section 4.1)
    model = CSDI(
        num_keypoints=17,  # 13 worker + 4 equipment keypoints per entity pair
        num_objects=2,     # worker + equipment
        keypoint_dim=2,    # x, y coordinates
        T=1000,            # diffusion timesteps (beta_1=1e-4, beta_T=0.02)
        d_model=64,        # base channel dimension
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    # Dummy data for testing
    B, T_seq, N = 4, 64, 34  # batch=4, 64 frames (8fps x 8s), 34 keypoints (17 per entity)
    x0 = torch.randn(B, T_seq, N, 2).to(device)
    mask = generate_training_mask((B, T_seq, N), strategy='hybrid').to(device)

    # Training step
    loss = model.training_loss(x0, mask)
    print(f"Training loss: {loss.item():.6f}")

    # Inference (Eq. 1 observed signal, then DDIM 50-step repair)
    x_obs = x0 * mask.unsqueeze(-1)  # Eq. 1: X_obs = X_0 * M (eps=0 for observed)
    x_repaired = repair_keypoints(model, x_obs, mask, num_inference_steps=50, device=device)
    print(f"Repaired keypoints shape: {x_repaired.shape}")
    print("CSDI model test passed!")
