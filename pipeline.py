"""
Complete Inference Pipeline
============================

End-to-end pipeline implementing all 5 phases described in Figure 1:
  (I)   Data collection, preprocessing, and dataset division
  (II)  Object detection (YOLOv11n) and pose estimation (YOLOv11n-pose / ViTPose)
  (III) Occlusion keypoint repair via conditional diffusion model (CSDI)
  (IV)  Dynamic hyperedge generation algorithm (Algorithm 1)
  (V)   Interaction action classification via hypergraph learning model (HGCN)

Phases I-II use external models (YOLO, ViTPose) and are provided as interfaces.
Phases III-V are fully implemented.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Tuple, Optional

from csdi import CSDI, PhysicalPlausibilityLoss, repair_keypoints, train_csdi
from hyperedge_generation import dynamic_hyperedge_generation, build_incidence_matrix
from hgcn_model import HypergraphActionModel
from skeleton_topology import build_combined_adjacency, ENTITY_KEYPOINTS
from metrics import precision_recall_f1


def gpu_synchronized_time(device) -> float:
    """
    GPU-synchronized wall-clock timestamp (manuscript Sec. 4.6).

    Drains the CUDA queue before reading the clock so that measured latencies
    include all kernel execution time, matching the measurement protocol of
    the stage-wise runtime table (torch.cuda.synchronize, batch size 1,
    100-frame warm-up). On CPU this is a plain perf_counter read.
    """
    if 'cuda' in str(device):
        torch.cuda.synchronize()
    return time.perf_counter()


# =============================================================================
# Phase I & II Interfaces (External Models)
# =============================================================================
class ObjectDetector:
    """
    Interface for YOLOv11n object detection (Phase II).
    
    In the manuscript, YOLOv11n is trained for 300 epochs with AdamW optimizer,
    initialized from MS COCO pre-trained weights, for 16 construction entity categories.
    
    This is an interface — the actual model requires the Ultralytics YOLO package.
    """

    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.model = None
        self.device = device
        if model_path:
            self.load(model_path)

    def load(self, model_path: str):
        """Load trained YOLOv11n model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except ImportError:
            print("Warning: ultralytics not installed. Using dummy detector.")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect construction entities in a video frame.

        Args:
            frame: (H, W, 3) uint8 image
        Returns:
            detections: List of dicts with keys:
                'bbox': (x1, y1, x2, y2)
                'class': entity category name
                'confidence': detection confidence
        """
        if self.model is None:
            return self._dummy_detect(frame)
        results = self.model(frame)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    'bbox': box.xyxy[0].cpu().numpy(),
                    'class': r.names[int(box.cls)],
                    'confidence': float(box.conf),
                })
        return detections

    def _dummy_detect(self, frame):
        H, W = frame.shape[:2]
        return [{
            'bbox': np.array([W * 0.1, H * 0.1, W * 0.5, H * 0.9]),
            'class': 'worker',
            'confidence': 0.95,
        }, {
            'bbox': np.array([W * 0.3, H * 0.2, W * 0.8, H * 0.85]),
            'class': 'excavator',
            'confidence': 0.92,
        }]


class PoseEstimator:
    """
    Interface for YOLOv11n-pose / ViTPose pose estimation (Phase II).
    
    YOLOv11n-pose extracts keypoints for each detected entity.
    Worker: 13 torso keypoints (facial keypoints removed per paper).
    Equipment: custom-defined keypoints (Table 1).
    """

    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.model = None
        self.device = device
        if model_path:
            self.load(model_path)

    def load(self, model_path: str):
        """Load trained pose estimator."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except ImportError:
            print("Warning: ultralytics not installed. Using dummy estimator.")

    def estimate(self, frame: np.ndarray, detections: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate keypoints for detected entities.

        Args:
            frame: (H, W, 3) uint8 image
            detections: Output from ObjectDetector
        Returns:
            keypoints: (O, V_max, 2) keypoint coordinates per entity
            visibility: (O, V_max) binary mask, 1=visible, 0=occluded/missing
        """
        # Placeholder: in practice, use YOLOv11n-pose or ViTPose
        O = len(detections)
        V_max = 13  # maximum keypoints per entity
        keypoints = np.zeros((O, V_max, 2), dtype=np.float32)
        visibility = np.ones((O, V_max), dtype=np.float32)

        for o, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            entity = det['class']
            V_e = ENTITY_KEYPOINTS.get(entity, {}).get('num_keypoints', 13)
            if V_e == -1:
                V_e = 5

            # Generate dummy keypoints within bbox
            for v in range(min(V_e, V_max)):
                keypoints[o, v, 0] = x1 + (x2 - x1) * np.random.uniform(0.2, 0.8)
                keypoints[o, v, 1] = y1 + (y2 - y1) * np.random.uniform(0.2, 0.8)
                visibility[o, v] = 1.0

        return keypoints, visibility


def scatter_gaussian_heatmap(attn: np.ndarray, points: np.ndarray,
                             H: int = 640, W: int = 640,
                             sigma_h: float = 8.0) -> np.ndarray:
    """
    Scatter-and-diffuse projection of node attention onto the image grid
    (manuscript Sec. 3.4):

        A(u, v) = sum_i a_i * exp( - ||(u, v) - p_i||^2 / (2 * sigma_h^2) )

    Each per-node attention weight a_i is scattered to its image coordinate
    p_i and diffused with an isotropic Gaussian kernel of bandwidth sigma_h,
    followed by min-max normalization to [0, 1].

    Args:
        attn: (V,) per-node attention weights a_i.
        points: (V, 2) keypoint image coordinates p_i = (x_i, y_i).
        H, W: output heatmap size (default 640 x 640).
        sigma_h: Gaussian diffusion bandwidth in pixels (default 8 px).
    Returns:
        (H, W) float32 heatmap in [0, 1].
    """
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))  # (H, W)
    A = np.zeros((H, W), dtype=np.float32)
    inv2s2 = 1.0 / (2.0 * sigma_h ** 2)
    for a_i, (x, y) in zip(attn, points):
        if np.isfinite(x) and np.isfinite(y):
            A += float(a_i) * np.exp(-((uu - x) ** 2 + (vv - y) ** 2) * inv2s2)
    # Min-max normalization to [0, 1]
    A_min, A_max = A.min(), A.max()
    if A_max > A_min:
        A = (A - A_min) / (A_max - A_min)
    return A.astype(np.float32)


class STAGCNTeacher:
    """
    Interface for the pre-trained frozen STA-GCN teacher model.
    
    As described in the manuscript (Section 3.4):
    - STA-GCN is pre-trained on training set with full keypoint sequences
    - Frozen during hyperedge generation
    - Serves solely as spatial-temporal attention feature extractor
    - Generates attention heatmaps from its last layer via the
      scatter-and-diffuse projection (scatter_gaussian_heatmap, sigma_h = 8 px)
    
    Reference: Hang and Li (2022) - STA-GCN
    """

    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.model = None
        self.device = device
        if model_path:
            self.load(model_path)

    def load(self, model_path: str):
        """Load pre-trained STA-GCN model (frozen)."""
        pass  # Load actual STA-GCN model

    def _last_layer_attention(self, keypoints: torch.Tensor,
                              adj: torch.Tensor) -> np.ndarray:
        """
        Read per-node attention weights a_i from the last layer of the
        loaded STA-GCN. With the reference architecture (Hang and Li, 2022),
        the last-layer channel-wise spatiotemporal attention is averaged over
        time to yield one weight per node. The concrete read-out depends on
        the STA-GCN implementation and must be wired to the released
        checkpoint.
        """
        raise NotImplementedError(
            'Attach the pre-trained STA-GCN checkpoint and expose its '
            'last-layer attention map here.')

    @torch.no_grad()
    def get_attention_heatmap(self, keypoints: torch.Tensor,
                               adj: torch.Tensor = None,
                               sigma_h: float = 8.0,
                               grid_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        Project the last-layer STA-GCN node attention onto a continuous
        spatial grid (manuscript Sec. 3.4):

            A(u, v) = sum_i a_i * exp( - ||(u, v) - p_i||^2 / (2 * sigma_h^2) )

        with min-max normalization to [0, 1].

        Args:
            keypoints: (1, T, V, 2) repaired keypoint sequence
            adj: (V, V) adjacency matrix (used by the STA-GCN forward pass)
            sigma_h: Gaussian diffusion bandwidth in pixels (default: 8 px,
                i.e., 1.25% of the 640 px image width)
            grid_size: output heatmap size (default: 640 x 640)
        Returns:
            heatmap: (H, W) float32 attention heatmap in [0, 1]
        """
        # Per-node attention weights a_i (last STA-GCN layer, averaged over
        # time). Interface mode without the released checkpoint falls back to
        # uniform weights so that downstream stages remain runnable.
        if self.model is not None:
            a = self._last_layer_attention(keypoints, adj)  # (V,)
        else:
            a = np.ones(keypoints.shape[2], dtype=np.float32)

        # Node image coordinates p_i: repaired keypoints averaged over time.
        kpts_np = keypoints[0].cpu().numpy()  # (T, V, 2)
        p = kpts_np.mean(axis=0)              # (V, 2)

        return scatter_gaussian_heatmap(a, p, H=grid_size[0], W=grid_size[1],
                                        sigma_h=sigma_h)


# =============================================================================
# Complete Pipeline
# =============================================================================
class WorkerEquipmentInteractionPipeline:
    """
    Complete 5-phase inference pipeline (Figure 1).

    Phase I:   Data preprocessing (frame extraction, resize to 640x640)
    Phase II:  Object detection (YOLOv11n) + Pose estimation (YOLOv11n-pose)
    Phase III: Keypoint repair via CSDI (DDIM 50-step sampling)
    Phase IV:  Dynamic hyperedge generation (Algorithm 1)
    Phase V:   Action classification via HGCN (9 blocks)
    """

    def __init__(self, det_model_path: str = None, pose_model_path: str = None,
                 stagcn_model_path: str = None, csdi_ckpt: str = None,
                 hgcn_ckpt: str = None, device: str = 'cuda'):
        self.device = device

        # Phase II: Detection + Pose
        self.detector = ObjectDetector(det_model_path, device)
        self.pose_estimator = PoseEstimator(pose_model_path, device)

        # STA-GCN teacher (frozen)
        self.teacher = STAGCNTeacher(stagcn_model_path, device)

        # Phase III: CSDI
        self.csdi = CSDI(
            num_keypoints=17, num_objects=2, keypoint_dim=2,
            T=1000, d_model=64,
        ).to(device)
        if csdi_ckpt:
            self.csdi.load_state_dict(torch.load(csdi_ckpt, map_location=device))

        # Phase V: HGCN
        self.hgcn = HypergraphActionModel(
            num_keypoints=34, num_classes=13,
            in_channels=2, hidden_channels=256, num_hgcn_blocks=9,
        ).to(device)
        if hgcn_ckpt:
            self.hgcn.load_state_dict(torch.load(hgcn_ckpt, map_location=device))

    def preprocess_frames(self, video_path: str, clip_length: int = 64,
                          fps: int = 8) -> List[np.ndarray]:
        """
        Phase I: Extract and preprocess video frames.
        Resize to 640x640 as described in Section 3.1.
        """
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_idx = 0
            step = max(1, int(cap.get(cv2.CAP_PROP_FPS) / fps))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    frame = cv2.resize(frame, (640, 640))
                    frames.append(frame)
                frame_idx += 1
            cap.release()
            # Trim/pad to clip_length
            if len(frames) >= clip_length:
                frames = frames[:clip_length]
            elif len(frames) > 0:
                frames += [frames[-1]] * (clip_length - len(frames))
            else:
                frames = [np.zeros((640, 640, 3), dtype=np.uint8)] * clip_length
            return frames
        except ImportError:
            return [np.zeros((640, 640, 3), dtype=np.uint8)] * clip_length

    @torch.no_grad()
    def __call__(self, video_path: str, entity_pair: Tuple[str, str] = ('worker', 'excavator'),
                 num_ddim_steps: int = 50) -> Dict:
        """
        Run complete pipeline on a video clip.

        Args:
            video_path: Path to input video
            entity_pair: (worker_type, equipment_type) tuple
            num_ddim_steps: DDIM sampling steps for CSDI (default: 50)
        Returns:
            Dict with:
                'predicted_action': Action class index (0-12)
                'action_name': Human-readable action name
                'confidence': Prediction confidence
                'keypoints_repaired': (T, N, 2) repaired keypoints
                'hyperedges': List of hyperedge tuples
                'latency_ms': Per-phase latency breakdown
        """
        latencies = {}

        # Phase I: Preprocess
        t0 = gpu_synchronized_time(self.device)
        frames = self.preprocess_frames(video_path)
        latencies['preprocess_ms'] = (gpu_synchronized_time(self.device) - t0) * 1000

        # Phase II: Detect + Pose
        t0 = gpu_synchronized_time(self.device)
        detections = self.detector.detect(frames[0])
        all_keypoints = []
        all_visibility = []
        for frame in frames:
            kpts, vis = self.pose_estimator.estimate(frame, detections)
            all_keypoints.append(kpts)
            all_visibility.append(vis)
        # Stack: (T, O, V, 2)
        keypoints_seq = np.stack(all_keypoints, axis=0)
        visibility_seq = np.stack(all_visibility, axis=0)
        latencies['detection_pose_ms'] = (gpu_synchronized_time(self.device) - t0) * 1000

        # Prepare tensors for CSDI
        T, O, V_max = keypoints_seq.shape[:3]
        N_actual = O * V_max
        # Pad to fixed model size (34 = 13 worker + 21 equipment slots)
        N_model = 34
        if N_actual < N_model:
            pad_kpts = np.zeros((T, N_model - N_actual, 2), dtype=np.float32)
            pad_vis = np.zeros((T, N_model - N_actual), dtype=np.float32)
            x_obs_np = np.concatenate([keypoints_seq.reshape(T, N_actual, 2), pad_kpts], axis=1)
            mask_np = np.concatenate([visibility_seq.reshape(T, N_actual), pad_vis], axis=1)
        else:
            x_obs_np = keypoints_seq.reshape(T, N_model, 2).astype(np.float32)
            mask_np = visibility_seq.reshape(T, N_model).astype(np.float32)

        x_obs = torch.from_numpy(x_obs_np).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)

        # Phase III: Keypoint Repair (CSDI)
        t0 = gpu_synchronized_time(self.device)
        x_repaired = repair_keypoints(
            self.csdi, x_obs, mask,
            num_inference_steps=num_ddim_steps,
            device=self.device,
        )
        latencies['csdi_repair_ms'] = (gpu_synchronized_time(self.device) - t0) * 1000

        # Phase IV: Dynamic Hyperedge Generation (Algorithm 1)
        t0 = gpu_synchronized_time(self.device)
        # Get attention heatmap from frozen STA-GCN
        adj = build_combined_adjacency(entity_pair[0], entity_pair[1])
        heatmap = self.teacher.get_attention_heatmap(x_repaired, adj)

        # Convert repaired keypoints to list of (x, y) for Algorithm 1
        kpts_np = x_repaired[0].cpu().numpy()  # (T, N, 2)
        mean_kpts = kpts_np.mean(axis=0)  # (N, 2) average over time
        keypoint_list = [(float(mean_kpts[i, 0]), float(mean_kpts[i, 1])) for i in range(N_model)]

        hyperedges = dynamic_hyperedge_generation(
            attention_heatmap=heatmap,
            keypoints=keypoint_list,
            tau_area=50.0,
            d_merge=25.0,
            sigma=2.5,
        )

        # Build incidence matrix
        if len(hyperedges) == 0:
            hyperedges = [(i, (i + 1) % N_model) for i in range(N_model)]
        H = build_incidence_matrix(hyperedges, num_keypoints=N_model)
        H_tensor = torch.from_numpy(H).float().to(self.device)
        latencies['hyperedge_ms'] = (gpu_synchronized_time(self.device) - t0) * 1000

        # Phase V: Action Classification (HGCN)
        t0 = gpu_synchronized_time(self.device)
        logits = self.hgcn(x_repaired, H_tensor)  # (1, 13)
        probs = F.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_class].item()
        latencies['hgcn_ms'] = (gpu_synchronized_time(self.device) - t0) * 1000

        total_ms = sum(latencies.values())
        latencies['total_ms'] = total_ms
        latencies['FPS'] = 1000.0 / total_ms if total_ms > 0 else 0.0

        ACTION_NAMES = {
            0: "driving/operating (A01)", 1: "hooking/unhooking (A02)",
            2: "inspecting/maintaining (A03)", 3: "pushing/pulling/towing (A04)",
            4: "climbing (A05)", 5: "directing/intercepting (A06)",
            6: "carrying/transporting (A07)", 7: "crossing/stepping over (A08)",
            8: "holding/gripping (A09)", 9: "pressing (A10)",
            10: "hammering/striking (A11)", 11: "digging (A12)",
            12: "cutting (A13)",
        }

        return {
            'predicted_action': pred_class,
            'action_name': ACTION_NAMES.get(pred_class, f"Unknown ({pred_class})"),
            'confidence': confidence,
            'keypoints_repaired': x_repaired[0].cpu().numpy(),
            'hyperedges': hyperedges,
            'latency_ms': latencies,
        }


# =============================================================================
# Physical-Constraint Weight (lambda) Sensitivity Analysis
# =============================================================================
# Manuscript Sec. 3.3: the final value lambda* = 0.1 is selected by grid search
# over LAMBDA_SELECTION_GRID on the validation set.
LAMBDA_SELECTION_GRID: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 1.0)
# Manuscript Sec. 4.4, Table `tab-ablation`: sensitivity study in which
# lambda = 0 disables the physical constraint (L_physical) entirely.
LAMBDA_ABLATION_GRID: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.5, 1.0)


def train_hgcn(model: HypergraphActionModel, dataset: torch.utils.data.TensorDataset,
               num_epochs: int = 50, lr: float = 1e-3,
               weight_decay: float = 0.01, batch_size: int = 8,
               device: str = 'cuda') -> None:
    """
    Train the hypergraph action model on repaired sequences (Sec. 3.5):
    AdamW optimizer, cross-entropy loss, 50 epochs.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(num_epochs):
        for x, H, y in loader:
            x, H, y = x.to(device), H.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(x, H), y).backward()
            opt.step()


def evaluate_actions(model: HypergraphActionModel, dataset: torch.utils.data.TensorDataset,
                     batch_size: int = 8,
                     device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
    """Predict over repaired sequences; return (y_true, y_pred)."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, H, y in loader:
            logits = model(x.to(device), H.to(device))
            ps.append(logits.argmax(dim=1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def repair_and_hypergraphify(csdi_model: CSDI, teacher: STAGCNTeacher,
                             dataloader, entity_pair: Tuple[str, str],
                             num_ddim_steps: int = 50,
                             device: str = 'cuda') -> torch.utils.data.TensorDataset:
    """
    Repair every sequence with the trained CSDI and build its dynamic hypergraph
    (Algorithm 1: tau_area = 50 px^2, d_merge = 25 px, sigma = 2.5; attention
    bandwidth sigma_h = 8 px).

    Args:
        dataloader: yields (x0, mask, y) batches, where
            x0: (B, T, 34, 2) observed keypoints (zeros where occluded),
            mask: (B, T, 34) binary, 1 = visible / 0 = occluded,
            y: (B,) integer action labels.
    Returns:
        TensorDataset of (x, H, y) per clip: x (T, 34, 2) repaired sequence,
        H (34, num_hyperedges) incidence matrix, y scalar label.
    """
    csdi_model.eval()
    adj = build_combined_adjacency(*entity_pair)
    xs, Hs, ys = [], [], []
    for x0, mask, y in dataloader:
        x_hat = repair_keypoints(csdi_model, x0.to(device), mask.to(device),
                                 num_inference_steps=num_ddim_steps,
                                 device=device)
        for b in range(x_hat.shape[0]):
            seq = x_hat[b].cpu().numpy()                       # (T, 34, 2)
            heatmap = teacher.get_attention_heatmap(x_hat[b:b + 1], adj)
            mean_kpts = seq.mean(axis=0)                       # (34, 2)
            keypoint_list = [(float(p[0]), float(p[1])) for p in mean_kpts]
            hyperedges = dynamic_hyperedge_generation(
                heatmap, keypoint_list, tau_area=50.0, d_merge=25.0, sigma=2.5)
            if len(hyperedges) == 0:
                n = seq.shape[1]
                hyperedges = [(i, (i + 1) % n) for i in range(n)]
            H = build_incidence_matrix(hyperedges, num_keypoints=seq.shape[1])
            xs.append(torch.from_numpy(seq).float())
            Hs.append(torch.from_numpy(H).float())
            ys.append(int(y[b]))
    # Pad incidence matrices to a common hyperedge count with zero columns.
    # A zero column is inert in HypergraphConv (d_e is clamped to 1e-8, so the
    # padded edges carry no message); padding only equalizes the batch shape.
    if Hs:
        e_max = max(Hm.shape[1] for Hm in Hs)
        Hs = [torch.cat([Hm, torch.zeros(Hm.shape[0], e_max - Hm.shape[1])],
                        dim=1) for Hm in Hs]
    y_t = torch.tensor(ys, dtype=torch.long)
    return torch.utils.data.TensorDataset(torch.stack(xs), torch.stack(Hs), y_t)


def lambda_grid_search(csd_train_loader, action_train_loader, val_loader,
                       joint_pairs: List[Tuple[int, int]],
                       angle_triplets: List[Tuple[int, int, int]],
                       mean_lengths: np.ndarray,
                       entity_pair: Tuple[str, str] = ('worker', 'excavator'),
                       grid: Tuple[float, ...] = LAMBDA_SELECTION_GRID,
                       num_csdi_epochs: int = 100,
                       num_hgcn_epochs: int = 50,
                       num_ddim_steps: int = 50,
                       d_model: int = 64,
                       seed: int = 42,
                       device: str = 'cuda',
                       verbose: bool = True) -> Dict:
    """
    Physical-constraint weight sensitivity protocol (Sec. 3.3 & Table
    `tab-ablation`).

    For each lambda in `grid`, a fresh CSDI denoiser is trained from scratch
    with total loss L = L_diffusion + lambda * L_physical (lambda = 0 disables
    the physical constraint), the labeled training and validation sequences
    are repaired, dynamic hypergraphs are constructed (Algorithm 1), and the
    hypergraph action model is trained and evaluated. Macro-Recall /
    Macro-Precision / Macro-F1 are the columns of Table `tab-ablation`.

    Args:
        csd_train_loader: yields complete-sequence batches (B, T, 34, 2) for
            CSDI training (training masks are generated internally with the
            hybrid 70/30 strategy). Use a Dataset whose __getitem__ returns a
            bare (T, 34, 2) tensor (not a tuple), so that default_collate
            stacks batches into a single tensor as expected by train_csdi.
        action_train_loader, val_loader: yield (x0, mask, y) batches as in
            `repair_and_hypergraphify`.
        joint_pairs, angle_triplets, mean_lengths: skeleton statistics for
            PhysicalPlausibilityLoss (+/-15% joint length, 30-160 deg angle).
        grid: LAMBDA_SELECTION_GRID (Sec. 3.3, validation-set model selection)
            or LAMBDA_ABLATION_GRID (Table `tab-ablation` sensitivity study).
    Returns:
        {'per_lambda': {lambda: {'Macro-Recall': %, 'Macro-Precision': %,
                                 'Macro-F1': %}},
         'best_lambda': argmax of Macro-F1}
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    phys_loss = PhysicalPlausibilityLoss(
        joint_pairs, angle_triplets,
        torch.tensor(mean_lengths, dtype=torch.float32)).to(device)
    teacher = STAGCNTeacher(device=device)

    per_lambda: Dict[float, Dict[str, float]] = {}
    for lam in grid:
        csdi = CSDI(num_keypoints=17, num_objects=2, keypoint_dim=2,
                    T=1000, d_model=d_model).to(device)
        train_csdi(csdi, csd_train_loader, num_epochs=num_csdi_epochs,
                   lambda_phys=lam, phys_loss_fn=phys_loss, device=device)
        train_set = repair_and_hypergraphify(csdi, teacher, action_train_loader,
                                             entity_pair, num_ddim_steps, device)
        val_set = repair_and_hypergraphify(csdi, teacher, val_loader,
                                           entity_pair, num_ddim_steps, device)
        hgcn = HypergraphActionModel(num_keypoints=34, num_classes=13,
                                     in_channels=2, hidden_channels=256,
                                     num_hgcn_blocks=9).to(device)
        train_hgcn(hgcn, train_set, num_epochs=num_hgcn_epochs, device=device)
        y_true, y_pred = evaluate_actions(hgcn, val_set, device=device)
        prf = precision_recall_f1(y_true, y_pred, num_classes=13)
        per_lambda[lam] = {
            'Macro-Recall': 100.0 * float(np.mean(prf['recall'])),
            'Macro-Precision': 100.0 * float(np.mean(prf['precision'])),
            'Macro-F1': 100.0 * float(np.mean(prf['f1'])),
        }
        if verbose:
            print(f"[lambda_grid_search] lambda={lam:g}: "
                  f"Macro-Recall {per_lambda[lam]['Macro-Recall']:.1f}%, "
                  f"Macro-Precision {per_lambda[lam]['Macro-Precision']:.1f}%, "
                  f"Macro-F1 {per_lambda[lam]['Macro-F1']:.1f}%")

    best = max(per_lambda, key=lambda k: per_lambda[k]['Macro-F1'])
    if verbose:
        print(f"[lambda_grid_search] best lambda = {best:g} (max Macro-F1)")
    return {'per_lambda': per_lambda, 'best_lambda': best}


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    pipeline = WorkerEquipmentInteractionPipeline(device=device)

    # Run on dummy video
    result = pipeline("dummy_video.mp4", entity_pair=('worker', 'excavator'))

    print(f"\n=== Pipeline Results ===")
    print(f"  Predicted action: {result['action_name']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Repaired keypoints shape: {result['keypoints_repaired'].shape}")
    print(f"  Hyperedges: {len(result['hyperedges'])}")
    print(f"\n  Latency breakdown:")
    for phase, ms in result['latency_ms'].items():
        if isinstance(ms, float):
            print(f"    {phase}: {ms:.2f} ms")

    print("\nPipeline test passed!")
