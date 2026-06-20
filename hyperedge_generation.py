"""
Dynamic Hyperedge Generation Algorithm (Algorithm 1 in Manuscript)
==================================================================

This module implements the dynamic hyperedge generation algorithm that constructs
interaction hypergraphs from STA-GCN attention heatmaps and repaired keypoint
coordinates, as described in Algorithm 1 of the manuscript.

The algorithm proceeds in 5 stages:
  Stage 1: Heatmap preprocessing (Gaussian blur + Otsu thresholding)
  Stage 2: Salient region extraction (contour detection + area filtering)
  Stage 3: Keypoint-region matching (point-in-rectangle test)
  Stage 4: Initial hyperedge generation (multi-keypoint regions → hyperedges)
  Stage 5: Spatial neighborhood merging (merge adjacent hyperedges)
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Set


def dynamic_hyperedge_generation(
    attention_heatmap: np.ndarray,
    keypoints: List[Tuple[float, float]],
    tau_area: float = 50.0,
    d_merge: float = 30.0,
    sigma: float = 2.5,
) -> List[Tuple[int, ...]]:
    """
    Dynamic Hyperedge Generation Algorithm (Algorithm 1).

    Constructs interaction hyperedges from STA-GCN attention heatmaps
    and repaired keypoint coordinates.

    Args:
        attention_heatmap: Attention heatmap from last layer of STA-GCN, shape (H, W).
                           Values should be in [0, 255] or [0, 1].
        keypoints: List of repaired keypoint coordinates [(x_i, y_i), ...].
        tau_area: Minimum region area threshold in pixels^2 (default: 50).
        d_merge: Maximum pixel distance to merge adjacent peaks (default: 30).
        sigma: Gaussian blur standard deviation (default: 2.5).

    Returns:
        E: Set of hyperedges, each hyperedge is a tuple of keypoint indices.
           e.g., [(0, 5, 13), (2, 8), ...]

    Reference: Algorithm 1 in the manuscript (steps 1-29).
    """

    # =========================================================================
    # Stage 1: Heatmap Preprocessing (Steps 1-3)
    # =========================================================================
    A_smooth = gaussian_blur(attention_heatmap, sigma=sigma)           # Step 1
    M = otsu_threshold(A_smooth)                                       # Step 2
    contours = find_contours(M)                                        # Step 3

    # =========================================================================
    # Stage 2: Salient Region Extraction (Steps 4-10)
    # =========================================================================
    R = []  # List of (x, y, w, h) bounding rectangles
    for c in contours:                                                 # Steps 5-9
        x, y, w, h = cv2.boundingRect(c)                              # Step 6
        if w * h >= tau_area:                                          # Step 7
            R.append((x, y, w, h))                                     # Step 8
    # End steps 9-10

    # =========================================================================
    # Stage 3: Keypoint-Region Matching (Steps 11-19)
    # =========================================================================
    region_map: Dict[int, List[int]] = {i: [] for i in range(len(R))}  # Step 11
    V = len(keypoints)

    for i in range(V):                                                 # Step 12
        xi, yi = keypoints[i]                                          # Step 13
        for r_idx, (x, y, w, h) in enumerate(R):                      # Step 14
            if x <= xi < x + w and y <= yi < y + h:                   # Step 15
                region_map[r_idx].append(i)                            # Step 16

    # =========================================================================
    # Stage 4: Initial Hyperedge Generation (Steps 20-27)
    # =========================================================================
    E_raw: List[Tuple[int, ...]] = []                                  # Step 20

    for r_idx in range(len(R)):                                        # Step 21
        K = region_map[r_idx]                                          # Step 22
        if len(K) >= 2:                                                # Step 23
            e = tuple(sorted(K))                                       # Step 24
            E_raw.append(e)                                            # Step 25

    # =========================================================================
    # Stage 5: Spatial Neighborhood Merging (Steps 28-29)
    # =========================================================================
    # Build mapping from R indices to E_raw indices (only regions with >=2 kpts)
    region_kpt_count = [len(region_map[r_idx]) for r_idx in range(len(R))]
    E = merge_hyperedges(E_raw, R, d_merge, region_kpt_count)          # Step 28

    return E                                                           # Step 29


# =============================================================================
# Helper Functions
# =============================================================================

def gaussian_blur(heatmap: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    """Step 1: Apply Gaussian blur to suppress noise in the attention heatmap."""
    # Normalize to uint8 if needed
    if heatmap.max() <= 1.0:
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    else:
        heatmap_uint8 = heatmap.astype(np.uint8)

    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(heatmap_uint8, (ksize, ksize), sigma)
    return blurred


def otsu_threshold(heatmap: np.ndarray) -> np.ndarray:
    """Step 2: Apply Otsu thresholding to binarize the smoothed heatmap."""
    _, binary = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def find_contours(binary_mask: np.ndarray) -> list:
    """Step 3: Extract contour boundaries from the binary mask."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def merge_hyperedges(
    E_raw: List[Tuple[int, ...]],
    R: List[Tuple[int, int, int, int]],
    d_merge: float,
    region_kpt_count: List[int] = None,
) -> List[Tuple[int, ...]]:
    """
    Step 28: Merge hyperedges whose corresponding region centers are within
    d_merge pixels of each other, to reduce over-segmentation.

    Uses Union-Find for efficient merging.

    Args:
        E_raw: List of hyperedges (only from regions with >= 2 keypoints).
        R: List of all salient regions (x, y, w, h).
        d_merge: Maximum distance for merging.
        region_kpt_count: Number of keypoints matched to each region in R.
                          Used to map E_raw indices back to R indices.
    """
    if len(E_raw) <= 1:
        return E_raw

    # Compute region centers for ALL regions
    centers = []
    for (x, y, w, h) in R:
        cx = x + w / 2.0
        cy = y + h / 2.0
        centers.append((cx, cy))

    # Build mapping from E_raw index to R index
    # E_raw contains only regions where len(K) >= 2
    if region_kpt_count is not None:
        e_to_r = []
        for r_idx, count in enumerate(region_kpt_count):
            if count >= 2:
                e_to_r.append(r_idx)
    else:
        # Fallback: assume E_raw indices correspond to first len(E_raw) regions
        e_to_r = list(range(len(E_raw)))

    # Union-Find
    parent = list(range(len(E_raw)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    # Check pairwise distances using correct R center indices
    for i in range(len(E_raw)):
        for j in range(i + 1, len(E_raw)):
            r_i = e_to_r[i] if i < len(e_to_r) else i
            r_j = e_to_r[j] if j < len(e_to_r) else j
            ci = centers[r_i] if r_i < len(centers) else (0, 0)
            cj = centers[r_j] if r_j < len(centers) else (0, 0)
            dist = np.sqrt((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2)
            if dist < d_merge:
                union(i, j)

    # Group merged hyperedges
    groups: Dict[int, Set[int]] = {}
    for i in range(len(E_raw)):
        root = find(i)
        if root not in groups:
            groups[root] = set()
        groups[root].update(E_raw[i])

    # Convert to sorted tuples
    E = [tuple(sorted(g)) for g in groups.values() if len(g) >= 2]
    return E


def build_incidence_matrix(
    hyperedges: List[Tuple[int, ...]],
    num_keypoints: int,
) -> np.ndarray:
    """
    Build the incidence matrix H from the hyperedge set.

    H[i][j] = 1 if vertex i belongs to hyperedge j, else 0.

    This is used by the hypergraph learning model (Eq. 9).
    """
    num_hyperedges = len(hyperedges)
    H = np.zeros((num_keypoints, num_hyperedges), dtype=np.float32)

    for j, edge in enumerate(hyperedges):
        for vertex_idx in edge:
            if vertex_idx < num_keypoints:
                H[vertex_idx, j] = 1.0

    return H


def build_hypergraph_laplacian(H: np.ndarray) -> np.ndarray:
    """
    Compute the hypergraph Laplacian matrix (Eq. 9).

    L = I - D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}

    where:
        D_v: Diagonal vertex degree matrix
        D_e: Diagonal hyperedge degree matrix
    """
    num_vertices = H.shape[0]

    # Vertex degree: D_v = diag(H * 1)
    d_v = H.sum(axis=1)
    D_v_inv_sqrt = np.diag(np.power(d_v + 1e-8, -0.5))

    # Edge degree: D_e = diag(H^T * 1)
    d_e = H.sum(axis=0)
    D_e_inv = np.diag(np.power(d_e + 1e-8, -1.0))

    # Hypergraph Laplacian
    I = np.eye(num_vertices, dtype=np.float32)
    L = I - D_v_inv_sqrt @ H @ D_e_inv @ H.T @ D_v_inv_sqrt

    return L


# =============================================================================
# Visualization Helper
# =============================================================================
def visualize_hypergraph(
    image: np.ndarray,
    keypoints: List[Tuple[float, float]],
    hyperedges: List[Tuple[int, ...]],
    physical_edges: List[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Visualize the constructed hypergraph on the input image.

    White edges: Physical connections (from pose estimator)
    Red edges:   Interaction hyperedges (from Algorithm 1)
    """
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # Draw physical edges (white)
    if physical_edges:
        for (i, j) in physical_edges:
            pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(vis, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw interaction hyperedges (red)
    for edge in hyperedges:
        pts = [(int(keypoints[idx][0]), int(keypoints[idx][1])) for idx in edge]
        # Draw all pairwise connections within the hyperedge
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                cv2.line(vis, pts[a], pts[b], (0, 0, 255), 2, cv2.LINE_AA)

    # Draw keypoints
    for i, (x, y) in enumerate(keypoints):
        cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)

    return vis


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == '__main__':
    # Simulate attention heatmap (H=640, W=640)
    np.random.seed(42)
    H, W = 640, 640
    heatmap = np.zeros((H, W), dtype=np.float32)

    # Add attention peaks around interaction regions
    # Simulate "worker pushing handcart" interaction
    # Worker keypoints around (200, 300), handcart keypoints around (350, 350)
    for cx, cy in [(200, 300), (280, 320), (350, 350)]:
        for dx in range(-40, 40):
            for dy in range(-40, 40):
                dist = np.sqrt(dx**2 + dy**2)
                heatmap[cy + dy, cx + dx] += max(0, 1.0 - dist / 40.0)

    heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)

    # Simulate repaired keypoints (worker: 13 kpts, handcart: 4 kpts = 17 total)
    keypoints = [
        # Worker keypoints (13)
        (180, 260), (200, 280), (220, 280),  # nose, left_shoulder, right_shoulder
        (175, 310), (225, 310),              # left_elbow, right_elbow
        (170, 340), (230, 340),              # left_wrist, right_wrist
        (190, 400), (210, 400),              # left_hip, right_hip
        (185, 440), (215, 440),              # left_knee, right_knee
        (190, 480),                          # left_ankle
        # Handcart keypoints (4)
        (300, 340), (380, 340),              # handle_left, handle_right
        (300, 380), (380, 380),              # wheel_left, wheel_right
    ]

    # Run Algorithm 1
    hyperedges = dynamic_hyperedge_generation(
        attention_heatmap=heatmap,
        keypoints=keypoints,
        tau_area=50.0,
        d_merge=30.0,
        sigma=2.5,
    )

    print(f"Generated {len(hyperedges)} hyperedges:")
    for i, e in enumerate(hyperedges):
        involved = [f"kp{j}" for j in e]
        print(f"  Hyperedge {i+1}: {involved}")

    # Build incidence matrix and Laplacian
    H_mat = build_incidence_matrix(hyperedges, num_keypoints=len(keypoints))
    L = build_hypergraph_laplacian(H_mat)
    print(f"\nIncidence matrix shape: {H_mat.shape}")
    print(f"Hypergraph Laplacian shape: {L.shape}")
    print("Dynamic Hyperedge Generation test passed!")
