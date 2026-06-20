"""
Skeleton Topology Definitions for 16 Construction Entity Categories
====================================================================

Implements the keypoint sequences and physical adjacency (skeleton topology)
for all 16 entity categories defined in TABLE 1 of the manuscript:

- worker (13 keypoints)
- excavator (8 keypoints)
- crane (8 keypoints)
- forklift (8 keypoints)
- truck (6 keypoints)
- aerial work platform (8 keypoints)
- ladder (4 keypoints)
- scaffold (N keypoints)
- handcart (7 keypoints)
- wheelbarrow (7 keypoints)
- enclosure (6 keypoints)
- hammer (4 keypoints)
- pickaxe (4 keypoints)
- shovel (5 keypoints)
- saw (3 keypoints)
- drill (5 keypoints)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


# =============================================================================
# Keypoint Definitions per Entity (Table 1)
# =============================================================================
ENTITY_KEYPOINTS = {
    'worker': {
        'num_keypoints': 13,
        'keypoint_names': [
            'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle'
        ],
        # Physical skeleton edges: (i, j) pairs
        'edges': [
            (0, 1), (0, 2),        # nose -> shoulders
            (1, 3), (3, 5),        # left arm: shoulder -> elbow -> wrist
            (2, 4), (4, 6),        # right arm: shoulder -> elbow -> wrist
            (1, 7), (2, 8),        # shoulders -> hips
            (7, 9), (9, 11),       # left leg: hip -> knee -> ankle
            (8, 10), (10, 12),     # right leg: hip -> knee -> ankle
            (7, 8),                # hip connection
            (1, 2),                # shoulder connection
        ]
    },
    'excavator': {
        'num_keypoints': 8,
        'keypoint_names': [
            'bucket_tip', 'bucket_arm_connection', 'boom_joint',
            'arm_center', 'cab_front', 'cab_rear', 'front_track',
            'rear_track'
        ],
        'edges': [
            (0, 1), (1, 2), (2, 3),    # bucket -> arm connection -> boom -> arm center
            (3, 4), (4, 5),              # arm center -> cab
            (4, 6), (5, 7),             # tracks
            (6, 7),                      # track axle
        ]
    },
    'crane': {
        'num_keypoints': 8,
        'keypoint_names': [
            'hook_center', 'boom_tip', 'boom_joint',
            'cab_front', 'cab_rear', 'front_wheel', 'rear_wheel',
            'counterweight_center'
        ],
        'edges': [
            (0, 1), (1, 2), (2, 3),     # hook -> boom tip -> boom joint -> cab
            (3, 4), (4, 7),              # cab -> counterweight
            (3, 5), (4, 6),             # wheels
            (5, 6),
        ]
    },
    'forklift': {
        'num_keypoints': 8,
        'keypoint_names': [
            'fork_tip', 'fork_center', 'mast_top',
            'cab_front', 'cab_rear', 'front_wheel', 'rear_wheel',
            'counterweight_center'
        ],
        'edges': [
            (0, 1), (1, 2), (2, 3),     # fork -> mast -> cab
            (3, 4), (4, 7),              # cab -> counterweight
            (3, 5), (4, 6),             # wheels
            (5, 6),
        ]
    },
    'truck': {
        'num_keypoints': 6,
        'keypoint_names': [
            'cab_front', 'cab_rear', 'cargo_front', 'cargo_rear',
            'front_wheel', 'rear_wheel'
        ],
        'edges': [
            (0, 1), (1, 2), (2, 3),     # cab -> cargo
            (0, 4), (3, 5),             # wheels
            (4, 5),
        ]
    },
    'aerial_work_platform': {
        'num_keypoints': 8,
        'keypoint_names': [
            'lift_top', 'lift_bottom',
            'platform_edge_0', 'platform_edge_1', 'platform_edge_2', 'platform_edge_3',
            'fence_left', 'fence_right'
        ],
        'edges': [
            (0, 1),                       # lifting mechanism
            (0, 2), (0, 3), (0, 4), (0, 5),  # platform top
            (2, 3), (3, 4), (4, 5), (5, 2),  # platform edges
            (2, 6), (5, 7),               # fence
            (6, 7),
        ]
    },
    'ladder': {
        'num_keypoints': 4,
        'keypoint_names': [
            'top', 'left_foot', 'right_foot', 'center_rung'
        ],
        'edges': [
            (0, 3), (3, 1), (3, 2), (1, 2),
        ]
    },
    'scaffold': {
        'num_keypoints': -1,  # Variable N
        'keypoint_names': ['pole_connection'],
        'edges': []  # Sequential connections between poles
    },
    'handcart': {
        'num_keypoints': 7,
        'keypoint_names': [
            'left_handle', 'right_handle',
            'tray_edge_0', 'tray_edge_1', 'tray_edge_2', 'tray_edge_3',
            'wheel_center'
        ],
        'edges': [
            (0, 2), (0, 3),              # left handle -> tray
            (1, 4), (1, 5),              # right handle -> tray
            (2, 3), (3, 4), (4, 5), (5, 2),  # tray edges
            (6, 3), (6, 4),              # wheel -> tray
        ]
    },
    'wheelbarrow': {
        'num_keypoints': 7,
        'keypoint_names': [
            'handle_center', 'plate_edge_0', 'plate_edge_1', 'plate_edge_2',
            'plate_edge_3', 'front_wheel', 'rear_wheel'
        ],
        'edges': [
            (0, 1), (0, 4),              # handle -> plate
            (1, 2), (2, 3), (3, 4), (4, 1),  # plate edges
            (5, 2), (5, 3),              # front wheel -> plate
            (6, 0),                      # rear wheel -> handle
        ]
    },
    'enclosure': {
        'num_keypoints': 6,
        'keypoint_names': [
            'top_rail_center', 'rear_rail_center',
            'edge_0', 'edge_1', 'edge_2', 'edge_3'
        ],
        'edges': [
            (0, 2), (0, 3),              # top rail -> edges
            (1, 4), (1, 5),              # rear rail -> edges
            (2, 3), (3, 4), (4, 5), (5, 2),  # edges
            (0, 1),                      # rails
        ]
    },
    'hammer': {
        'num_keypoints': 4,
        'keypoint_names': [
            'head_front', 'head_rear', 'head_handle_connection', 'handle_end'
        ],
        'edges': [
            (0, 2), (1, 2), (2, 3),
        ]
    },
    'pickaxe': {
        'num_keypoints': 4,
        'keypoint_names': [
            'tip_front', 'head_rear', 'head_handle_connection', 'handle_end'
        ],
        'edges': [
            (0, 2), (1, 2), (2, 3),
        ]
    },
    'shovel': {
        'num_keypoints': 5,
        'keypoint_names': [
            'tip', 'head_edge_0', 'head_edge_1',
            'head_handle_connection', 'handle_end'
        ],
        'edges': [
            (0, 3), (1, 3), (2, 3), (3, 4),
            (0, 1), (1, 2),
        ]
    },
    'saw': {
        'num_keypoints': 3,
        'keypoint_names': [
            'cutting_edge', 'body_center', 'handle_end'
        ],
        'edges': [
            (0, 1), (1, 2),
        ]
    },
    'drill': {
        'num_keypoints': 5,
        'keypoint_names': [
            'bit_tip', 'trigger_switch', 'body_front', 'body_rear', 'handle_end'
        ],
        'edges': [
            (0, 2), (1, 2), (2, 3), (3, 4), (1, 4),
        ]
    },
}

# Total: worker(13) + max one equipment entity per pair
# Common pairing: worker(13) + one of 15 equipment types
# Maximum total keypoints: 13 + 8 = 21 (e.g., worker + excavator)
# For fixed-size model, pad to 34 (13 worker + 21 equipment slots)


def build_adjacency_matrix(entity_type: str, num_keypoints: int = None) -> torch.Tensor:
    """
    Build normalized adjacency matrix (A_tilde) for a given entity type.
    A_tilde = A + I, then normalized as D_tilde^{-1/2} A_tilde D_tilde^{-1/2}

    Args:
        entity_type: One of the 16 entity categories
        num_keypoints: Override number of keypoints (for scaffold with variable N)
    Returns:
        adj: Normalized adjacency matrix (V, V)
    """
    if entity_type not in ENTITY_KEYPOINTS:
        raise ValueError(f"Unknown entity type: {entity_type}. "
                         f"Must be one of {list(ENTITY_KEYPOINTS.keys())}")

    info = ENTITY_KEYPOINTS[entity_type]
    V = num_keypoints if num_keypoints is not None else info['num_keypoints']
    if V == -1:
        V = num_keypoints or 2  # Default for scaffold

    # Build adjacency
    adj = torch.zeros(V, V)
    for (i, j) in info['edges']:
        if i < V and j < V:
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    # Add self-loops: A_tilde = A + I (Eq. 7)
    adj = adj + torch.eye(V)

    # Normalize: D_tilde^{-1/2} A_tilde D_tilde^{-1/2}
    degree = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
    adj = adj / degree.sqrt() / degree.sqrt().t()

    return adj


def build_combined_adjacency(worker_type: str = 'worker',
                             equipment_type: str = 'excavator',
                             cross_edges: List[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Build combined adjacency matrix for a worker-equipment pair.
    Creates a block-diagonal adjacency with optional cross-entity edges
    for interaction modeling.

    Args:
        worker_type: Worker entity type (default: 'worker', 13 kpts)
        equipment_type: Equipment entity type
        cross_edges: Optional list of (worker_kpt_idx, equip_kpt_idx) for cross edges
    Returns:
        adj: Combined normalized adjacency (V_w + V_e, V_w + V_e)
    """
    V_w = ENTITY_KEYPOINTS[worker_type]['num_keypoints']
    V_e = ENTITY_KEYPOINTS[equipment_type]['num_keypoints']
    if V_e == -1:
        V_e = 2  # scaffold default
    V_total = V_w + V_e

    adj = torch.zeros(V_total, V_total)

    # Worker edges
    for (i, j) in ENTITY_KEYPOINTS[worker_type]['edges']:
        if i < V_w and j < V_w:
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    # Equipment edges
    for (i, j) in ENTITY_KEYPOINTS[equipment_type]['edges']:
        ii, jj = i + V_w, j + V_w
        if ii < V_total and jj < V_total:
            adj[ii, jj] = 1.0
            adj[jj, ii] = 1.0

    # Cross-entity edges (interaction links)
    if cross_edges:
        for (wi, ei) in cross_edges:
            if wi < V_w and ei < V_e:
                adj[wi, ei + V_w] = 1.0
                adj[ei + V_w, wi] = 1.0

    # Add self-loops and normalize
    adj = adj + torch.eye(V_total)
    degree = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
    adj = adj / degree.sqrt() / degree.sqrt().t()

    return adj


def get_all_entity_types() -> list:
    """Return all 16 entity category names."""
    return list(ENTITY_KEYPOINTS.keys())


def get_keypoint_count(entity_type: str) -> int:
    """Return number of keypoints for an entity type."""
    return ENTITY_KEYPOINTS[entity_type]['num_keypoints']


if __name__ == '__main__':
    print("=== Skeleton Topology for 16 Construction Entity Categories (Table 1) ===\n")
    for name, info in ENTITY_KEYPOINTS.items():
        nk = info['num_keypoints']
        ne = len(info['edges'])
        print(f"  {name:25s}: {nk:3d} keypoints, {ne:3d} edges")

    # Test: worker adjacency
    adj_worker = build_adjacency_matrix('worker')
    print(f"\nWorker adjacency: {adj_worker.shape}")
    print(f"  Symmetric: {torch.allclose(adj_worker, adj_worker.T)}")

    # Test: combined worker-excavator adjacency
    adj_combined = build_combined_adjacency('worker', 'excavator')
    print(f"\nWorker+Excavator adjacency: {adj_combined.shape}")
    print(f"  Symmetric: {torch.allclose(adj_combined, adj_combined.T)}")

    # Test all entity types build without error
    for etype in ENTITY_KEYPOINTS:
        nk = ENTITY_KEYPOINTS[etype]['num_keypoints']
        if nk == -1:
            nk = 5  # scaffold test
        adj = build_adjacency_matrix(etype, num_keypoints=nk)
        assert adj.shape == (nk, nk), f"Failed for {etype}"

    print("\nAll 16 entity topologies verified successfully!")
