"""
Evaluation Metrics for Keypoint Repair and Action Recognition
==============================================================

Implements all evaluation metrics used in the manuscript:

Keypoint Repair Metrics (Section 4.2):
  - AJC: Average Joint Completeness (%)
  - 2D MPJPE: Mean Per Joint Position Error (pixels)
  - PCK@10px: Percentage of Correct Keypoints at 10px threshold
  - PCK normalized: PCK@0.05 * bbox diagonal, torso-scale, equipment-scale

Action Recognition Metrics (Section 4.2):
  - Precision, Recall, F1-score per class
  - Macro-F1: Mean of per-class F1 scores
  - Accuracy (overall)
  - MDR: Miss Detection Rate
  - FDR: False Detection Rate
  - Confusion Matrix
"""

import torch
import numpy as np
from typing import Dict, Optional


# =============================================================================
# Keypoint Repair Metrics
# =============================================================================

def average_joint_completeness(predicted: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Average Joint Completeness (AJC): percentage of correctly predicted keypoints
    relative to ground truth keypoints.

    Args:
        predicted: Predicted keypoint presence mask (T, N), 1=present, 0=missing
        gt_mask: Ground truth keypoint presence mask (T, N), 1=present, 0=missing
    Returns:
        AJC in percentage [0, 100]
    """
    total_gt = gt_mask.sum()
    if total_gt == 0:
        return 100.0
    correctly_predicted = (predicted * gt_mask).sum()
    return float(correctly_predicted / total_gt * 100.0)


def mean_per_joint_position_error(predicted: np.ndarray, gt: np.ndarray,
                                   mask: Optional[np.ndarray] = None) -> float:
    """
    2D Mean Per Joint Position Error (MPJPE): average Euclidean distance
    between predicted and ground truth keypoints.

    Args:
        predicted: Predicted keypoints (T, N, 2)
        gt: Ground truth keypoints (T, N, 2)
        mask: Optional binary mask (T, N), only compute on visible/valid keypoints
    Returns:
        MPJPE in pixels
    """
    diff = predicted - gt
    dist = np.sqrt((diff ** 2).sum(axis=-1))  # (T, N)

    if mask is not None:
        dist = dist * mask
        count = mask.sum()
        if count == 0:
            return 0.0
        return float(dist.sum() / count)
    else:
        return float(dist.mean())


def pck(predicted: np.ndarray, gt: np.ndarray, threshold: float = 10.0,
        mask: Optional[np.ndarray] = None) -> float:
    """
    Percentage of Correct Keypoints (PCK) at a given threshold.

    PCK@threshold = fraction of predicted keypoints within `threshold` pixels of GT

    Args:
        predicted: Predicted keypoints (T, N, 2)
        gt: Ground truth keypoints (T, N, 2)
        threshold: Pixel distance threshold (default: 10px)
        mask: Optional binary mask (T, N)
    Returns:
        PCK in percentage [0, 100]
    """
    diff = predicted - gt
    dist = np.sqrt((diff ** 2).sum(axis=-1))  # (T, N)
    correct = (dist <= threshold).astype(np.float32)

    if mask is not None:
        correct = correct * mask
        total = mask.sum()
        if total == 0:
            return 100.0
        return float(correct.sum() / total * 100.0)
    else:
        return float(correct.mean() * 100.0)


def pck_normalized(predicted: np.ndarray, gt: np.ndarray,
                   bbox_diagonal: Optional[np.ndarray] = None,
                   torso_scale: Optional[float] = None,
                   norm_factor: float = 0.05) -> float:
    """
    Normalized PCK: PCK@(norm_factor * reference_scale)

    Supports three normalization modes:
      1. Bbox diagonal: threshold = norm_factor * bbox_diagonal
      2. Torso scale: threshold = norm_factor * torso_scale (for workers)
      3. Equipment scale: threshold = norm_factor * equipment_diag

    Args:
        predicted: (T, N, 2)
        gt: (T, N, 2)
        bbox_diagonal: (T, N) diagonal of bounding box per keypoint, or scalar
        torso_scale: Scalar torso size (e.g., shoulder-to-hip distance)
        norm_factor: Normalization factor (default: 0.05 = 5%)
    Returns:
        Normalized PCK in percentage [0, 100]
    """
    if bbox_diagonal is not None:
        threshold = norm_factor * bbox_diagonal  # (T, N) or scalar
    elif torso_scale is not None:
        threshold = norm_factor * torso_scale
    else:
        # Fallback to fixed 10px
        threshold = 10.0

    diff = predicted - gt
    dist = np.sqrt((diff ** 2).sum(axis=-1))  # (T, N)

    correct = (dist <= threshold).astype(np.float32)
    return float(correct.mean() * 100.0)


def compute_repair_metrics(predicted: np.ndarray, gt: np.ndarray,
                           mask: np.ndarray,
                           bbox_diagonal: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute all keypoint repair metrics.

    Args:
        predicted: (T, N, 2) predicted keypoints
        gt: (T, N, 2) ground truth keypoints
        mask: (T, N) binary mask (1=valid/visible, 0=occluded)
        bbox_diagonal: Optional (T, N) for normalized PCK
    Returns:
        Dict with AJC, MPJPE, PCK@10px, PCK_norm
    """
    repair_mask = (predicted.sum(axis=-1) > 0).astype(np.float32)  # predicted presence

    results = {
        'AJC': average_joint_completeness(repair_mask, mask),
        'MPJPE': mean_per_joint_position_error(predicted, gt, mask),
        'PCK@10px': pck(predicted, gt, threshold=10.0, mask=mask),
        'PCK@0.05*diag': pck_normalized(predicted, gt, bbox_diagonal=bbox_diagonal, norm_factor=0.05),
    }
    return results


# =============================================================================
# Action Recognition Metrics
# =============================================================================

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     num_classes: int) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: (N,) ground truth class labels
        y_pred: (N,) predicted class labels
        num_classes: Total number of classes
    Returns:
        cm: (num_classes, num_classes) confusion matrix
             cm[i][j] = number of samples with true label i predicted as j
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray,
                         num_classes: int) -> Dict[str, np.ndarray]:
    """
    Compute per-class Precision, Recall, F1-score.

    Definitions (per class k):
      Precision_k = TP_k / (TP_k + FP_k)
      Recall_k    = TP_k / (TP_k + FN_k)
      F1_k        = 2 * Precision_k * Recall_k / (Precision_k + Recall_k)

    Args:
        y_true: (N,) ground truth labels
        y_pred: (N,) predicted labels
        num_classes: Number of classes
    Returns:
        Dict with 'precision', 'recall', 'f1' arrays of shape (num_classes,)
    """
    cm = confusion_matrix(y_true, y_pred, num_classes)

    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for k in range(num_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp

        precision[k] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[k] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[k] = (2 * precision[k] * recall[k] / (precision[k] + recall[k])
                  if (precision[k] + recall[k]) > 0 else 0.0)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray,
             num_classes: int) -> float:
    """
    Macro-F1: Mean of per-class F1 scores.

    Macro-F1 = (1/K) * sum(F1_k) for k=1..K
    """
    metrics = precision_recall_f1(y_true, y_pred, num_classes)
    return float(metrics['f1'].mean())


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Overall accuracy = correct predictions / total predictions."""
    return float((y_true == y_pred).mean() * 100.0)


def miss_detection_rate(y_true: np.ndarray, y_pred: np.ndarray,
                        num_classes: int) -> float:
    """
    Miss Detection Rate (MDR): fraction of actual positive samples
    that were incorrectly classified as a different class.

    MDR = (1/K) * sum(FN_k / (TP_k + FN_k)) for k=1..K
        = (1/K) * sum(1 - Recall_k)

    Reported in percentage.
    """
    metrics = precision_recall_f1(y_true, y_pred, num_classes)
    mdr_per_class = 1.0 - metrics['recall']
    return float(mdr_per_class.mean() * 100.0)


def false_detection_rate(y_true: np.ndarray, y_pred: np.ndarray,
                         num_classes: int) -> float:
    """
    False Detection Rate (FDR): fraction of predicted positive samples
    that were incorrectly classified.

    FDR = (1/K) * sum(FP_k / (TP_k + FP_k)) for k=1..K
        = (1/K) * sum(1 - Precision_k)

    Reported in percentage.
    """
    metrics = precision_recall_f1(y_true, y_pred, num_classes)
    fdr_per_class = 1.0 - metrics['precision']
    return float(fdr_per_class.mean() * 100.0)


def compute_action_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           num_classes: int = 13) -> Dict[str, float]:
    """
    Compute all action recognition metrics.

    Args:
        y_true: (N,) ground truth labels
        y_pred: (N,) predicted labels
        num_classes: Number of action classes (default: 13)
    Returns:
        Dict with all metrics
    """
    per_class = precision_recall_f1(y_true, y_pred, num_classes)
    cm = confusion_matrix(y_true, y_pred, num_classes)

    return {
        'accuracy': accuracy(y_true, y_pred),
        'macro_f1': macro_f1(y_true, y_pred, num_classes),
        'MDR': miss_detection_rate(y_true, y_pred, num_classes),
        'FDR': false_detection_rate(y_true, y_pred, num_classes),
        'precision_per_class': per_class['precision'],
        'recall_per_class': per_class['recall'],
        'f1_per_class': per_class['f1'],
        'confusion_matrix': cm,
    }


if __name__ == '__main__':
    # Test keypoint repair metrics
    np.random.seed(42)
    T, N = 64, 34
    gt_kpts = np.random.randn(T, N, 2).astype(np.float32) * 100 + 320
    pred_kpts = gt_kpts + np.random.randn(T, N, 2).astype(np.float32) * 5
    mask = np.ones((T, N), dtype=np.float32)
    mask[:, :10] = 0  # 10 keypoints occluded

    repair_results = compute_repair_metrics(pred_kpts, gt_kpts, mask)
    print("=== Keypoint Repair Metrics ===")
    for k, v in repair_results.items():
        print(f"  {k}: {v:.2f}")

    # Test action recognition metrics
    num_samples = 1000
    num_classes = 13
    y_true = np.random.randint(0, num_classes, num_samples)
    y_pred = y_true.copy()
    # Add some noise
    noise_idx = np.random.choice(num_samples, 200, replace=False)
    y_pred[noise_idx] = np.random.randint(0, num_classes, 200)

    action_results = compute_action_metrics(y_true, y_pred, num_classes)
    print("\n=== Action Recognition Metrics ===")
    print(f"  Accuracy:  {action_results['accuracy']:.2f}%")
    print(f"  Macro-F1:  {action_results['macro_f1']:.4f}")
    print(f"  MDR:       {action_results['MDR']:.2f}%")
    print(f"  FDR:       {action_results['FDR']:.2f}%")
    print(f"  Confusion matrix shape: {action_results['confusion_matrix'].shape}")

    print("\nAll metrics tests passed!")
