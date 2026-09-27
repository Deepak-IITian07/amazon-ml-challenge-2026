"""
Official Evaluation Metric: Macro-averaged F_0.5 Score for Entity Resolution
As specified in ML Challenge 2026.
"""

from typing import Dict, List, Set, Union
import numpy as np


def compute_entity_f05(pred_ids: Union[Set[str], List[str]], true_ids: Union[Set[str], List[str]]) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity:
    - If true matches is empty (singleton):
        returns 1.0 if pred is empty, 0.0 otherwise.
    - If true matches is non-empty:
        returns 0.0 if pred is empty.
        otherwise:
            precision = TP / len(pred)
            recall = TP / len(true)
            F_0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)
    """
    pred_set = set(pred_ids) if not isinstance(pred_ids, set) else pred_ids
    true_set = set(true_ids) if not isinstance(true_ids, set) else true_ids

    # Remove any empty string tokens
    pred_set = {x.strip() for x in pred_set if x and x.strip()}
    true_set = {x.strip() for x in true_set if x and x.strip()}

    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0

    if len(pred_set) == 0:
        return 0.0

    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    denominator = 0.25 * precision + recall

    if denominator <= 0:
        return 0.0

    return (1.25 * precision * recall) / denominator


def evaluate_macro_f05(
    predictions: Dict[str, Union[Set[str], List[str]]],
    ground_truth: Dict[str, Union[Set[str], List[str]]]
) -> Dict[str, float]:
    """
    Computes Macro F_0.5 across all entities in ground_truth.
    Returns:
        {
            'macro_f05': float,
            'singleton_accuracy': float,
            'non_singleton_f05': float,
            'avg_precision': float,
            'avg_recall': float,
            'total_entities': int,
            'singleton_count': int
        }
    """
    scores = []
    singleton_scores = []
    non_singleton_scores = []
    precisions = []
    recalls = []

    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, set())
        true_set = {x.strip() for x in true_matches if x and x.strip()}
        pred_set = {x.strip() for x in pred_matches if x and x.strip()}

        score = compute_entity_f05(pred_set, true_set)
        scores.append(score)

        if len(true_set) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)
            if len(pred_set) > 0:
                tp = len(pred_set & true_set)
                p = tp / len(pred_set)
                r = tp / len(true_set)
                precisions.append(p)
                recalls.append(r)
            else:
                precisions.append(0.0)
                recalls.append(0.0)

    return {
        "macro_f05": float(np.mean(scores)) if scores else 0.0,
        "singleton_accuracy": float(np.mean(singleton_scores)) if singleton_scores else 0.0,
        "non_singleton_f05": float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0,
        "avg_precision": float(np.mean(precisions)) if precisions else 0.0,
        "avg_recall": float(np.mean(recalls)) if recalls else 0.0,
        "total_entities": len(ground_truth),
        "singleton_count": len(singleton_scores),
    }


if __name__ == "__main__":
    # Test with official example from README:
    # S1-00001: pred = [S2-00047, S2-00193, S3-00812], true = [S2-00047, S3-00812]
    # Precision = 2/3, Recall = 1.0 -> F_0.5 = 0.714
    score = compute_entity_f05(
        {"S2-00047", "S2-00193", "S3-00812"},
        {"S2-00047", "S3-00812"}
    )
    print(f"Sample test F_0.5: {score:.3f} (expected: 0.714)")
    assert abs(score - 0.7142857) < 1e-4, "Test calculation mismatch!"
    print("Metrics self-test passed!")
