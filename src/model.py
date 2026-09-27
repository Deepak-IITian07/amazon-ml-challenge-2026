"""
Entity Resolution Matching Model and F_0.5 Calibration Engine
Uses LightGBM with custom precision-weighted calibration.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import lightgbm as lgb

from src.metrics import evaluate_macro_f05


class ERMatchingModel:
    """
    Pairwise Classifier and Calibrated Decision Engine for Entity Resolution.
    """

    def __init__(self, optimal_threshold: float = 0.45):
        self.optimal_threshold = optimal_threshold
        self.model: Optional[lgb.Booster] = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[dict] = None,
        num_boost_round: int = 150
    ):
        """Trains LightGBM model on extracted pairwise feature vectors."""
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)

        default_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.08,
            "num_leaves": 31,
            "feature_fraction": 0.85,
            "verbose": -1,
            "seed": 42
        }
        if params:
            default_params.update(params)

        self.model = lgb.train(
            default_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts positive match probability for feature matrix X."""
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        if len(X) == 0:
            return np.array([])
        return self.model.predict(X)

    def optimize_threshold(
        self,
        val_pairs: List[Tuple[str, str]],
        val_probs: np.ndarray,
        val_s1_ids: List[str],
        ground_truth: Dict[str, Set[str]]
    ) -> float:
        """
        Grid searches over decision threshold to directly maximize Macro F_0.5.
        """
        best_thresh = 0.45
        best_f05 = 0.0
        val_gt = {s1_id: ground_truth.get(s1_id, set()) for s1_id in val_s1_ids}

        for thresh in np.arange(0.35, 0.85, 0.05):
            entity_preds = defaultdict(set)
            for (s1_id, cid), prob in zip(val_pairs, val_probs):
                if prob >= thresh:
                    entity_preds[s1_id].add(cid)

            eval_res = evaluate_macro_f05(entity_preds, val_gt)
            f05 = eval_res["macro_f05"]
            if f05 > best_f05:
                best_f05 = f05
                best_thresh = float(thresh)

        self.optimal_threshold = best_thresh
        return best_thresh

    def save(self, filepath: str):
        """Saves model to disk."""
        if self.model:
            self.model.save_model(filepath)

    def load(self, filepath: str):
        """Loads model from disk."""
        self.model = lgb.Booster(model_file=filepath)
