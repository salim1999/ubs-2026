"""Hyperparameter tuning for HistGradientBoostingClassifier.

Every prior comparison in src/model.py ran HGB on defaults while logistic
regression got its C tuned (src/l1_logreg.py) - not a fair fight. This
does the equivalent for HGB: a random search over the parameters that
matter most for a boosted-tree model on a small (2000-row), high-class-
count (8) dataset - tree complexity (max_leaf_nodes, min_samples_leaf),
learning rate/iteration count trade-off, and L2 regularization.

Same protocol as the L1 C sweep: evaluated directly on valid_features.csv
via macro-F1, which is exactly what valid_labels.csv is for per the
README ("local validation and model selection").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score

from src.model import ALL_LABELS, RANDOM_SEED, load_xy, set_global_seed

N_ITER = 40

PARAM_DISTRIBUTIONS = {
    "learning_rate": [0.02, 0.05, 0.1, 0.2, 0.3],
    "max_leaf_nodes": [7, 15, 31, 63],
    "min_samples_leaf": [5, 10, 20, 40, 60],
    "l2_regularization": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
    "max_iter": [100, 200, 300],
    "max_depth": [None, 3, 5, 7],
}


def sample_params(rng: np.random.RandomState) -> dict:
    return {k: v[rng.randint(len(v))] for k, v in PARAM_DISTRIBUTIONS.items()}


def build_hgb_tuned(**params) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        class_weight="balanced", random_state=RANDOM_SEED, **params
    )


if __name__ == "__main__":
    set_global_seed()

    X_train, y_train = load_xy("data/processed/train_features.csv")
    X_valid, y_valid = load_xy("data/processed/valid_features.csv")

    rng = np.random.RandomState(RANDOM_SEED)
    seen = set()
    rows = []
    best = None

    while len(rows) < N_ITER:
        params = sample_params(rng)
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)

        model = build_hgb_tuned(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        macro_f1 = f1_score(y_valid, preds, average="macro", labels=ALL_LABELS, zero_division=0)

        rows.append({**params, "macro_f1": macro_f1})
        if best is None or macro_f1 > best[0]:
            best = (macro_f1, params, model, preds)

    sweep = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print(f"=== Random search: {N_ITER} configs ===")
    print(sweep.head(15).to_string(index=False))
    print()

    best_f1, best_params, best_model, best_preds = best
    print(f"=== Best HGB config: macro-F1={best_f1:.4f} ===")
    print(best_params)
    print()
    print(classification_report(y_valid, best_preds, labels=ALL_LABELS, zero_division=0))

    print("=== Default HGB (for reference) ===")
    default_model = build_hgb_tuned()
    default_model.fit(X_train, y_train)
    default_preds = default_model.predict(X_valid)
    default_f1 = f1_score(y_valid, default_preds, average="macro", labels=ALL_LABELS, zero_division=0)
    print(f"macro-F1: {default_f1:.4f}")
