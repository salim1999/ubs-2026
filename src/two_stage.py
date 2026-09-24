"""Two-stage model: (1) will this client have ANY recurring merchant in the
next 90 days, (2) if so, which of the 7 categories.

Motivation (from src/analyze.py's segment breakdown): a single multinomial
model has to simultaneously decide "is this client changing at all" and
"which of 7 categories" using the same weight vector per class softmax.
The coefficient inspection showed a lot of cross-category noise (e.g.
n_occurrences_gym pushing toward a cloud prediction) - splitting the
decision should let stage 1 focus purely on saturation/propensity features
(n_active_categories, adoption_rate_per_year, financial capacity) while
stage 2 focuses purely on which-category features (recent_txns_*,
recency_days_*, mean_gap_days_*) without a 'none' option diluting its
weights.

Both stages are logistic regression, class-balanced, same imputation +
scaling as the single-stage baseline (src.model.build_logreg) so any
difference in results is attributable to the two-stage structure, not a
change in the base learner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.category_map import TARGET_CATEGORIES
from src.model import ALL_LABELS, LABEL_COL, RANDOM_SEED, load_xy, set_global_seed


def build_stage1() -> Pipeline:
    """Binary: will the client have any recurring merchant next (not 'none')?"""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED
                ),
            ),
        ]
    )


def build_stage2() -> Pipeline:
    """Multiclass over the 7 categories only (fit on non-'none' rows)."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED
                ),
            ),
        ]
    )


def fit_two_stage(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Pipeline, Pipeline]:
    y1_train = (y_train != "none").astype(int)
    stage1 = build_stage1()
    stage1.fit(X_train, y1_train)

    mask = y_train != "none"
    stage2 = build_stage2()
    stage2.fit(X_train[mask], y_train[mask])

    return stage1, stage2


def two_stage_predict(
    stage1: Pipeline, stage2: Pipeline, X: pd.DataFrame, threshold: float = 0.5
) -> np.ndarray:
    p_active = stage1.predict_proba(X)[:, 1]
    is_active = p_active >= threshold

    preds = np.full(len(X), "none", dtype=object)
    if is_active.any():
        preds[is_active] = stage2.predict(X[is_active])
    return preds


def sweep_thresholds(
    stage1: Pipeline, stage2: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series
) -> pd.DataFrame:
    rows = []
    for t in np.arange(0.2, 0.81, 0.05):
        preds = two_stage_predict(stage1, stage2, X_valid, threshold=t)
        macro_f1 = f1_score(y_valid, preds, average="macro", labels=ALL_LABELS, zero_division=0)
        rows.append({"threshold": round(t, 2), "macro_f1": macro_f1})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    set_global_seed()

    X_train, y_train = load_xy("data/processed/train_features.csv")
    X_valid, y_valid = load_xy("data/processed/valid_features.csv")

    stage1, stage2 = fit_two_stage(X_train, y_train)

    preds_default = two_stage_predict(stage1, stage2, X_valid, threshold=0.5)
    macro_f1_default = f1_score(
        y_valid, preds_default, average="macro", labels=ALL_LABELS, zero_division=0
    )
    print(f"=== Two-stage (threshold=0.5) === macro-F1: {macro_f1_default:.4f}")
    print(classification_report(y_valid, preds_default, labels=ALL_LABELS, zero_division=0))

    print("=== Threshold sweep (stage-1 'is active' probability cutoff) ===")
    sweep = sweep_thresholds(stage1, stage2, X_valid, y_valid)
    print(sweep.to_string(index=False))

    best_t = sweep.loc[sweep["macro_f1"].idxmax(), "threshold"]
    preds_best = two_stage_predict(stage1, stage2, X_valid, threshold=best_t)
    macro_f1_best = f1_score(y_valid, preds_best, average="macro", labels=ALL_LABELS, zero_division=0)
    print()
    print(f"=== Two-stage (best threshold={best_t}) === macro-F1: {macro_f1_best:.4f}")
    print(classification_report(y_valid, preds_best, labels=ALL_LABELS, zero_division=0))
