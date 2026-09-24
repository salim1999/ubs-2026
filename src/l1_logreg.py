"""L1-regularized logistic regression: sweep C to find the sparsity level
that gives the best macro-F1 on valid_features.csv.

Motivation: src/analyze.py's coefficient inspection found real
multicollinearity in the L2 (default) model - many cross-category
coefficients with non-trivial magnitude (e.g. n_occurrences_gym pushing
toward a cloud prediction), and every feature addition since the original
79-feature baseline has been flat-to-negative on this ~2000-row dataset.
L1 drives redundant/correlated coefficients to exactly zero instead of
spreading weight across them the way L2 does, which should let the model
naturally prune the columns that were hurting rather than requiring us to
hand-pick which features to keep.

Solver: 'saga' is required for L1 + multinomial (the default 'lbfgs'
solver only supports L2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model import ALL_LABELS, RANDOM_SEED, load_xy, set_global_seed

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def build_logreg_l1(C: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=C,
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def sparsity(model: Pipeline) -> tuple[int, int]:
    coef = model.named_steps["clf"].coef_
    nonzero = int(np.sum(np.abs(coef) > 1e-8))
    total = coef.size
    return nonzero, total


if __name__ == "__main__":
    set_global_seed()

    X_train, y_train = load_xy("data/processed/train_features.csv")
    X_valid, y_valid = load_xy("data/processed/valid_features.csv")

    print(f"Feature count: {X_train.shape[1]}, classes: {len(ALL_LABELS)}")
    print()

    rows = []
    best = None
    for C in C_GRID:
        model = build_logreg_l1(C)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        macro_f1 = f1_score(y_valid, preds, average="macro", labels=ALL_LABELS, zero_division=0)
        nonzero, total = sparsity(model)
        rows.append(
            {
                "C": C,
                "macro_f1": macro_f1,
                "nonzero_coefs": nonzero,
                "total_coefs": total,
                "pct_nonzero": nonzero / total,
            }
        )
        if best is None or macro_f1 > best[1]:
            best = (C, macro_f1, model, preds)

    sweep = pd.DataFrame(rows)
    print("=== C sweep (L1, class_weight=balanced) ===")
    print(sweep.to_string(index=False))
    print()

    best_C, best_f1, best_model, best_preds = best
    print(f"=== Best: C={best_C}, macro-F1={best_f1:.4f} ===")
    print(classification_report(y_valid, best_preds, labels=ALL_LABELS, zero_division=0))

    nonzero, total = sparsity(best_model)
    print(f"Nonzero coefficients: {nonzero}/{total} ({nonzero/total:.1%})")

    # Which features survived L1 pruning across ALL classes (i.e. every
    # class's coefficient for that feature got zeroed) vs which features
    # are load-bearing for at least one class.
    coef = best_model.named_steps["clf"].coef_
    feature_names = list(X_train.columns)
    fully_pruned = [feature_names[i] for i in range(len(feature_names)) if np.all(np.abs(coef[:, i]) < 1e-8)]
    print()
    print(f"Features pruned to zero for ALL classes ({len(fully_pruned)}/{len(feature_names)}):")
    print(fully_pruned)
