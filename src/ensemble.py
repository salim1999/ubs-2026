"""Soft-voting ensemble of logreg_l1 + hgb_tuned.

Motivation: src/failure_analysis.py found the two models agree on only
53.4% of valid predictions, with balanced complementary errors (15.1% of
clients only logreg_l1 gets right, 14.7% only hgb_tuned) - logreg_l1 is
notably better at continuation (0.602 vs 0.517 accuracy), hgb_tuned a bit
better at new-adoption and true-none. That little overlap is exactly the
condition under which averaging predicted probabilities tends to help:
neither model dominates, and their mistakes aren't the same mistakes.

Weight (alpha) is swept on valid_features.csv, same protocol used for
every other tuning step in this pipeline (C for logreg, HGB's
hyperparameters) - valid_labels.csv is explicitly for this per the README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.model import ALL_LABELS, build_hgb_tuned, build_logreg_l1, load_xy, set_global_seed


def get_proba(model, X: pd.DataFrame) -> pd.DataFrame:
    """predict_proba as a DataFrame with columns in ALL_LABELS order,
    regardless of the order sklearn assigned to model.classes_."""
    raw = model.predict_proba(X)
    df = pd.DataFrame(raw, columns=model.classes_, index=X.index)
    return df[ALL_LABELS]


def ensemble_predict(proba_l1: pd.DataFrame, proba_hgb: pd.DataFrame, alpha: float) -> np.ndarray:
    """alpha = weight on logreg_l1; (1 - alpha) on hgb_tuned."""
    combined = alpha * proba_l1 + (1 - alpha) * proba_hgb
    return combined.idxmax(axis=1).to_numpy()


def sweep_alpha(proba_l1: pd.DataFrame, proba_hgb: pd.DataFrame, y_valid: pd.Series) -> pd.DataFrame:
    rows = []
    for alpha in np.arange(0.0, 1.01, 0.1):
        preds = ensemble_predict(proba_l1, proba_hgb, alpha)
        macro_f1 = f1_score(y_valid, preds, average="macro", labels=ALL_LABELS, zero_division=0)
        rows.append({"alpha_logreg_l1": round(alpha, 1), "macro_f1": macro_f1})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    set_global_seed()

    X_train, y_train = load_xy("data/processed/train_features.csv")
    X_valid, y_valid = load_xy("data/processed/valid_features.csv")

    logreg = build_logreg_l1()
    logreg.fit(X_train, y_train)
    hgb = build_hgb_tuned()
    hgb.fit(X_train, y_train)

    proba_l1 = get_proba(logreg, X_valid)
    proba_hgb = get_proba(hgb, X_valid)

    print("=== Alpha sweep (0.0 = pure hgb_tuned, 1.0 = pure logreg_l1) ===")
    sweep = sweep_alpha(proba_l1, proba_hgb, y_valid)
    print(sweep.to_string(index=False))
    print()

    best_alpha = sweep.loc[sweep["macro_f1"].idxmax(), "alpha_logreg_l1"]
    best_preds = ensemble_predict(proba_l1, proba_hgb, best_alpha)
    best_f1 = f1_score(y_valid, best_preds, average="macro", labels=ALL_LABELS, zero_division=0)

    print(f"=== Best ensemble: alpha={best_alpha} (weight on logreg_l1), macro-F1={best_f1:.4f} ===")
    print(classification_report(y_valid, best_preds, labels=ALL_LABELS, zero_division=0))

    print("=== Confusion matrix: best ensemble ===")
    cm = confusion_matrix(y_valid, best_preds, labels=ALL_LABELS)
    print(pd.DataFrame(cm, index=ALL_LABELS, columns=ALL_LABELS))

    print()
    print("=== For reference: individual models ===")
    for name, model in [("logreg_l1", logreg), ("hgb_tuned", hgb)]:
        preds = model.predict(X_valid)
        f1 = f1_score(y_valid, preds, average="macro", labels=ALL_LABELS, zero_division=0)
        print(f"  {name:10s} macro_f1={f1:.4f}")
