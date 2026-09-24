"""Deeper analysis of the best model (Logistic Regression) on valid_features.csv.

Trains on train_features.csv only (never touches valid labels during fit -
this is a pure held-out evaluation, distinct from the train+valid refit
used for the final test submission in generate_submission.py). Reuses the
fixed RANDOM_SEED from src.model for reproducibility.

Three angles beyond the plain classification report:
  1. Which features drive each class (LR coefficients) - sanity-checks
     that the model is using the engineered features the way we intended.
  2. Accuracy split by continuation vs. new-adoption vs. true-none - tests
     whether the model is actually better at one of these two very
     different sub-problems (see recurrence.py's write-up: ~53%
     continuation / 47% new-adoption among non-'none' labels).
  3. Most common confusion pairs - which categories get swapped for which.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.category_map import TARGET_CATEGORIES
from src.model import ALL_LABELS, LABEL_COL, build_logreg_l1, load_xy, set_global_seed

TOP_N_FEATURES = 8


def coefficient_report(model, feature_names: list[str]) -> pd.DataFrame:
    clf = model.named_steps["clf"]
    coefs = clf.coef_  # shape (n_classes, n_features)
    classes = clf.classes_

    rows = []
    for cls, row in zip(classes, coefs):
        order = np.argsort(row)
        top_pos = [(feature_names[i], row[i]) for i in order[::-1][:TOP_N_FEATURES]]
        top_neg = [(feature_names[i], row[i]) for i in order[:TOP_N_FEATURES]]
        rows.append({"class": cls, "top_positive": top_pos, "top_negative": top_neg})
    return pd.DataFrame(rows)


def print_coefficients(model, feature_names: list[str]) -> None:
    report = coefficient_report(model, feature_names)
    for _, row in report.iterrows():
        print(f"--- class = {row['class']} ---")
        print("  pushes TOWARD this class:")
        for name, w in row["top_positive"]:
            print(f"    {name:35s} {w:+.3f}")
        print("  pushes AWAY from this class:")
        for name, w in row["top_negative"]:
            print(f"    {name:35s} {w:+.3f}")
        print()


def continuation_vs_adoption_breakdown(X: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray) -> None:
    def segment(row_true, row_active_map) -> str:
        if row_true == "none":
            return "true_none"
        if row_active_map.get(row_true, 0) == 1:
            return "continuation"
        return "new_adoption"

    segments = []
    for i, true_label in enumerate(y_true):
        active_map = {cat: X.iloc[i][f"active_{cat}"] for cat in TARGET_CATEGORIES}
        segments.append(segment(true_label, active_map))
    segments = pd.Series(segments, index=y_true.index, name="segment")

    df = pd.DataFrame({"segment": segments, "true": y_true.values, "pred": y_pred})
    print("=== Accuracy by segment ===")
    for seg, g in df.groupby("segment"):
        acc = (g["true"] == g["pred"]).mean()
        print(f"  {seg:15s} n={len(g):4d}  accuracy={acc:.3f}")
    print()


def top_confusions(y_true: pd.Series, y_pred: np.ndarray, top_n: int = 10) -> None:
    df = pd.DataFrame({"true": y_true.values, "pred": y_pred})
    mistakes = df[df["true"] != df["pred"]]
    pair_counts = mistakes.groupby(["true", "pred"]).size().sort_values(ascending=False)
    print(f"=== Top {top_n} confusion pairs (true -> predicted) ===")
    for (true_lbl, pred_lbl), count in pair_counts.head(top_n).items():
        print(f"  {true_lbl:10s} -> {pred_lbl:10s}   {count}")
    print()


if __name__ == "__main__":
    set_global_seed()

    X_train, y_train = load_xy("data/processed/train_features.csv")
    X_valid, y_valid = load_xy("data/processed/valid_features.csv")

    model = build_logreg_l1()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    print_coefficients(model, list(X_train.columns))
    continuation_vs_adoption_breakdown(X_valid, y_valid, y_pred)
    top_confusions(y_valid, y_pred)
