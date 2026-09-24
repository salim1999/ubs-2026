"""Trains and compares baseline models for target_next_recurring_merchant.

All models are fit on train_features.csv and scored on valid_features.csv
using macro-F1, matching the competition metric exactly (README.md ->
Task Specification). Four models, in increasing sophistication:

  1. majority  - always predict the most common training label ('none').
                 The floor: any real model must beat this.
  2. rule      - hand-written heuristic using only the recurring-stream
                 features (no learning). Tests how much a trained model
                 actually adds over the engineered features alone.
  3. logreg    - multinomial logistic regression, class-balanced. Needs
                 imputation + scaling since it can't handle NaN/unscaled
                 inputs the way trees can.
  4. hgb       - HistGradientBoostingClassifier, class-balanced. Handles
                 NaN natively (important: NaN here means "no active
                 subscription in this category", a real, informative
                 state, not a value to impute away).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.category_map import TARGET_CATEGORIES

LABEL_COL = "target_next_recurring_merchant"
ALL_LABELS = TARGET_CATEGORIES + ["none"]


def load_xy(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = df[LABEL_COL]
    X = df.drop(columns=["client_id", LABEL_COL])
    return X, y


def majority_predict(y_train: pd.Series, n: int) -> np.ndarray:
    majority = y_train.value_counts().idxmax()
    return np.full(n, majority)


def rule_predict(
    X: pd.DataFrame,
    min_over_days: float | None = None,
    tie_margin: float | None = None,
) -> np.ndarray:
    """No learning: use the engineered features directly.

    'none' if no stream is live (all overdue = stopped) or the longest live
    stream has only 3-4 charges (a trial that ends); else the live family
    due soonest (largest over_days = closest to its next charge).

    min_over_days: extra confidence gate on top of is_live - tested and
    REJECTED (macro-F1 0.4844 -> monotonically worse at every threshold
    tried, down to 0.1163 at min_over_days=0). is_live only requires
    over_days <= 5 with no lower bound, so a distant-due stream can win by
    default when it's the only live one - a blanket gate does catch those
    false positives, but converts more correct non-'none' predictions into
    wrong 'none' guesses than it fixes, since macro-F1 weighs all 8 classes
    equally. Left in as an opt-in parameter (default None = off) rather
    than removed, so the rejected experiment stays runnable/documented.

    tie_margin: when >1 live candidate's over_days are within this many
    days of the top one, break the tie by stream reliability
    (n_occurrences desc, then gap_cv asc) instead of picking whichever is
    marginally closer to due. Targets close calls like C000006/C000042
    (validation examples where the wrong candidate won by a few days'
    margin) without the blanket-gate's false-positive-to-false-negative
    tradeoff. None = off (original argmax-only behavior).
    """
    preds = []
    for _, row in X.iterrows():
        live = [
            (cat, row[f"over_days_{cat}"])
            for cat in TARGET_CATEGORIES
            if row[f"is_live_{cat}"] == 1
        ]
        if not live or row["short_live_stream"] == 1:
            preds.append("none")
            continue
        live.sort(key=lambda t: -t[1])

        if tie_margin is not None and len(live) > 1:
            top_over = live[0][1]
            contenders = [c for c in live if top_over - c[1] <= tie_margin]
            if len(contenders) > 1:
                def reliability(item):
                    cat = item[0]
                    n_occ = row.get(f"n_occurrences_{cat}", 0)
                    gap_cv = row.get(f"gap_cv_{cat}", 1.0)
                    gap_cv = 1.0 if pd.isna(gap_cv) else gap_cv
                    return (n_occ, -gap_cv)

                winner_cat, winner_over = max(contenders, key=reliability)
            else:
                winner_cat, winner_over = live[0]
        else:
            winner_cat, winner_over = live[0]

        if min_over_days is not None and winner_over < min_over_days:
            preds.append("none")
            continue
        preds.append(winner_cat)
    return np.array(preds)


def build_logreg() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            ),
        ]
    )


def build_hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=0)


def evaluate(name: str, y_true: pd.Series, y_pred: np.ndarray) -> float:
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=ALL_LABELS, zero_division=0)
    print(f"=== {name} ===")
    print(f"macro-F1: {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, labels=ALL_LABELS, zero_division=0))
    print()
    return macro_f1


if __name__ == "__main__":
    X_train, y_train = load_xy("data/processed/train_features.csv")
    X_valid, y_valid = load_xy("data/processed/valid_features.csv")

    results = {}

    pred_majority = majority_predict(y_train, len(y_valid))
    results["majority"] = evaluate("Majority baseline", y_valid, pred_majority)

    pred_rule = rule_predict(X_valid)
    results["rule"] = evaluate("Rule-based (no learning)", y_valid, pred_rule)

    logreg = build_logreg()
    logreg.fit(X_train, y_train)
    pred_logreg = logreg.predict(X_valid)
    results["logreg"] = evaluate("Logistic Regression", y_valid, pred_logreg)

    hgb = build_hgb()
    hgb.fit(X_train, y_train)
    pred_hgb = hgb.predict(X_valid)
    results["hgb"] = evaluate("HistGradientBoosting", y_valid, pred_hgb)

    print("=== Summary (macro-F1) ===")
    for name, score in sorted(results.items(), key=lambda t: -t[1]):
        print(f"  {name:10s} {score:.4f}")

    print()
    print("=== Confusion matrix: best model ===")
    best_name = max(results, key=results.get)
    best_pred = {"majority": pred_majority, "rule": pred_rule, "logreg": pred_logreg, "hgb": pred_hgb}[
        best_name
    ]
    cm = confusion_matrix(y_valid, best_pred, labels=ALL_LABELS)
    cm_df = pd.DataFrame(cm, index=ALL_LABELS, columns=ALL_LABELS)
    print(cm_df)
