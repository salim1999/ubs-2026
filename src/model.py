"""Trains and compares baseline models for target_next_recurring_merchant.

All models are fit on train_features.csv and scored on valid_features.csv
using macro-F1, matching the competition metric exactly (README.md ->
Task Specification). Five models, in increasing sophistication:

  1. majority   - always predict the most common training label ('none').
                  The floor: any real model must beat this.
  2. rule       - hand-written heuristic using only the recurring-stream
                  features (no learning). Tests how much a trained model
                  actually adds over the engineered features alone.
  3. logreg_l2  - multinomial logistic regression, L2, tuned C=3.0.
  4. logreg_l1  - multinomial logistic regression, L1, tuned C=10.0.
                  Kept alongside l2 rather than picking one - see
                  src/l1_logreg.py for the sweep behind both C choices;
                  they're close enough (0.3963 vs 0.3982 macro-F1) that
                  either is a reasonable default depending on what's built
                  on top next.
  5. hgb        - HistGradientBoostingClassifier, class-balanced. Handles
                  NaN natively (important: NaN here means "no active
                  subscription in this category", a real, informative
                  state, not a value to impute away).
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.category_map import TARGET_CATEGORIES

LABEL_COL = "target_next_recurring_merchant"
ALL_LABELS = TARGET_CATEGORIES + ["none"]

# Single seed reused everywhere a model or the environment has a random
# component, so a full re-run reproduces identical predictions and metrics.
RANDOM_SEED = 42


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_xy(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = df[LABEL_COL]
    X = df.drop(columns=["client_id", LABEL_COL])
    return X, y


def majority_predict(y_train: pd.Series, n: int) -> np.ndarray:
    majority = y_train.value_counts().idxmax()
    return np.full(n, majority)


def rule_predict(X: pd.DataFrame) -> np.ndarray:
    """No learning: use the engineered features directly.

    Preference order per client: (a) a category with fresh early-adoption
    signal (recent transactions, not yet recurring) and no other active
    subscription competing for attention, ranked by most recent activity;
    else (b) the active category due for its next payment soonest
    (smallest recency_days, i.e. most recently charged - likely to recur
    again soon); else (c) 'none'.
    """
    preds = []
    for _, row in X.iterrows():
        recent_candidates = [
            (cat, row[f"recent_txns_{cat}"])
            for cat in TARGET_CATEGORIES
            if row[f"recent_txns_{cat}"] > 0 and row[f"active_{cat}"] == 0
        ]
        if recent_candidates:
            recent_candidates.sort(key=lambda t: -t[1])
            preds.append(recent_candidates[0][0])
            continue

        active_candidates = [
            (cat, row[f"recency_days_{cat}"])
            for cat in TARGET_CATEGORIES
            if row[f"active_{cat}"] == 1 and not pd.isna(row[f"recency_days_{cat}"])
        ]
        if active_candidates:
            active_candidates.sort(key=lambda t: t[1])
            preds.append(active_candidates[0][0])
            continue

        preds.append("none")
    return np.array(preds)


def build_logreg_l2(C: float = 3.0) -> Pipeline:
    """L2 (default sklearn penalty), tuned C. See src/l1_logreg.py for the
    sweep: the untuned default (C=1.0) macro-F1 was 0.3808; C=3.0 gets
    0.3963. Kept alongside build_logreg_l1 rather than picking one - the
    two are close enough (0.3963 vs 0.3982) that the "right" choice may
    depend on which model you're building on top of next."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C, max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED
                ),
            ),
        ]
    )


def build_logreg_l1(C: float = 10.0) -> Pipeline:
    """L1, tuned C - best score found in the sweep (0.3982), with most
    coefficients still nonzero (89%) so it isn't meaningfully sparser than
    L2 here; the gain came mostly from loosening regularization strength,
    not from L1 specifically. Needs solver='saga' (the default 'lbfgs'
    only supports L2) and more iterations to converge cleanly."""
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


def build_hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_SEED)


def build_hgb_tuned() -> HistGradientBoostingClassifier:
    """Best of a 40-config random search (src/hgb_tune.py): macro-F1 0.3962,
    vs 0.3514 on defaults - untuned HGB was never a fair comparison against
    tuned logreg. Now roughly tied with logreg_l2/logreg_l1 (0.3963/0.3982);
    the gap that looked like "trees don't suit this task" was mostly just
    missing tuning, not an architecture difference."""
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=RANDOM_SEED,
        learning_rate=0.02,
        max_leaf_nodes=63,
        min_samples_leaf=5,
        l2_regularization=2.0,
        max_iter=300,
        max_depth=5,
    )


def evaluate(name: str, y_true: pd.Series, y_pred: np.ndarray) -> float:
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=ALL_LABELS, zero_division=0)
    print(f"=== {name} ===")
    print(f"macro-F1: {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, labels=ALL_LABELS, zero_division=0))
    print()
    return macro_f1


if __name__ == "__main__":
    set_global_seed()

    X_train, y_train = load_xy("data/processed/train_features.csv")
    X_valid, y_valid = load_xy("data/processed/valid_features.csv")

    results = {}

    pred_majority = majority_predict(y_train, len(y_valid))
    results["majority"] = evaluate("Majority baseline", y_valid, pred_majority)

    pred_rule = rule_predict(X_valid)
    results["rule"] = evaluate("Rule-based (no learning)", y_valid, pred_rule)

    logreg_l2 = build_logreg_l2()
    logreg_l2.fit(X_train, y_train)
    pred_logreg_l2 = logreg_l2.predict(X_valid)
    results["logreg_l2"] = evaluate("Logistic Regression (L2, C=3.0)", y_valid, pred_logreg_l2)

    logreg_l1 = build_logreg_l1()
    logreg_l1.fit(X_train, y_train)
    pred_logreg_l1 = logreg_l1.predict(X_valid)
    results["logreg_l1"] = evaluate("Logistic Regression (L1, C=10.0)", y_valid, pred_logreg_l1)

    hgb_tuned = build_hgb_tuned()
    hgb_tuned.fit(X_train, y_train)
    pred_hgb_tuned = hgb_tuned.predict(X_valid)
    results["hgb_tuned"] = evaluate("HistGradientBoosting (tuned)", y_valid, pred_hgb_tuned)

    print("=== Summary (macro-F1) ===")
    for name, score in sorted(results.items(), key=lambda t: -t[1]):
        print(f"  {name:10s} {score:.4f}")

    print()
    print("=== Confusion matrix: best model ===")
    best_name = max(results, key=results.get)
    best_pred = {
        "majority": pred_majority,
        "rule": pred_rule,
        "logreg_l2": pred_logreg_l2,
        "logreg_l1": pred_logreg_l1,
        "hgb_tuned": pred_hgb_tuned,
    }[best_name]
    cm = confusion_matrix(y_valid, best_pred, labels=ALL_LABELS)
    cm_df = pd.DataFrame(cm, index=ALL_LABELS, columns=ALL_LABELS)
    print(cm_df)
