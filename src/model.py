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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
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


def rule_predict(X: pd.DataFrame) -> np.ndarray:
    """No learning: use the engineered features directly.

    'none' if no stream is live (all overdue = stopped) or the longest live
    stream has only 3-4 charges (a trial that ends); else the live family
    due soonest (largest over_days = closest to its next charge).
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
        preds.append(live[0][0])
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


class SparsePerLabelLogReg(BaseEstimator, ClassifierMixin):
    """Benchmark model 2: one sparse yes/no logistic regression per label.

    For each label ("is it gym?", "is it music?", ..., "is it none?") an
    L1-penalised logistic regression keeps only the predictors that matter
    for that label. Each label picks its own penalty C by inner CV on the
    training data, so easy labels end up very sparse and messy ones (none)
    keep more. Prediction = label with the highest probability.

    Each coefficient reads as: effect of +1 standard deviation of the
    feature on the log-odds of *this* label vs all others. See coef_table().
    """

    def __init__(self, Cs=(0.03, 0.1, 0.3, 1.0), inner_folds=3):
        self.Cs = Cs
        self.inner_folds = inner_folds

    def _model(self, C):
        return LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                  class_weight="balanced", max_iter=2000, random_state=0)

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.array(ALL_LABELS)
        self.models_, self.C_ = {}, {}
        for label in self.classes_:
            yb = (y == label).astype(int)
            folds = StratifiedKFold(self.inner_folds, shuffle=True, random_state=0)
            cv = {
                C: np.mean([f1_score(yb[te], self._model(C).fit(X[tr], yb[tr]).predict(X[te]), zero_division=0)
                            for tr, te in folds.split(X, yb)])
                for C in self.Cs
            }
            self.C_[label] = max(cv, key=cv.get)
            self.models_[label] = self._model(self.C_[label]).fit(X, yb)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        return np.column_stack([self.models_[l].predict_proba(X)[:, 1] for l in self.classes_])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_sparse_logreg() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", SparsePerLabelLogReg()),
        ]
    )


def coef_table(pipe: Pipeline, feature_names, top: int = 5) -> pd.DataFrame:
    """Top predictors per label of a fitted build_sparse_logreg() pipeline.

    odds_ratio = multiplier on the odds of the label per +1 std of the feature.
    """
    clf = pipe.named_steps["clf"]
    rows = []
    for label in clf.classes_:
        coef = pd.Series(clf.models_[label].coef_[0], index=list(feature_names))
        nz = coef[coef != 0]
        for feat, c in nz.reindex(nz.abs().sort_values(ascending=False).index)[:top].items():
            rows.append({"label": label, "C": clf.C_[label], "n_predictors": len(nz),
                         "feature": feat, "coef": round(c, 3), "odds_ratio": round(float(np.exp(c)), 2)})
    return pd.DataFrame(rows)


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
