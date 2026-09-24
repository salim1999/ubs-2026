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
SEED = 42
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
    """No learning: live-aware rule (ported from timmyo).

    'none' if no stream is live (all overdue by more than
    features.LIVE_MAX_OVER_DAYS = stopped) or the longest live stream has
    only 3-4 charges (a trial that ends); else the live family due soonest
    (smallest days_to_next). On timmyo's features this rule scored
    0.472-0.487 valid, above every trained model there.
    """
    preds = []
    for _, row in X.iterrows():
        live = [
            (cat, row[f"days_to_next_{cat}"])
            for cat in TARGET_CATEGORIES
            if row[f"active_{cat}"] == 1 and not pd.isna(row[f"days_to_next_{cat}"])
        ]
        if not live or row["short_live_stream"] == 1:
            preds.append("none")
            continue
        live.sort(key=lambda t: t[1])
        preds.append(live[0][0])
    return np.array(preds)


class RuleModel:
    """rule_predict behind the fit/predict interface; nothing to learn."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return rule_predict(X)


def build_logreg() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
            ),
        ]
    )


def build_hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)


def _to_long(X: pd.DataFrame) -> pd.DataFrame:
    """Wide client table -> one row per (client, category) candidate.

    Per-category columns (suffix _<cat>) become shared columns, so one
    binary model learns "is this the stream that recurs next" across all
    families, alongside the client-level context columns and the category id.
    """
    suffixes = tuple(f"_{c}" for c in TARGET_CATEGORIES)
    per_cat = [c for c in X.columns if c.endswith(suffixes)]
    client_cols = [c for c in X.columns if c not in per_cat]
    frames = []
    for i, cat in enumerate(TARGET_CATEGORIES):
        cols = [c for c in per_cat if c.endswith(f"_{cat}")]
        part = X[cols].copy()
        part.columns = [c[: -len(cat) - 1] for c in cols]
        part = pd.concat([part, X[client_cols].add_prefix("client_")], axis=1)
        part["cat_id"] = i
        part["row"] = np.arange(len(X))
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


class RankingModel:
    """Candidate ranker for the family + multiclass model for 'none'.

    p(cat) = (1 - p_none) * share of the ranker's score for cat;
    p(none) comes from the multiclass HGB, which sees all client context.
    """

    def __init__(self, seed: int = SEED):
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RankingModel":
        long = _to_long(X)
        target = (np.array(TARGET_CATEGORIES)[long["cat_id"]] == y.to_numpy()[long["row"]]).astype(int)
        feats = long.drop(columns=["row"])
        self.ranker_ = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=15, l2_regularization=1.0,
            categorical_features=(feats.columns == "cat_id"), random_state=self.seed,
        ).fit(feats, target)
        self.none_ = build_hgb().fit(X, y)
        self.classes_ = np.array(ALL_LABELS)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        long = _to_long(X)
        score = self.ranker_.predict_proba(long.drop(columns=["row"]))[:, 1]
        S = np.zeros((len(X), len(TARGET_CATEGORIES)))
        S[long["row"].to_numpy(), long["cat_id"].to_numpy()] = score
        share = S / S.sum(axis=1, keepdims=True).clip(min=1e-9)
        mc = self.none_.predict_proba(X)
        p_none = mc[:, list(self.none_.classes_).index("none")]
        return np.column_stack([(1 - p_none)[:, None] * share, p_none])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(1)]


class EnsembleModel:
    """Mean class probabilities of logreg, hgb and the ranking model."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleModel":
        self.members_ = [build().fit(X, y) for build in (build_logreg, build_hgb, RankingModel)]
        self.classes_ = np.array(ALL_LABELS)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = []
        for m in self.members_:
            p = m.predict_proba(X)
            probas.append(p[:, [list(m.classes_).index(c) for c in ALL_LABELS]])
        return np.mean(probas, axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(1)]


MODELS = {
    "rule": RuleModel,
    "logreg": build_logreg,
    "hgb": build_hgb,
    "rank": RankingModel,
    "ens": EnsembleModel,
}


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
