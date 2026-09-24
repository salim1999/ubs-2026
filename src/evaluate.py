"""Cross-validated evaluation harness for model selection.

A single 1000-client valid split is noisy (+-~0.02 macro-F1), so every
change is scored with 5-fold stratified CV over train + valid (3000
labelled clients) with a fixed seed, alongside the plain train->valid
score for continuity with src.model.

Usage (after `python -m src.features`):
    python -m src.evaluate [--no-personas]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.model import ALL_LABELS, RankingModel, build_hgb, load_xy

N_FOLDS = 5
SEED = 0


def load_all(processed_dir: str = "data/processed") -> tuple[pd.DataFrame, pd.Series, int]:
    """Train + valid stacked; returns (X, y, n_train) so the original split is recoverable."""
    X_train, y_train = load_xy(f"{processed_dir}/train_features.csv")
    X_valid, y_valid = load_xy(f"{processed_dir}/valid_features.csv")
    X = pd.concat([X_train, X_valid[X_train.columns]], ignore_index=True)
    y = pd.concat([y_train, y_valid], ignore_index=True)
    return X, y, len(X_train)


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", labels=ALL_LABELS, zero_division=0)


def cv_oof_proba(build_model, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Out-of-fold class probabilities, columns ordered as ALL_LABELS."""
    oof = np.zeros((len(X), len(ALL_LABELS)))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, y):
        model = build_model()
        model.fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[te])
        cols = [list(model.classes_).index(c) for c in ALL_LABELS]
        oof[te] = proba[:, cols]
    return oof


def tune_class_weights(proba: np.ndarray, y, n_rounds: int = 3) -> np.ndarray:
    """Per-class multipliers w maximising macro-F1 of argmax(w * proba).

    Coordinate ascent over a log-spaced grid. Plain argmax optimises
    accuracy; macro-F1 rewards trading some 'none' precision for recall on
    the small classes, which these weights learn from OOF predictions.
    """
    labels = np.array(ALL_LABELS)
    w = np.ones(len(labels))
    grid = np.exp(np.linspace(np.log(0.3), np.log(3.0), 41))
    best = macro_f1(y, labels[(proba * w).argmax(1)])
    for _ in range(n_rounds):
        for k in range(len(labels)):
            for g in grid:
                trial = w.copy()
                trial[k] = g
                score = macro_f1(y, labels[(proba * trial).argmax(1)])
                if score > best + 1e-9:
                    best, w = score, trial
    return w


def nested_weight_score(proba: np.ndarray, y: pd.Series) -> float:
    """Honest estimate of the tuned-weights gain: fit weights on 4/5 of the
    OOF rows, score on the held-out 1/5."""
    labels = np.array(ALL_LABELS)
    pred = np.empty(len(y), dtype=object)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 1)
    for tr, te in skf.split(proba, y):
        w = tune_class_weights(proba[tr], y.iloc[tr])
        pred[te] = labels[(proba[te] * w).argmax(1)]
    return report("  + tuned weights (nested)", y, pred)


def report(name: str, y: pd.Series, pred: np.ndarray) -> float:
    score = macro_f1(y, pred)
    per_class = f1_score(y, pred, average=None, labels=ALL_LABELS, zero_division=0)
    cls = "  ".join(f"{c[:5]}={s:.2f}" for c, s in zip(ALL_LABELS, per_class))
    print(f"{name:28s} macro-F1 {score:.4f} | {cls}")
    return score


MODELS = {"hgb": build_hgb, "rank": RankingModel}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-personas", action="store_true")
    parser.add_argument("--models", default="hgb,rank")
    args = parser.parse_args()

    X, y, n_train = load_all()
    if args.no_personas:
        X = X.drop(columns=[c for c in X.columns if c.startswith("persona_")])
    print(f"{X.shape[1]} features, {len(X)} labelled clients")

    labels = np.array(ALL_LABELS)
    oofs = {}
    for name in args.models.split(","):
        oofs[name] = cv_oof_proba(MODELS[name], X, y)
        report(f"{name} 5-fold CV", y, labels[oofs[name].argmax(1)])
        nested_weight_score(oofs[name], y)
        model = MODELS[name]().fit(X.iloc[:n_train], y.iloc[:n_train])
        report(f"{name} train->valid", y.iloc[n_train:], model.predict(X.iloc[n_train:]))

    if len(oofs) > 1:
        blend = np.mean(list(oofs.values()), axis=0)
        report("blend 5-fold CV", y, labels[blend.argmax(1)])
        nested_weight_score(blend, y)
