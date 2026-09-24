"""One-command experiment loop: rebuild features, score on valid, log result.

    python -m src.evaluate "short description of the change"
    python -m src.evaluate "..." --submit    # also write data/submission.csv

Rebuilds train/valid features from the raw jsonl (so changes in
category_map / recurrence / features are picked up), joins labels, scores
the rule and the classifiers with macro-F1 on valid, and appends one line
per run to experiments.csv so every tiny change can be compared.

Also prints step-C diagnostics on valid, so recurrence detection can be
judged on its own, independent of the classifier:
  c_detect      share of non-'none' clients whose target family is active
  c_nextdue     share of non-'none' clients where the active stream due
                soonest (last + mean gap, lapsed streams skipped) is the target
  c_none_active share of 'none' clients with >=1 active stream
  c_fams        mean number of active families per client
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.features import build_features
from src.model import ALL_LABELS, LABEL_COL, build_logreg, rule_predict
from src.recurrence import detect_streams, load_transactions
from sklearn.ensemble import HistGradientBoostingClassifier

CUTOFF = "2026-01-01"
LOG_PATH = "experiments.csv"


def labelled_features(split: str) -> pd.DataFrame:
    feats = build_features(f"data/{split}_transactions.jsonl", CUTOFF)
    labels = pd.read_csv(f"data/{split}_labels.csv")[["client_id", LABEL_COL]]
    return labels.merge(feats, on="client_id", how="left")


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", labels=ALL_LABELS, zero_division=0)


def fit_hgb(X: pd.DataFrame, y: pd.Series) -> HistGradientBoostingClassifier:
    # sklearn < 1.2 has no class_weight on HGB; balanced sample weights are equivalent.
    m = HistGradientBoostingClassifier(random_state=0)
    m.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return m


def cv_macro_f1(X: pd.DataFrame, y: pd.Series, fit, k: int = 5) -> tuple[float, float]:
    """k-fold macro-F1 on train+valid: less noisy than the single valid split."""
    scores = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=0).split(X, y):
        m = fit(X.iloc[tr], y.iloc[tr])
        scores.append(macro_f1(y.iloc[te], m.predict(X.iloc[te])))
    return float(np.mean(scores)), float(np.std(scores))


def stream_diagnostics(split: str) -> dict[str, float]:
    cutoff = pd.Timestamp(CUTOFF, tz="UTC")
    labels = pd.read_csv(f"data/{split}_labels.csv").set_index("client_id")[LABEL_COL]
    streams = detect_streams(load_transactions(f"data/{split}_transactions.jsonl"))
    active = streams[streams["is_recurring"] & streams["category"].notna()].copy()

    families = active.groupby("client_id")["category"].apply(set)
    active["due"] = active["mean_gap_days"] - (cutoff - active["last_date"]).dt.days
    live = active[active["due"] >= -active["mean_gap_days"]]
    first_due = live.sort_values("due").groupby("client_id")["category"].first()

    targets = labels[labels != "none"]
    nones = labels[labels == "none"]
    return {
        "c_detect": np.mean([t in families.get(c, set()) for c, t in targets.items()]),
        "c_nextdue": np.mean([first_due.get(c) == t for c, t in targets.items()]),
        "c_none_active": np.mean([len(families.get(c, set())) > 0 for c in nones.index]),
        "c_fams": np.mean([len(families.get(c, set())) for c in labels.index]),
    }


def main() -> None:
    note = sys.argv[1] if len(sys.argv) > 1 else ""
    submit = "--submit" in sys.argv

    train = labelled_features("train")
    valid = labelled_features("valid")
    y_train, y_valid = train[LABEL_COL], valid[LABEL_COL]
    X_train = train.drop(columns=["client_id", LABEL_COL])
    X_valid = valid.drop(columns=["client_id", LABEL_COL])[X_train.columns]

    logreg = build_logreg().fit(X_train, y_train)
    preds = {
        "logreg": logreg.predict(X_valid),
        "hgb": fit_hgb(X_train, y_train).predict(X_valid),
        "rule": rule_predict(X_valid),
    }
    scores = {k: macro_f1(y_valid, p) for k, p in preds.items()}
    X_all = pd.concat([X_train, X_valid], ignore_index=True)
    y_all = pd.concat([y_train, y_valid], ignore_index=True)
    cv = {
        "cv_logreg": cv_macro_f1(X_all, y_all, lambda X, y: build_logreg().fit(X, y)),
        "cv_hgb": cv_macro_f1(X_all, y_all, fit_hgb),
    }
    diag = stream_diagnostics("valid")
    best = max(scores, key=scores.get)

    print(f"\n{note}")
    for k, s in scores.items():
        print(f"  {k:8s} macro-F1 = {s:.4f}")
    for k, (mean, std) in cv.items():
        print(f"  {k:8s} 5-fold    = {mean:.4f} +- {std:.4f}")
    print("step C diagnostics (valid):")
    for k, v in diag.items():
        print(f"  {k:14s} {v:.3f}")
    per_class = f1_score(y_valid, preds[best], average=None, labels=ALL_LABELS, zero_division=0)
    print(f"\nper-class F1 ({best}):")
    for lab, s in zip(ALL_LABELS, per_class):
        print(f"  {lab:10s} {s:.3f}")
    print("\nconfusion (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(y_valid, preds[best], labels=ALL_LABELS),
                       index=ALL_LABELS, columns=ALL_LABELS))

    new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "note", *scores.keys(), *cv.keys(), *diag.keys()])
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), note,
                    *(f"{s:.4f}" for s in scores.values()),
                    *(f"{m:.4f}" for m, _ in cv.values()),
                    *(f"{v:.3f}" for v in diag.values())])

    if submit:
        test = build_features("data/test_transactions.jsonl", CUTOFF)
        sample = pd.read_csv("data/sample_submission.csv")[["client_id"]]
        test = sample.merge(test, on="client_id", how="left")
        X_test = test.drop(columns=["client_id"]).reindex(columns=X_train.columns)
        if best == "rule":
            pred = rule_predict(X_test)
        else:
            X_full, y_full = pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid])
            m = build_logreg().fit(X_full, y_full) if best == "logreg" else fit_hgb(X_full, y_full)
            pred = m.predict(X_test)
        out = pd.DataFrame({"client_id": sample["client_id"],
                            "predicted_next_recurring_merchant": pred})
        assert out["predicted_next_recurring_merchant"].isin(ALL_LABELS).all()
        out.to_csv("data/submission.csv", index=False)
        print(f"\nwrote data/submission.csv using {best} (trained on train+valid)")


if __name__ == "__main__":
    main()
