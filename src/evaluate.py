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

from src.features import build_features, select_features
from src.model import ALL_LABELS, LABEL_COL, build_logreg, build_sparse_logreg, rule_predict
from src.pseudo import pseudo_labelled_features
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
    # --fast: only the lean models + final model and the rule (no full-set models, HGB or CV)
    fast = "--fast" in sys.argv

    train = labelled_features("train")
    valid = labelled_features("valid")
    y_train, y_valid = train[LABEL_COL], valid[LABEL_COL]
    X_train = train.drop(columns=["client_id", LABEL_COL])
    X_valid = valid.drop(columns=["client_id", LABEL_COL])[X_train.columns]

    # Benchmark models (always both), on both feature sets:
    #   logreg = global L2 LogReg, sparse_logreg = one sparse L1 LogReg per label;
    #   no suffix = "full" set (all features), "_lean" = fewer-features model.
    Xtr_lean, Xva_lean = select_features(X_train, "lean"), select_features(X_valid, "lean")
    # Final model: sparse per-label LogReg on lean, trained on train + pseudo-labelled
    # pretrain clients where all three label sources agree (src/pseudo.py).
    pseudo = pseudo_labelled_features()
    Xtr_final = pd.concat([Xtr_lean, pseudo[Xtr_lean.columns]], ignore_index=True)
    ytr_final = pd.concat([y_train, pseudo[LABEL_COL]], ignore_index=True)
    preds = {
        "final_sparse_lean_pseudo": build_sparse_logreg().fit(Xtr_final, ytr_final).predict(Xva_lean),
        "logreg_lean": build_logreg().fit(Xtr_lean, y_train).predict(Xva_lean),
        "sparse_logreg_lean": build_sparse_logreg().fit(Xtr_lean, y_train).predict(Xva_lean),
        "rule": rule_predict(X_valid),
    }
    if not fast:
        preds["logreg"] = build_logreg().fit(X_train, y_train).predict(X_valid)
        preds["sparse_logreg"] = build_sparse_logreg().fit(X_train, y_train).predict(X_valid)
        preds["hgb"] = fit_hgb(X_train, y_train).predict(X_valid)
    scores = {k: macro_f1(y_valid, p) for k, p in preds.items()}
    X_all = pd.concat([X_train, X_valid], ignore_index=True)
    y_all = pd.concat([y_train, y_valid], ignore_index=True)
    cv = {} if fast else {
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
    bench = [k for k in ("logreg", "sparse_logreg", "logreg_lean", "sparse_logreg_lean", "final_sparse_lean_pseudo")
             if k in preds]
    print(f"\nper-class F1 (benchmark models; full = {X_train.shape[1]} features, lean = {Xtr_lean.shape[1]}, "
          f"final = lean + {len(pseudo)} pseudo rows):")
    print(f"  {'':10s} " + " ".join(f"{k[:8]:>8s}" for k in bench))
    per_class = {k: f1_score(y_valid, preds[k], average=None, labels=ALL_LABELS, zero_division=0) for k in bench}
    for i, lab in enumerate(ALL_LABELS):
        print(f"  {lab:10s} " + " ".join(f"{per_class[k][i]:8.3f}" for k in bench))
    print("\nconfusion (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(y_valid, preds[best], labels=ALL_LABELS),
                       index=ALL_LABELS, columns=ALL_LABELS))

    header = ["time", "note", *scores.keys(), *cv.keys(), *diag.keys()]
    row = {"time": dt.datetime.now().isoformat(timespec="seconds"), "note": note,
           **{k: f"{v:.4f}" for k, v in scores.items()},
           **{k: f"{m:.4f}" for k, (m, _) in cv.items()},
           **{k: f"{v:.3f}" for k, v in diag.items()}}
    old = pd.read_csv(LOG_PATH, dtype=str) if os.path.exists(LOG_PATH) else pd.DataFrame(columns=header)
    # New columns (e.g. sparse_logreg) are appended; older rows stay empty there.
    cols = list(old.columns) + [c for c in header if c not in old.columns]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True)[cols].to_csv(LOG_PATH, index=False)

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
