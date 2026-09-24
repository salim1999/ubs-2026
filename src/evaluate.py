"""One-command experiment loop: rebuild features, score, log (ported from timmyo).

    python -m src.evaluate "short description of the change"

Rebuilds train/valid/test features from the raw jsonl (so any change in
category_map / recurrence / features / personas is picked up; persona
scorer fit on train only), then scores every model two ways:

  valid  fit on train, score the 1000 valid clients (continuity with the
         other branches' numbers).
  vcv    5-fold StratifiedKFold over the *valid* clients: each fold fits on
         train + the other 4/5 of valid and predicts the held-out fifth; one
         macro-F1 over all 1000 pooled predictions. This is the decision
         metric. Pooling train+valid in plain k-fold CV was optimistic:
         valid clients are noisier than train and look more like test
         (median stream gap_cv 0.11 train, 0.29 valid, 0.33 test), so train
         clients must not dominate the scored folds.

Step-C diagnostics on valid judge stream detection on its own:
  c_detect      share of non-'none' clients whose target family has a
                recurring stream (live or lapsed)
  c_nextdue     share of non-'none' clients whose soonest-due *live* stream
                (same live test as src.features) is the target family
  c_none_active share of 'none' clients with >=1 recurring stream
  c_fams        mean number of recurring families per client

Every run appends one row to experiments.csv, with the git short SHA
("-dirty" = uncommitted changes on top of that commit).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from src.features import CUTOFF, build_all, recurring_streams
from src.model import ALL_LABELS, LABEL_COL, MODELS

LOG_PATH = "experiments.csv"
N_FOLDS = 5
VCV_SEED = 0


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", labels=ALL_LABELS, zero_division=0)


def split_xy(feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return feats.drop(columns=["client_id", LABEL_COL]), feats[LABEL_COL]


def vcv_predictions(build, X_train, y_train, X_valid, y_valid) -> np.ndarray:
    """Pooled out-of-fold predictions for the valid clients (see module docstring)."""
    pred = np.empty(len(X_valid), dtype=object)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=VCV_SEED)
    for tr, te in skf.split(X_valid, y_valid):
        X_fit = pd.concat([X_train, X_valid.iloc[tr]], ignore_index=True)
        y_fit = pd.concat([y_train, y_valid.iloc[tr]], ignore_index=True)
        pred[te] = build().fit(X_fit, y_fit).predict(X_valid.iloc[te])
    return pred


def stream_diagnostics(streams: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    """labels: target family indexed by client_id."""
    rec = recurring_streams(streams, pd.Timestamp(CUTOFF, tz="UTC"))
    families = rec.groupby("client_id")["category"].apply(set)
    live = rec[rec["is_live"]]
    first_due = live.sort_values("days_to_next").groupby("client_id")["category"].first()

    targets = labels[labels != "none"]
    nones = labels[labels == "none"]
    return {
        "c_detect": np.mean([t in families.get(c, set()) for c, t in targets.items()]),
        "c_nextdue": np.mean([first_due.get(c) == t for c, t in targets.items()]),
        "c_none_active": np.mean([len(families.get(c, set())) > 0 for c in nones.index]),
        "c_fams": np.mean([len(families.get(c, set())) for c in labels.index]),
    }


def git_sha() -> str:
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=no", "--", "src")
    return git("rev-parse", "--short", "HEAD") + ("-dirty" if dirty else "")


def log_row(note: str, valid: dict, vcv: dict, diag: dict) -> None:
    """Models not run this time (--models) get blank cells, so columns stay aligned."""
    header = ["time", "sha", "note", *(f"valid_{k}" for k in MODELS), *(f"vcv_{k}" for k in MODELS), *diag]
    row = [dt.datetime.now().isoformat(timespec="seconds"), git_sha(), note,
           *(f"{valid[k]:.4f}" if k in valid else "" for k in MODELS),
           *(f"{vcv[k]:.4f}" if k in vcv else "" for k in MODELS),
           *(f"{v:.3f}" for v in diag.values())]
    new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("note", nargs="?", default="")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    data = build_all()
    X_train, y_train = split_xy(data["train"])
    X_valid, y_valid = split_xy(data["valid"])
    X_valid = X_valid[X_train.columns]
    all_nan = [c for c in X_train.columns if X_train[c].isna().all() or X_valid[c].isna().all()]
    assert not all_nan, f"all-NaN feature columns: {all_nan}"
    print(f"{X_train.shape[1]} features; {len(X_train)} train, {len(X_valid)} valid clients")

    valid, vcv, vcv_preds = {}, {}, {}
    for name in args.models.split(","):
        build = MODELS[name]
        valid[name] = macro_f1(y_valid, build().fit(X_train, y_train).predict(X_valid))
        vcv_preds[name] = vcv_predictions(build, X_train, y_train, X_valid, y_valid)
        vcv[name] = macro_f1(y_valid, vcv_preds[name])
        print(f"  {name:7s} valid {valid[name]:.4f}   vcv {vcv[name]:.4f}")

    diag = stream_diagnostics(data["valid_streams"], data["valid"].set_index("client_id")[LABEL_COL])
    print("step C diagnostics (valid):")
    for k, v in diag.items():
        print(f"  {k:14s} {v:.3f}")

    best = max(vcv, key=vcv.get)
    per_class = f1_score(y_valid, vcv_preds[best], average=None, labels=ALL_LABELS, zero_division=0)
    print(f"\nper-class F1, {best} (vcv):")
    for lab, s in zip(ALL_LABELS, per_class):
        print(f"  {lab:10s} {s:.3f}")
    print("\nconfusion (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(y_valid, vcv_preds[best], labels=ALL_LABELS),
                       index=ALL_LABELS, columns=[c[:5] for c in ALL_LABELS]))

    if not args.no_log:
        log_row(args.note, valid, vcv, diag)
        print(f"\nlogged to {LOG_PATH}")


if __name__ == "__main__":
    main()
