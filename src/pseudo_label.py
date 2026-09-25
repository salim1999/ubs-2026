"""Pseudo-label the 10,000 unlabeled pretrain clients with our own model.

`unlabeled_pretrain_transactions.jsonl` has 5x more client histories than
train, with the same pre-cutoff window and no labels. Labelling them with an
LLM (src.extend_training_data) was tried first: gpt-4.1-mini reached
macro-F1 0.39-0.47 on held-out valid clients, below model.rule_predict
(0.41-0.53) and well below our trained models (vcv ~0.55). So the teacher
here is our own model: fit on the labelled clients, predict the unlabeled
ones, keep the confident predictions as extra training rows (self-training).

Whether that helps is measured with the same folds as src.evaluate's vcv.
Per fold the teacher is fit on train + the other 4/5 of valid only, so the
held-out fifth's labels never reach the pseudo-labels, and a student is fit
on the same rows plus the pseudo-labels at or above each confidence
threshold. The "baseline" row is the teacher's own vcv score.

First run (rank, 2026-09-24): baseline vcv 0.5445; with pseudo-labels
0.5608 (all 10,000), 0.5670 (conf >= 0.5, ~7,300 rows), 0.5618 / 0.5634 /
0.5582 at 0.6 / 0.7 / 0.8. Every cutoff helps, and 7 of 8 classes improve at
0.5 (streaming -0.003). Choosing the cutoff on these same folds is
optimistic; the gain over the whole range is the robust part.

Outputs (client-level, same cutoff as train):
    data/labeled_pretrain_transactions.jsonl  label, confidence, class probabilities
    data/labeled_pretrain_labels.csv          train_labels.csv schema + confidence
Both come from a teacher fit on all 3,000 labelled clients, so they have
seen every valid label: fine for the submission (predict.py refits on
train+valid anyway), but any score on valid from a model trained on them is
optimistic. `--teacher-data train` fits the final teacher on the 2,000 train
clients only and writes to the *_trainonly paths instead; use those files
whenever valid is the scoring set.

Usage (rebuilds train/valid/test features first, like src.evaluate):
    python -m src.pseudo_label [--model rank] [--thresholds 0,0.5,0.6,0.7,0.8] [--skip-eval]
                               [--teacher-data train|train+valid]
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.evaluate import N_FOLDS, VCV_SEED, macro_f1, split_xy
from src.features import CUTOFF, DATA_DIR, PROCESSED_DIR, build_all, build_features
from src.model import ALL_LABELS, LABEL_COL, MODELS

UNLABELED_PATH = f"{DATA_DIR}/unlabeled_pretrain_transactions.jsonl"
UNLABELED_FEATURES_PATH = f"{PROCESSED_DIR}/unlabeled_features.csv"
OUTPUT_JSONL = "data/labeled_pretrain_transactions.jsonl"
OUTPUT_CSV = "data/labeled_pretrain_labels.csv"
# Valid-free variant: final teacher fit on train only (--teacher-data train).
OUTPUT_JSONL_TRAINONLY = "data/labeled_pretrain_transactions_trainonly.jsonl"
OUTPUT_CSV_TRAINONLY = "data/labeled_pretrain_labels_trainonly.csv"
DEFAULT_MODEL = "rank"
DEFAULT_THRESHOLDS = "0,0.5,0.6,0.7,0.8"
# The rule has no class probabilities, so it can't give a confidence.
PROBA_MODELS = [name for name in MODELS if name != "rule"]


def unlabeled_features() -> pd.DataFrame:
    feats = build_features(UNLABELED_PATH, CUTOFF)
    feats.attrs = {}
    feats.to_csv(UNLABELED_FEATURES_PATH, index=False)
    print(f"unlabeled: {feats.shape} -> {UNLABELED_FEATURES_PATH}")
    return feats


def teach(build, X_fit: pd.DataFrame, y_fit: pd.Series, X_unl: pd.DataFrame):
    """Fit the teacher; return it with class probabilities (ALL_LABELS order) for X_unl."""
    model = build().fit(X_fit, y_fit)
    proba = model.predict_proba(X_unl)
    order = [list(model.classes_).index(c) for c in ALL_LABELS]
    return model, proba[:, order]


def vcv_with_pseudo(build, X_train, y_train, X_valid, y_valid, X_unl, thresholds):
    """Pooled out-of-fold valid predictions: teacher alone and student per threshold."""
    preds = {key: np.empty(len(X_valid), dtype=object) for key in ["baseline", *thresholds]}
    n_used = {t: [] for t in thresholds}
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=VCV_SEED)
    for fold, (tr, te) in enumerate(skf.split(X_valid, y_valid), 1):
        X_fit = pd.concat([X_train, X_valid.iloc[tr]], ignore_index=True)
        y_fit = pd.concat([y_train, y_valid.iloc[tr]], ignore_index=True)
        teacher, proba = teach(build, X_fit, y_fit, X_unl)
        preds["baseline"][te] = teacher.predict(X_valid.iloc[te])
        pseudo = pd.Series(np.array(ALL_LABELS)[proba.argmax(1)])
        conf = proba.max(1)
        for t in thresholds:
            keep = conf >= t
            n_used[t].append(int(keep.sum()))
            student = build().fit(
                pd.concat([X_fit, X_unl[keep]], ignore_index=True),
                pd.concat([y_fit, pseudo[keep]], ignore_index=True),
            )
            preds[t][te] = student.predict(X_valid.iloc[te])
        print(f"  fold {fold}/{N_FOLDS} done")
    return preds, {t: int(np.mean(v)) for t, v in n_used.items()}


def report(preds: dict, n_used: dict, y_valid: pd.Series) -> None:
    rows = []
    for key, pred in preds.items():
        per_class = f1_score(y_valid, pred, average=None, labels=ALL_LABELS, zero_division=0)
        rows.append({
            "run": "baseline (no pseudo)" if key == "baseline" else f"+ pseudo conf >= {key}",
            "n_pseudo": 0 if key == "baseline" else n_used[key],
            "vcv_macro_f1": macro_f1(y_valid, pred),
            **dict(zip(ALL_LABELS, per_class)),
        })
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def write_outputs(client_ids: pd.Series, proba: np.ndarray, model_name: str,
                  jsonl_path: str = OUTPUT_JSONL, csv_path: str = OUTPUT_CSV) -> None:
    labels = np.array(ALL_LABELS)[proba.argmax(1)]
    conf = proba.max(1)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for cid, lab, c, p in zip(client_ids, labels, conf, proba):
            f.write(json.dumps({
                "client_id": cid, "cutoff_date": CUTOFF, LABEL_COL: lab,
                "confidence": round(float(c), 4), "model": model_name,
                **{f"p_{k}": round(float(v), 4) for k, v in zip(ALL_LABELS, p)},
            }) + "\n")
    out = pd.DataFrame({"client_id": client_ids, "cutoff_date": CUTOFF, LABEL_COL: labels,
                        "confidence": conf.round(4)})
    out.to_csv(csv_path, index=False)
    print(f"wrote {len(out)} pseudo-labels to {jsonl_path} and {csv_path}")
    print(pd.DataFrame({
        "share": out[LABEL_COL].value_counts(normalize=True),
        "mean_conf": out.groupby(LABEL_COL)["confidence"].mean(),
    }).reindex(ALL_LABELS).round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=PROBA_MODELS, default=DEFAULT_MODEL)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS,
                        help="comma-separated confidence cutoffs to evaluate")
    parser.add_argument("--skip-eval", action="store_true", help="only write the pseudo-labels")
    parser.add_argument("--teacher-data", choices=["train+valid", "train"], default="train+valid",
                        help="labelled clients the final teacher is fit on; 'train' keeps valid unseen")
    args = parser.parse_args()
    thresholds = [float(t) for t in args.thresholds.split(",")]
    build = MODELS[args.model]

    data = build_all()
    X_train, y_train = split_xy(data["train"])
    X_valid, y_valid = split_xy(data["valid"])
    X_valid = X_valid[X_train.columns]
    unl = unlabeled_features()
    X_unl = unl[X_train.columns]
    pathlib.Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_eval:
        print(f"\nvcv with pseudo-labels, model={args.model}:")
        preds, n_used = vcv_with_pseudo(build, X_train, y_train, X_valid, y_valid, X_unl, thresholds)
        report(preds, n_used, y_valid)

    if args.teacher_data == "train":
        X_full, y_full = X_train, y_train
        paths = (OUTPUT_JSONL_TRAINONLY, OUTPUT_CSV_TRAINONLY)
    else:
        X_full = pd.concat([X_train, X_valid], ignore_index=True)
        y_full = pd.concat([y_train, y_valid], ignore_index=True)
        paths = (OUTPUT_JSONL, OUTPUT_CSV)
    _, proba = teach(build, X_full, y_full, X_unl)
    print(f"\nteacher {args.model} fit on {len(X_full)} labelled clients ({args.teacher_data})")
    write_outputs(unl["client_id"], proba, args.model, *paths)


if __name__ == "__main__":
    main()
