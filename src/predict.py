"""Produces the competition submission from the processed feature tables.

The model is chosen by validation macro-F1 (fit on train, scored on valid)
unless --model names one. It is then refit on train + valid (all labelled
clients) and predicts target_next_recurring_merchant for every client in
sample_submission.csv. Clients missing from test_features.csv (no usable
history) fall back to 'none', the majority class.

Usage (after `python -m src.features`):
    python -m src.predict [--model auto|logreg|hgb] [--out submission.csv]
"""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd
from sklearn.metrics import f1_score

from src.model import ALL_LABELS, build_hgb, build_logreg, load_xy

MODELS = {"logreg": build_logreg, "hgb": build_hgb}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=["auto", *sorted(MODELS)], default="auto")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--data-dir", default="data/dataset/dataset")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    processed = pathlib.Path(args.processed_dir)
    X_train, y_train = load_xy(str(processed / "train_features.csv"))
    X_valid, y_valid = load_xy(str(processed / "valid_features.csv"))
    X_full = pd.concat([X_train, X_valid], ignore_index=True)
    y_full = pd.concat([y_train, y_valid], ignore_index=True)

    test = pd.read_csv(processed / "test_features.csv")
    X_test = test.drop(columns=["client_id"])[X_full.columns]

    name = args.model
    if name == "auto":
        scores = {}
        for candidate, build in MODELS.items():
            pred = build().fit(X_train, y_train).predict(X_valid)
            scores[candidate] = f1_score(y_valid, pred, average="macro", labels=ALL_LABELS, zero_division=0)
            print(f"  valid macro-F1 {candidate:7s} {scores[candidate]:.4f}")
        name = max(scores, key=scores.get)

    model = MODELS[name]()
    model.fit(X_full, y_full)
    preds = pd.Series(model.predict(X_test), index=test["client_id"])

    sample = pd.read_csv(pathlib.Path(args.data_dir) / "sample_submission.csv")
    n_missing = (~sample["client_id"].isin(preds.index)).sum()
    submission = pd.DataFrame(
        {
            "client_id": sample["client_id"],
            "predicted_next_recurring_merchant": sample["client_id"].map(preds).fillna("none"),
        }
    )

    assert submission["predicted_next_recurring_merchant"].isin(ALL_LABELS).all()
    submission.to_csv(args.out, index=False)

    print(f"Fit {name} on {len(X_full)} labelled clients; wrote {len(submission)} rows to {args.out}")
    if n_missing:
        print(f"  {n_missing} clients had no features and were set to 'none'")
    print(submission["predicted_next_recurring_merchant"].value_counts().to_string())


if __name__ == "__main__":
    main()
