"""Produces the competition submission from the processed feature tables.

The model is chosen by vcv macro-F1 (src.evaluate: 5 folds over the valid
clients, each fit on train + the other 4/5 of valid) unless --model names
one; the single train->valid split is too noisy (+-~0.02) to choose on. It is then refit on train + valid (all labelled
clients) and predicts target_next_recurring_merchant for every client in
sample_submission.csv. Clients missing from test_features.csv (no usable
history) fall back to 'none', the majority class.

Usage (after `python -m src.features`):
    python -m src.predict [--model auto|rule|logreg|hgb|rank|ens] [--out submission.csv]
"""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd

from src.evaluate import macro_f1, vcv_predictions
from src.model import ALL_LABELS, MODELS, load_xy


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
            pred = vcv_predictions(build, X_train, y_train, X_valid[X_train.columns], y_valid)
            scores[candidate] = macro_f1(y_valid, pred)
            print(f"  vcv macro-F1 {candidate:7s} {scores[candidate]:.4f}")
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
