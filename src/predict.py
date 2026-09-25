"""Produces the competition submission from the processed feature tables.

The model is chosen by vcv macro-F1 (src.evaluate: 5 folds over the valid
clients, each fit on train + the other 4/5 of valid) unless --model names
one; the single train->valid split is too noisy (+-~0.02) to choose on. It is then refit on train + valid (all labelled
clients) and predicts target_next_recurring_merchant for every client in
sample_submission.csv. Clients missing from test_features.csv (no usable
history) fall back to 'none', the majority class.

--pseudo-labels adds the unlabeled pretrain clients labelled by src.pseudo_label
(a rank teacher fit on all 3,000 labelled clients) to the final fit, keeping
those with confidence >= --pseudo-min-conf. Leak-free vcv (teacher refit per
fold, 2026-09-24): rank 0.5445 -> 0.5670 at conf >= 0.5, and every cutoff in
0..0.8 helped (0.558-0.567). Model choice in --model auto stays on vcv without
pseudo-labels: the file's teacher saw all valid labels, so scoring it on valid
folds is leaky (0.5837 at 0.5).

Usage (after `python -m src.features`; --pseudo-labels also needs
data/processed/unlabeled_features.csv from `python -m src.pseudo_label`):
    python -m src.predict [--model auto|rule|logreg|hgb|rank|ens] [--out submission.csv]
                          [--pseudo-labels data/labeled_pretrain_clients.csv] [--pseudo-min-conf 0.5]
"""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd

from src.evaluate import macro_f1, vcv_predictions
from src.model import ALL_LABELS, LABEL_COL, MODELS, load_xy

PSEUDO_MIN_CONF = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=["auto", *sorted(MODELS)], default="auto")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--data-dir", default="data/dataset/dataset")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--pseudo-labels", default=None,
                        help="csv with client_id, label, confidence for the unlabeled clients")
    parser.add_argument("--pseudo-min-conf", type=float, default=PSEUDO_MIN_CONF)
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

    X_fit, y_fit = X_full, y_full
    if args.pseudo_labels:
        unl = pd.read_csv(processed / "unlabeled_features.csv")
        pseudo = pd.read_csv(args.pseudo_labels).set_index("client_id").reindex(unl["client_id"])
        assert pseudo[LABEL_COL].notna().all(), "pseudo-labels missing for some unlabeled clients"
        keep = (pseudo["confidence"] >= args.pseudo_min_conf).to_numpy()
        X_fit = pd.concat([X_full, unl.loc[keep, X_full.columns]], ignore_index=True)
        y_fit = pd.concat([y_full, pseudo.loc[keep, LABEL_COL].reset_index(drop=True)], ignore_index=True)
        print(f"  + {keep.sum()} pseudo-labelled clients (conf >= {args.pseudo_min_conf})")

    model = MODELS[name]()
    model.fit(X_fit, y_fit)
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

    print(f"Fit {name} on {len(X_full)} labelled + {len(X_fit) - len(X_full)} pseudo-labelled clients; wrote {len(submission)} rows to {args.out}")
    if n_missing:
        print(f"  {n_missing} clients had no features and were set to 'none'")
    print(submission["predicted_next_recurring_merchant"].value_counts().to_string())


if __name__ == "__main__":
    main()
