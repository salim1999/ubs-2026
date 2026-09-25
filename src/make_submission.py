"""Build a test submission with one of the project's models.

    py -3.10 -m src.make_submission rule
    py -3.10 -m src.make_submission logreg    # benchmark 1: global LogReg
    py -3.10 -m src.make_submission sparse    # benchmark 2: sparse per-label LogReg
    py -3.10 -m src.make_submission sparse --features lean   # fewer-features model
"""

import argparse

import pandas as pd

from src.evaluate import CUTOFF, labelled_features, macro_f1
from src.features import FEATURE_SETS, build_features, select_features
from src.model import ALL_LABELS, LABEL_COL, build_logreg, build_sparse_logreg, rule_predict

BUILDERS = {"logreg": build_logreg, "sparse": build_sparse_logreg}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["rule", "logreg", "sparse"])
    parser.add_argument("--features", choices=FEATURE_SETS, default="full",
                        help="feature set: full (all features, default) or lean (fewer-features model)")
    parser.add_argument("--suffix", help="output suffix (defaults to the model name)")
    args = parser.parse_args()
    default_suffix = args.model if args.features == "full" else f"{args.model}_{args.features}"
    out_path = f"data/submission_{args.suffix or default_suffix}.csv"

    train = labelled_features("train")
    valid = labelled_features("valid")
    X_train = select_features(train.drop(columns=["client_id", LABEL_COL]), args.features)
    X_valid = valid.drop(columns=["client_id", LABEL_COL]).reindex(columns=X_train.columns)

    if args.model == "rule":
        valid_pred = rule_predict(X_valid)
    else:
        selection_model = BUILDERS[args.model]().fit(X_train, train[LABEL_COL])
        valid_pred = selection_model.predict(X_valid)
    print(f"{args.model} valid macro-F1: {macro_f1(valid[LABEL_COL], valid_pred):.4f}")

    sample = pd.read_csv("data/sample_submission.csv")
    test = sample[["client_id"]].merge(
        build_features("data/test_transactions.jsonl", CUTOFF),
        on="client_id",
        how="left",
    )
    X_test = test.drop(columns=["client_id"]).reindex(columns=X_train.columns)

    if args.model == "rule":
        test_pred = rule_predict(X_test)
    else:
        X_full = pd.concat([X_train, X_valid], ignore_index=True)
        y_full = pd.concat(
            [train[LABEL_COL], valid[LABEL_COL]], ignore_index=True
        )
        final_model = BUILDERS[args.model]().fit(X_full, y_full)
        test_pred = final_model.predict(X_test)

    sub = pd.DataFrame(
        {
            "client_id": sample["client_id"],
            "predicted_next_recurring_merchant": test_pred,
        }
    )

    # Submission contract: exact client IDs, one row each, allowed labels.
    assert list(sub.columns) == list(sample.columns)
    assert len(sub) == len(sample) and sub["client_id"].is_unique
    assert set(sub["client_id"]) == set(sample["client_id"])
    assert sub["predicted_next_recurring_merchant"].isin(ALL_LABELS).all()

    sub.to_csv(out_path, index=False)
    print(f"wrote {out_path}: {len(sub)} rows")


if __name__ == "__main__":
    main()
