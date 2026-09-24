"""Build a test submission with either of the project's two main models.

    py -3.10 -m src.make_submission rule
    py -3.10 -m src.make_submission logreg
"""

import argparse

import pandas as pd

from src.evaluate import CUTOFF, labelled_features, macro_f1
from src.features import build_features
from src.model import ALL_LABELS, LABEL_COL, build_logreg, rule_predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["rule", "logreg"])
    parser.add_argument("--suffix", help="output suffix (defaults to the model name)")
    args = parser.parse_args()
    out_path = f"data/submission_{args.suffix or args.model}.csv"

    train = labelled_features("train")
    valid = labelled_features("valid")
    X_train = train.drop(columns=["client_id", LABEL_COL])
    X_valid = valid.drop(columns=["client_id", LABEL_COL]).reindex(columns=X_train.columns)

    if args.model == "rule":
        valid_pred = rule_predict(X_valid)
    else:
        selection_model = build_logreg().fit(X_train, train[LABEL_COL])
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
        final_model = build_logreg().fit(X_full, y_full)
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
