"""Test submission: LogReg refit on train + valid, predict test.

    python -m src.make_submission [suffix]    # writes data/submission_<suffix>.csv
"""

import sys

import pandas as pd

from src.evaluate import CUTOFF, labelled_features, macro_f1
from src.features import build_features
from src.model import ALL_LABELS, LABEL_COL, build_logreg

OUT = f"data/submission_{sys.argv[1] if len(sys.argv) > 1 else 'thirdsubmission'}.csv"

train = labelled_features("train")
valid = labelled_features("valid")
X_train = train.drop(columns=["client_id", LABEL_COL])
X_valid = valid.drop(columns=["client_id", LABEL_COL]).reindex(columns=X_train.columns)

# Selection score: train-only fit, scored on valid.
print(f"valid macro-F1 (fit on train only): {macro_f1(valid[LABEL_COL], build_logreg().fit(X_train, train[LABEL_COL]).predict(X_valid)):.4f}")

# Final model: refit on train + valid.
X_full = pd.concat([X_train, X_valid], ignore_index=True)
y_full = pd.concat([train[LABEL_COL], valid[LABEL_COL]], ignore_index=True)
model = build_logreg().fit(X_full, y_full)

sample = pd.read_csv("data/sample_submission.csv")
test = sample[["client_id"]].merge(build_features("data/test_transactions.jsonl", CUTOFF), on="client_id", how="left")
X_test = test.drop(columns=["client_id"]).reindex(columns=X_train.columns)
sub = pd.DataFrame({"client_id": sample["client_id"], "predicted_next_recurring_merchant": model.predict(X_test)})

# Submission contract: exact client ids, one row each, allowed labels, column names.
assert list(sub.columns) == list(sample.columns)
assert len(sub) == len(sample) and sub["client_id"].is_unique
assert set(sub["client_id"]) == set(sample["client_id"])
assert sub["predicted_next_recurring_merchant"].isin(ALL_LABELS).all()

sub.to_csv(OUT, index=False)
print(f"wrote {OUT}: {len(sub)} rows")
