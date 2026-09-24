"""Confusion matrices + failure analysis for the two current best models,
trained on train_features.csv ONLY (not combined with valid - this is a
pure held-out read, distinct from the train+valid refit used for the
actual test submission in the ad-hoc script that produced
data/submission_logreg_l1.csv / data/submission_hgb_tuned.csv).

Three things per model:
  1. Confusion matrix on valid_features.csv.
  2. Accuracy by segment (continuation / new-adoption / true-none) - see
     src/analyze.py for why this split matters.
  3. Top confusion pairs.

Plus one cross-model comparison: how much do the two models' mistakes
overlap? If they fail on different clients, an ensemble would be worth
trying next; if they fail on the same clients, it wouldn't.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from src.analyze import continuation_vs_adoption_breakdown, top_confusions
from src.category_map import TARGET_CATEGORIES
from src.model import ALL_LABELS, build_hgb_tuned, build_logreg_l1, load_xy, set_global_seed

if __name__ == "__main__":
    set_global_seed()

    X_train, y_train = load_xy("data/processed/train_features.csv")
    X_valid, y_valid = load_xy("data/processed/valid_features.csv")

    models = {
        "logreg_l1": build_logreg_l1(),
        "hgb_tuned": build_hgb_tuned(),
    }

    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        preds[name] = pred
        macro_f1 = f1_score(y_valid, pred, average="macro", labels=ALL_LABELS, zero_division=0)

        print(f"{'=' * 70}")
        print(f"=== {name}  (macro-F1 on valid, train-only fit: {macro_f1:.4f}) ===")
        print(f"{'=' * 70}")

        cm = confusion_matrix(y_valid, pred, labels=ALL_LABELS)
        cm_df = pd.DataFrame(cm, index=ALL_LABELS, columns=ALL_LABELS)
        print("--- Confusion matrix (rows=true, cols=predicted) ---")
        print(cm_df)
        print()

        continuation_vs_adoption_breakdown(X_valid, y_valid, pred)
        top_confusions(y_valid, pred, top_n=8)

    print(f"{'=' * 70}")
    print("=== Cross-model error overlap ===")
    print(f"{'=' * 70}")
    correct_l1 = preds["logreg_l1"] == y_valid.values
    correct_hgb = preds["hgb_tuned"] == y_valid.values

    both_correct = (correct_l1 & correct_hgb).sum()
    both_wrong = (~correct_l1 & ~correct_hgb).sum()
    only_l1_correct = (correct_l1 & ~correct_hgb).sum()
    only_hgb_correct = (~correct_l1 & correct_hgb).sum()

    print(f"Both correct:        {both_correct:4d} ({both_correct/len(y_valid):.1%})")
    print(f"Both wrong:          {both_wrong:4d} ({both_wrong/len(y_valid):.1%})")
    print(f"Only logreg_l1 right:{only_l1_correct:4d} ({only_l1_correct/len(y_valid):.1%})")
    print(f"Only hgb_tuned right:{only_hgb_correct:4d} ({only_hgb_correct/len(y_valid):.1%})")

    agreement = (preds["logreg_l1"] == preds["hgb_tuned"]).mean()
    print(f"\nPrediction agreement rate (regardless of correctness): {agreement:.1%}")
