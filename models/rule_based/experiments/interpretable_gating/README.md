# Interpretable liveness and ranking experiment

This experiment asks whether the due-date baseline can be improved without
turning it into a black-box model. All choices are selected with the 2,000
training clients. The 1,000 validation clients are loaded only after the
rules are locked.

Run from the repository root:

```powershell
py -3.10 models/rule_based/experiments/interpretable_gating/experiment.py
```

The script detects recurring streams, caches them locally (the cache is
gitignored), runs the train-only searches, and writes the CSV/JSON artifacts
in this directory.

## Ideas tested

1. **Correct family aggregation.** The project feature builder uses the
   minimum `over_days` when a family has multiple streams, although the final
   rule ranks larger values as due sooner. The corrected rule first keeps the
   closest live stream in each family, then compares families.
2. **Global liveness and trial gate.** Search a small, declared set of maximum
   overdue windows and four simple trial gates.
3. **Confidence-adjusted due date.** Rank a live stream by

   `over_days - a * gap_cv - b * (1 - keyword_coverage)`

   where `a` and `b` are day-equivalent penalties selected on train.
4. **Per-family liveness.** Use one maximum-overdue threshold for each target
   family, selected in a single coordinate-search pass on train. This allows,
   for example, mobile and software to have different cancellation behavior.

## Results

| Rule | Train macro-F1 | Validation macro-F1 | Changed validation predictions |
|---|---:|---:|---:|
| Existing baseline | 0.4920 | **0.4844** | 0 |
| Correct multi-stream aggregation | 0.4897 | 0.4843 | 12 |
| Train-selected global overdue window (7 days) | 0.4922 | 0.4805 | 8 |
| Confidence-adjusted closest-live ranking | 0.4912 | 0.4813 | 30 |
| Per-family thresholds, existing `min` aggregation | 0.4968 | 0.4846 | 16 |
| Per-family thresholds + confidence/correct aggregation | 0.4970 | **0.4853** | 35 |

The numerically best variant uses these maximum-overdue windows:

| cloud | gym | insurance | mobile | music | software | streaming |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 7 | 7 | 10 | 7 | 0 | 5 |

It also uses two-day penalties for both gap irregularity and weak keyword
coverage. Its gain over baseline is only **0.0009 macro-F1**. A paired
2,000-resample client bootstrap gives a 95% delta interval of
**[-0.0088, +0.0110]** and only 58.9% of resamples favor it. This is not
convincing evidence of improvement.

## Recommendation

Keep the existing 0.4844 rule as the main model. The corrected multi-stream
aggregation is logically cleaner but empirically neutral. Per-family
liveness is the only extension worth retaining as a documented candidate;
it needs another untouched split or cross-cutoff backtest before promotion.

Key outputs:

- `results.csv`: overall scores, changes, and bootstrap uncertainty.
- `per_class_results.csv`: validation F1 for every label.
- `selected_configs.json`: exact locked rules.
- `*_search_train_only.csv`: full training-only search traces.
- `validation_predictions.csv`: auditable row-level predictions.
