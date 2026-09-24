# Rule-based model experiments

The original rule estimates a stream's next payment as:

```text
last payment date + mean historical payment interval
```

Three independent extensions were selected using training data only and then
scored once on the fixed validation split.

| Experiment | Validation macro-F1 | Change vs 0.4844 baseline | Decision |
|---|---:|---:|---|
| Robust median cadence | **0.4969** | **+0.0124** | Promote as candidate |
| Least-squares timing | 0.4797 | -0.0047 | Do not promote |
| Interpretable liveness/gating | 0.4853 | +0.0009 | Inconclusive |

## Recommended advancement

Use the median rather than the mean historical gap:

```text
next due date = last payment date + median(historical payment gaps)
```

This change is robust to a skipped, early, or delayed payment and remains fully
account-specific and explainable. A paired 5,000-resample client bootstrap of
the fixed validation predictions estimated a macro-F1 improvement of `+0.0125`
with a 95% interval of `[+0.0016, +0.0243]`; 98.7% of resamples favored the
median rule.

The callable candidate is `models.rule_based.median_gap_predict`. Full model
selection and audit artifacts are in `robust_cadence/`.

## What the other experiments taught us

- Ridge regression reduced next-gap MAE from 4.28 to 3.77 days, but better
  timing accuracy did not translate into better merchant-family macro-F1.
- Family-specific overdue thresholds and confidence penalties changed few
  predictions and produced no credible improvement.
- Correcting the multi-stream family aggregation was logically attractive but
  empirically neutral on this validation split.

Every experiment has its own runnable script, README, train-only selection
trace, and locked validation results.
