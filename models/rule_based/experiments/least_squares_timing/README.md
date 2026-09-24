# Least-squares next-payment timing

This experiment keeps the existing rule-based decision but replaces its raw
historical mean interval with a pooled ridge-regression estimate of the next
interval.

For every recurring stream, rolling historical pseudo-cutoffs provide training
examples. The regression inputs are previous gap statistics, occurrence count,
amount stability, merchant family, and (for the calendar variant) the current
month/day. Merchant labels after the cutoff are **not** used to fit timing.

The decision remains transparent:

```text
predicted next date = last payment date + predicted interval
discard streams that are too far overdue
choose the family whose predicted payment is earliest
otherwise predict none
```

Feature variant and ridge strength are selected by grouped cross-validation on
pseudo-history, with whole payment streams kept in one fold. Blend weight and
the small overdue tolerance are selected on the training labels only. Validation
is read only after these settings are locked.

Run from the repository root:

```powershell
py -3.10 models/rule_based/experiments/least_squares_timing/experiment.py
```

Optional: use the larger unlabeled transaction set for pseudo-history:

```powershell
py -3.10 models/rule_based/experiments/least_squares_timing/experiment.py --pseudo-source both
```

Outputs:

- `pseudo_timing_cv.csv`: grouped pseudo-history timing comparison
- `train_decision_grid.csv`: train-only decision settings
- `results.json`: locked validation macro-F1 and per-class F1

## Result

The run used only the 2,000 labeled-training clients' transaction histories to
create 16,871 pseudo-cutoff examples from 3,165 streams. Labels were not needed
for the timing fit. Grouped pseudo-history CV selected the calendar feature set
with ridge `alpha=100`.

| Measurement | Historical mean | Ridge timing |
|---|---:|---:|
| Pseudo-history next-gap MAE | 4.2801 days | **3.7726 days** |
| Held-out validation macro-F1 | **0.4844** | 0.4797 |

The train-only decision grid selected a 50/50 blend of the ridge estimate and
historical mean, with a three-day overdue allowance. Its training macro-F1 was
0.4915. On the untouched 1,000-client validation split it lost 0.0047 macro-F1.

Per-class validation F1 for the learned timing rule:

| cloud | gym | insurance | mobile | music | software | streaming | none |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5521 | 0.5311 | 0.4585 | 0.5314 | 0.4022 | 0.4035 | 0.4211 | 0.5376 |

## Interpretation

Ridge predicts the exact next interval more accurately, but macro-F1 only cares
whether the winning family is ranked first. Small timing improvements usually do
not change that winner; a few changed rankings are harmful. The plain mean-gap
rule therefore remains the recommended model. This experiment is evidence that
the next useful improvement should focus on stream liveness/category detection,
not a more elaborate interval estimator.
