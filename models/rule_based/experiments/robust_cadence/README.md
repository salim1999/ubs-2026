# Robust cadence experiment

This experiment changes one part of the due-date rule: how it estimates a
stream's usual payment interval. The production baseline uses the arithmetic
mean of all observed gaps:

```text
next due date = last payment date + mean(historical gaps)
```

The selected rule uses the median instead:

```text
next due date = last payment date + median(historical gaps)
```

The median is still simple and account-specific, but one late, skipped, or
early payment cannot pull it as strongly as the mean. Everything else remains
rule-based: the same recurring-stream detector is used, a stream more than the
chosen grace period overdue is treated as inactive, the existing 3--4 payment
short-stream gate predicts `none`, and the family due first is selected.

## Leakage-safe selection

Six cadence estimates were compared using **train labels only**. For each one,
the liveness grace period was tried at 0, 3, 5, 7, and 10 days. Only after the
best pair had been fixed was it evaluated on validation.

Best train setting for each cadence family:

| Cadence estimate | Best grace | Train macro-F1 |
|---|---:|---:|
| Median of all gaps | 5 days | **0.503286** |
| 50/50 mean-median blend | 5 days | 0.498242 |
| Trimmed mean (drop min/max with >=5 gaps) | 5 days | 0.495221 |
| Median of latest 3 gaps | 7 days | 0.494725 |
| Mean of all gaps | 5 days | 0.492846 |
| Median day-of-month calendar forecast | 3 days | 0.458472 |

The complete 30-setting train table is in `train_selection.csv`.

## Final result

| Rule | Untouched validation macro-F1 |
|---|---:|
| Existing mean-gap baseline | 0.484420 |
| Selected median-gap rule, 5-day grace | **0.496853** |

Absolute improvement: **+0.012432 macro-F1** (about 2.6% relative).

A paired client bootstrap was run on the two fixed validation prediction
vectors (5,000 resamples of the 1,000 clients, seed 17). It does not refit or
retune either rule:

| Bootstrap quantity | Result |
|---|---:|
| Mean macro-F1 delta | +0.012512 |
| Percentile 95% interval for delta | **[+0.001556, +0.024273]** |
| Fraction of resamples with delta > 0 | **0.9866** |

The paired interval excludes zero, providing useful evidence that the median
improvement is not just which particular clients landed in validation. It is
still an uncertainty estimate from one validation set, not a second external
test set.

Per-class validation F1 for the selected rule:

| cloud | gym | insurance | mobile | music | software | streaming | none |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5625 | 0.5432 | 0.4950 | 0.5288 | 0.4545 | 0.4103 | 0.4444 | 0.5360 |

The raw-stream reconstruction of the mean rule disagrees with the official
feature-based baseline on 8 of 1,000 validation clients (boundary/tie handling)
and scores 0.485502. The reported comparison above conservatively uses the
official baseline score, not that reconstruction.

## Run

From the repository root:

```powershell
py -3.10 models/rule_based/experiments/robust_cadence/experiment.py
```

The script writes `train_selection.csv`, `validation_predictions.csv`, and
`results.json` in this directory. The prediction file contains client ID, true
label, official baseline prediction, and median-gap prediction, so the paired
comparison can be audited. The script does not modify the production model.
