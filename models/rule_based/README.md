# Due-date rule model

This folder contains the project's primary rule-based model. It is deliberately
simple and has no fitted coefficients.

For each client, the model:

1. reads the live recurring-payment features produced by `src/features.py`;
2. predicts `none` when no recurring stream is live;
3. also predicts `none` when the strongest stream looks like a short
   3--4-payment trial;
4. otherwise selects the live merchant family whose next payment is due
   soonest.

The expected date comes from the stream's last payment and historical mean
interval. On the fixed validation split, this original model achieves **0.4844
macro-F1**.

## Robust median-gap candidate

Three independent extensions are documented in [`experiments/`](experiments/).
The strongest changes the expected interval from the historical mean to the
historical median:

```text
next due date = last payment date + median(historical payment gaps)
```

It remains entirely rule-based and improves validation macro-F1 to **0.4969**.
It is available as `median_gap_predict` from this package; the original
`rule_predict` is retained as the comparison baseline.

Generate its test submission from the repository root:

```powershell
py -3.10 -m src.make_submission rule
```
