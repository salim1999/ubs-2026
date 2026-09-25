# Transaction Activity Forecasting: Next Recurring Merchant Family

> How can we use AI to predict a client's future transaction activity from
> their historical financial transaction patterns?

For each client we predict the **next recurring merchant family** in the 90
days after the cutoff `2026-01-01`:

```text
cloud, gym, insurance, mobile, music, software, streaming, none
```

`none` means no recurring family recurs in the window. The metric is
**macro-F1** over all 8 labels, so every label counts equally.

Our approach: **find the right features, keep the model lean, make the
forecast understandable.** We engineer behavioural and timing features from
raw transactions and use a sparse, per-label logistic regression, so every
prediction can be explained by a few coefficients.

## Final model

| | |
|---|---|
| **Model** | **Sparse per-label logistic regression**: one L1-penalised (lasso) yes/no logistic regression per label ("is it gym?", ..., "is it none?"). Each label picks its own penalty by inner CV, so it keeps only its own predictors (music 23, none 80 of 103). Prediction = most probable label. |
| **Features** | `lean` set, 103 features in 5 blocks (see below) |
| **Training data** | 2,000 train clients + 3,152 pseudo-labelled pretrain clients whose three label sources agree (unsupervised, LLM, model trained on train only) |
| **Valid macro-F1** | **0.513** (fit without valid) |
| **Code** | `build_sparse_logreg()` in `src/model.py`, pseudo rows in `src/pseudo.py` |

The test predictions come from exactly this model; it is **not** refit on valid.

```bash
py -3.10 -m src.make_submission sparse --features lean --pseudo --suffix final
# -> data/submission_final.csv
```

### Results on valid (macro-F1, models fit without valid)

| Model | Valid macro-F1 |
|---|---|
| **Sparse per-label LogReg, lean + pseudo (final)** | **0.513** |
| Sparse per-label LogReg, lean, train only | 0.504 |
| Global L2 LogReg, lean | 0.489 |
| Due-date rule (no training) | 0.484 |
| Per-label gradient boosting (not used, black box) | 0.530 |

Benchmarking always reports both logistic models (`py -3.10 -m src.evaluate "<note>"`,
`--fast` for the lean models only). Noise level with 1,000 valid clients: ~0.01-0.02.

## Pipeline

```
raw transactions
  -> B  tag each transaction with a family (keywords first, then clean MCCs)     src/category_map.py
  -> C  detect recurring streams: pool a client's charges across MCCs,           src/recurrence.py
        cluster by amount, keep ~monthly and stable ones
  -> D  one feature row per client (5 blocks, below)                              src/features.py
  -> G  sparse per-label L1 logistic regression                                   src/model.py
  -> H  argmax of the 8 label probabilities
```

### Feature blocks (lean set) and their importance

Importance = macro-F1 points lost on valid when the block is shuffled.

| Block | Features | What it captures | Importance |
|---|---|---|---|
| Subscription status & timing | 42 | still live? overdue? due first? short trial? billing day | 22.5 |
| Habits | 7 | recent payments per family (last 90 days) | 13.5 |
| Subscription history | 28 | age, number of charges, amount per family | 4.4 |
| Portfolio changes & refunds | 13 | families gained or dropped, refunds | 3.6 |
| Account behaviour | 13 | payment mix (card / ATM / p2p), top-ups, merchant variety, travel | 2.9 |

`src/features.py` also keeps a `full` set (132 columns, every feature we built);
`lean` drops 21 redundant columns and most client-level story features.

## What the model tells us (verified on train and valid)

- **Habits are sticky:** recent payments in a family raise its odds ×1.9-2.8 per SD;
  no recent payment ~2% vs 3-5 payments ~31% renewal.
- **The calendar decides the race:** among several live subscriptions, the one due
  first wins; insurance due first lowers gym odds (×0.6).
- **Endings are predictable:** short trials (3-4 charges) end in `none` 67% of the time;
  one missed cycle -> 68% `none` vs 23% when the next charge is not yet due.
- **Lifestyle:** frequent travellers end with nothing new less often (38% -> 20% `none`);
  clients with varied spending go to gym / insurance / mobile, narrow digital users to streaming / cloud.
- Don't read correlated twins alone (`n_occurrences_*` vs subscription age): their signs offset.

Details: `extra_info/INTERPRETATION.md` (verified statements) and
`extra_info/COEFFICIENTS.md` (every label's coefficients as sentences).

## What did not help

Tested and dropped (full log with numbers: `extra_info/EXPERIMENT_LOG.md`):
client personas / clusters / segment models, fixed small scorecards,
last-k payment sequences, time-of-day and new-year / seasonal features,
income stress, relaxed lasso, invariance / anchor regression, per-class
decision biases, a competing-risks hazard model, billing-day due dates in the
model (they help the rule only), looser live definitions, lower weight for
pseudo rows. Main reasons: ~10% of targets are brand-new families with no trace
in the history, billing dates scatter by ±3 days, and train is cleaner than
valid / test (20% vs 6% of subscription charges on a foreign MCC).

## Repository

```
README.md               this document
src/
  category_map.py       transaction -> family
  recurrence.py         recurring-stream detection
  features.py           features + feature sets (full / lean)
  pseudo.py             pseudo-labelled pretrain clients (cached features)
  model.py              sparse per-label L1 (final), global LogReg, rule
  evaluate.py           score on valid + log
  make_submission.py    submission CSV
extra_info/             experiment log (+ experiments.csv), interpretation, coefficients
data/                   raw data (gitignored except the zips) + pretrain_labels_merged.csv (pseudo-labels, tracked)
```

Setup: Python >= 3.10, `pip install -e .`, unzip `data/dataset.zip` into `data/`.

## Challenge reference

**Data** (`data/`): `train_transactions.jsonl` + `train_labels.csv` (2,000 clients),
`valid_transactions.jsonl` + `valid_labels.csv` (1,000), `test_transactions.jsonl`
(1,000, hidden labels), `unlabeled_pretrain_transactions.jsonl` (10,000, no labels),
`sample_submission.csv`. Each transaction: `client_id, timestamp, amount, currency,
direction, type, mcc, description, fee`. History covers 2024-11 to 2025-12.

**Submission contract**: CSV with columns `client_id,predicted_next_recurring_merchant`,
exactly the client IDs of `sample_submission.csv` (one row each), only the 8 labels above.
Submit via [this form](https://forms.gle/3mgyM9D8d2quqXQ59) with team name and repo link;
the best macro-F1 over all valid milestone submissions counts.
