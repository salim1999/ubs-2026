# Handover: Next Recurring Merchant Prediction

Start here for team context. The official challenge spec is in [README.md](README.md).
Step names (A–I) for discussing changes, plus the experiment log: [PIPELINE.md](PIPELINE.md).

## Goal

For each client, predict the **next recurring merchant family** in the 90 days
after the cutoff date `2026-01-01`. The score is **macro-F1** across 8 labels,
so every class counts equally, including `none`.

```
cloud, gym, insurance, mobile, music, software, streaming, none
```

## Data at a glance

| Split | Clients | Transactions | Labels |
|---|---|---|---|
| train | 2,000 | 147k | ✅ `train_labels.csv` |
| valid | 1,000 | 74k | ✅ `valid_labels.csv` |
| test | 1,000 | 76k | ❌ hidden (we submit these) |
| pretrain | 10,000 | 750k | ❌ optional extra histories |

- **Time window:** every split covers 2024-11-07 → 2025-12-31. The split is by
  **client**, not by time, and the labels describe the unseen period Jan–Mar 2026.
- **Label balance:** `none` is about 30% of labels; each other class is about 10%.
- **Columns:** `client_id, timestamp, amount, currency, direction, type, mcc,
  description, fee`. `description` is the only free-text column.
- **What's missing:** there is **no "recurring" flag and no "family" column**.
  Both have to be inferred.

## Key findings

1. **MCC alone isn't enough.** Streaming and music sit under 5812 (restaurants)
   and cloud under 5732 (electronics). Use `mcc` + `description`.
2. **Descriptions rotate.** The same subscription shows up as "premium plan",
   "media streaming", "digital plus"… The stable signals are the **amount** and
   the **day of the month**.
3. **The data contains decoys.** Examples: "member cloud backup digital" under
   grocery (5411), and subscription words under the wrong MCC.
4. **Most clients have about 2 recurring families.** The label is usually the one
   **due next after the cutoff**, not just any one the client has.
5. **`none` clients also have recurring streams** (about 2.3 on average).
   Separating them from the rest is the hardest part.

| MCC | Family | Typical amount |
|---|---|---|
| 7997 | gym | ~68 |
| 6300 | insurance | ~110 |
| 4814 | mobile | ~47 |
| 5734 | software | ~39 |
| 5732 (cloud/backup/storage text) | cloud | varies |
| 5812 (audio streaming) | music | ~10–20 |
| 5812 (media streaming / video access) | streaming | ~10–20 |

## Approach

```
transactions ─▶ A. family tagging ─▶ B. recurring-stream detection ─▶ stream table
                  (mcc + description)   (amount clusters, ~monthly gap)
                                                                    │
labels ─▶ C. eval harness (macro-F1) ◀── D. predict next due / none ◀┘
                                              (rule → GBDT → ensemble)
```

- **Part 1, pattern detection:** tag each transaction with a family, then build
  one row per client × family with `n_occurrences`, `mean_gap_days`,
  `last_date`, `mean_amount` and `gap_cv`.
- **Part 2, prediction:** pick the family whose next charge
  (`last_date + mean_gap`) falls first after the cutoff, or `none`. Start with a
  rule, then learn it with a classifier.

## Code (`src/`)

| File | Does |
|---|---|
| `category_map.py` | Transaction → family (keywords first, then MCC fallback for 4814/5734/6300/7997) |
| `recurrence.py` | Stream detection per client × MCC; amount clustering for 5812/5732 |
| `features.py` | One feature row per client: general behaviour, per-family stream stats, recent activity |
| `model.py` | Majority / rule / logistic regression / gradient boosting baselines, scored with macro-F1 on valid |

### Run

Requires Python ≥ 3.12 (see `pyproject.toml`).

```bash
pip install -e .                               # or: uv sync
unzip data/dataset.zip -d data/                # raw jsonl/csv files
unzip data/processed.zip -d data/              # prebuilt feature CSVs
python -m src.recurrence data/train_transactions.jsonl
python -m src.features   data/train_transactions.jsonl 2026-01-01
python -m src.model                            # reads data/processed/*_features.csv
```

## Known issues / next steps (priority order)

1. **The rule picks the most recently paid family.** For monthly subscriptions
   that's the one due *last*. Fix: add
   `days_until_next_due = mean_gap_days - recency_days` per family in
   `features.py`, and sort by it in `model.rule_predict`. A draft of this fix
   is in `git stash` on this branch.
2. **Stream detection recall is suspect.** The code comment says about 47% of
   targets are "brand-new categories", but a quick check found about 80% of
   labelled families recurring in 3+ months of history. Things to check:
   skipped months pushing the gap above 45 days, double charges in one month,
   and non-clustered MCCs mixing in noise.
3. **Decoys leak into the rule.** Keyword-first tagging lets decoy transactions
   become "recent adoption" predictions.
4. **The pipeline is incomplete.** Nothing regenerates `data/processed/*.csv`
   with labels, and nothing predicts test or writes the submission CSV.
5. **More training data:** move the cutoff back (e.g. 2025-10-01) to create
   extra labels on train + pretrain (14k clients).
6. **Housekeeping:** add a `.gitignore` (`__pycache__/`, data dumps), merge
   `main` (1 commit behind), and add `encoding="utf-8"` to `open()`.

## Submission

- **File format:** `client_id,predicted_next_recurring_merchant`, with exactly
  the IDs from `sample_submission.csv` and only the 8 allowed labels.
- **Where:** submit via the form linked in [README.md](README.md#submission-workflow).
- **Milestones:** Day 1 12:00 · Day 1 17:00 · Day 2 12:00 · Day 2 17:30 (final).

## Branches

| Branch | Owner | Content |
|---|---|---|
| `main` | – | Challenge spec + data |
| `shipra1` | colleague | First submission code |
| `timmyo` | Tim | Copy of `shipra1` + this handover |
