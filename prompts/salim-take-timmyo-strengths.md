# Prompt: fix `Salim`'s weak spots with `timmyo`'s strengths

Paste everything below the line into Claude Code, started in a checkout of
this repo on branch `Salim`.

---

You are working in the `ubs-2026` repo on branch **`Salim`**. Read `README.md`
first. The task is to predict each client's next recurring merchant family
(`cloud, gym, insurance, mobile, music, software, streaming, none`) in the 90
days after the cutoff `2026-01-01`. The score is **macro-F1** over those 8
labels.

`timmyo` (Tim's branch) scores clearly higher than `Salim`. Its rule-based
predictor gets **0.487** valid macro-F1, and its tuned HGB about 0.47.
`Salim`'s best model gets about **0.40**. Almost all of the gap comes from how
`timmyo` finds subscription streams, not from the models. `Salim` has its own
strengths that `timmyo` lacks: persona scores, active-vs-lapsed features,
`predict.py`, and tuned models. Keep those.

Your job is to replace each `Salim` weakness below with the matching `timmyo`
strength, one at a time, and measure each step. Run `git fetch origin` first.
**Do not `git merge origin/timmyo`.** It carries committed `__pycache__/`, a
different data layout (`data/*.jsonl`), and it deletes `personas.py` and
`predict.py`. Port code with `git show origin/timmyo:<path>`, or copy it by
hand, and mention "ported from timmyo" in each commit message.

## Weakness → strength map

### W1. Streams are grouped by family first, so they fall apart when merchants switch MCC or description (the biggest gap)
- **`Salim` now:** `src/recurrence.py::_client_streams` labels each row with
  `classify()`. It clusters amounts inside each family, then attaches rows
  with generic descriptions to the nearest-amount cluster.
- **Why it fails:** the same subscription switches MCC and description from
  month to month. One streaming charge can read "digital plus", then "premium
  plan", then "media streaming", while its MCC alternates between 5812 and
  5411. About 20% of valid charges switch MCC, against 6% in train. Streams
  get split into pieces that fail the `n >= 2` and cadence checks.
- **`timmyo` strength:** `detect_streams` in `origin/timmyo:src/recurrence.py`:
  1. Keep outgoing transactions only (`direction == "out"`).
  2. Drop types `atm, fee, p2p_transfer`. Drop any row whose description
     contains a phrase from `NON_SUBSCRIPTION_PHRASES`, matched as a substring
     on any MCC.
  3. Put all of a client's remaining rows in one pool, across every MCC, and
     cluster them by amount.
  4. Label each cluster by majority vote over its members' keyword families,
     and record `category_agreement` and `n_keyword_hits`.
- **Measured effect** (from `timmyo`'s `experiments.csv`):
  - `c_detect`: 0.37 → 0.67 on valid.
  - LogReg: 0.374 → 0.424.
  - HGB: 0.353 → 0.441.

### W2. The amount tolerance is too loose once charges are pooled
- **`Salim` now:** `AMOUNT_REL_TOL = 0.25` and `AMOUNT_ABS_TOL = 3.0`.
- **Why it fails:** once all MCCs share one pool, a 25% tolerance merges
  neighbouring subscriptions, for example cloud at about 9 and music at
  about 11.
- **`timmyo` strength:** a cluster breaks when an amount is more than
  `max(0.3, 0.03 × running mean)` above the running mean.
  - On train, 0.5/0.05 down to 0.2/0.02 detect the same streams.
  - 0.1/0.01 starts splitting real streams.

### W3. Cadence bands beyond monthly add false streams
- **`Salim` now:** `CADENCE_BANDS` covers weekly through annual, with
  `MAX_GAP_CV = 0.6`.
- **`timmyo` strength:** monthly only, with an average gap of 20–45 days and
  `MAX_GAP_CV = 0.5`.
- A shared benchmark tried adding `Salim`'s bands on top of `timmyo`'s
  detection. It was negative: LogReg lost about 0.02, and the valid ensemble
  lost about 0.03.
- If you keep `cadence_code` as a feature, it becomes constant. Drop it.

### W4. The "still active" test is far too loose
- **`Salim` now:** a stream counts as live if
  `recency_days <= 2 × median_gap + 5`. Under that rule a monthly stream that
  has been silent for 65 days still counts as live.
- **`timmyo` strength:** in `origin/timmyo:src/features.py::_live_stream_features`:
  - A stream counts as live only if its next charge is overdue by **at most
    5 days** (`LIVE_MAX_OVER_DAYS = 5`).
  - It adds these features: `over_days_<cat>`, `is_live_<cat>`,
    `n_live_streams`, `max_live_n_occurrences`, `min_over_days`, and
    `short_live_stream`.
  - `short_live_stream` is set when the longest live stream has only 3–4
    charges, which looks like a trial that ends.
  - Measured effect: LogReg 0.424 → 0.448, HGB 0.441 → 0.456.
- **How to port:** set `Salim`'s own live/lapsed split to the 5-day
  threshold, and keep `lapsed_<cat>` so "had it, dropped it" is still a
  separate state. Add `timmyo`'s `short_live_stream`, `max_live_n_occurrences`
  and `min_over_days`. Don't duplicate columns: `over_days` is just
  `-days_to_next`.

### W5. The rule baseline is weak
- **`Salim` now:** `src/model.py::rule_predict` prefers categories that have
  recent transactions but no active stream. Its own docstring calls it weak.
- **`timmyo` strength:** a rule that knows which streams are still live:
  - Predict `none` if no stream is live, or if `short_live_stream == 1`.
  - Otherwise predict the live family due soonest.
  - It scores **0.472–0.487** valid, which beats every trained model on
    `timmyo`.
- **How to port:** replace `rule_predict`, then make `src/predict.py` treat
  the rule as a candidate model. If the rule wins on the vcv protocol below,
  it can be the submission.

### W6. Evaluation is optimistic and nothing is logged
- **`Salim` now:** `src/evaluate.py` runs 5-fold CV over train+valid pooled
  together. That is optimistic: valid clients are noisier than train and look
  more like test (median gap_cv is 0.11 in train, 0.29 in valid, 0.33 in
  test). No history of runs is kept.
- **`timmyo` strength:** in `origin/timmyo:src/evaluate.py`:
  - Every run appends a row to `experiments.csv` with a note, the scores, and
    step-C diagnostics.
  - `c_detect` is the share of non-`none` clients whose true family has a
    detected stream.
  - `c_nextdue` is the share of those clients whose soonest-due stream is the
    true family.
  - `c_none_active` and `c_fams` are also logged.
- **How to port:** combine the two harnesses.
  1. Rebuild features from the raw jsonl. Fit the persona scorer on train
     only.
  2. Score each model on **valid**: fit on train, score valid.
  3. Also score each model on **vcv**, and use it for decisions. Split the
     valid clients with `StratifiedKFold(5, shuffle=True, random_state=0)`.
     For each fold, fit on train plus the other 4/5 of valid and predict the
     held-out fifth. Compute one macro-F1 over all 1,000 pooled predictions.
  4. Print per-class F1 and the step-C diagnostics.
  5. Append one row to `experiments.csv` per run, including the git short
     SHA.

## Keep from `Salim` (do not regress)
- `src/personas.py`, with the scorer fit on train only. It is a small but
  consistent gain of +0.005 to +0.01.
- `lapsed_<cat>`, `mean_gap_days_<cat>`, `days_to_next_<cat>`, `due_rank`,
  `is_soonest`.
- `src/predict.py`: choose the model, refit on train+valid, write the
  submission in `sample_submission.csv` order.
- The data layout: `unzip data/dataset.zip -d data/dataset`.
- Python ≥3.11. Use `encoding="utf-8"` on every `open()`.
- **The classifier is an open question, not a weakness.** `Salim`'s
  `classify()` is richer than `timmyo`'s: canonical phrases, abbreviation
  expansion, and affix stripping. `timmyo`'s is keyword-first with an MCC
  fallback on only 4 clean MCCs. Start with `timmyo`'s version, since that is
  what was measured. Then try `Salim`'s `classify()` as the per-row voter
  inside `timmyo`'s clustering, as experiment 4 below.

## Interface gap you must close

`Salim`'s `features.py` reads stream fields that `timmyo`'s `_stream_stats`
does not produce:
- `median_gap_days`, `cadence`
- `n_last_30d/60d/90d`
- `n_refunds`, `refund_ratio`, `last_event_is_refund`
- `amount_trend`, `n_mccs`
- `cat_strength`

Handle them as follows:
- **Add back to `timmyo`'s `_stream_stats`:** `median_gap_days` (switch to
  the median gap), `n_last_*d` (needs `cutoff`), `amount_trend`, and
  `n_mccs`.
- **Refund fields:** match each client's refunds to a cluster by amount,
  using the same tolerance as W2.
- **Drop:** `cadence`, `cadence_code`, and `cat_strength`.

Check that the feature matrix has no all-NaN columns before scoring.

## Order of work (commit after each step, log each run)

0. **Baseline.** Build the combined `evaluate.py` from W6 and run it on
   unchanged `Salim`. Note: `0 Salim baseline`. Expected: vcv tuned HGB about
   0.419, ensemble about 0.408.
1. **Detection: W1 + W2 + W3.** Port `timmyo`'s `recurrence.py` and
   `category_map.py`, and close the interface gap. Target:
   - vcv ensemble about 0.47 or higher.
   - `c_detect` about 0.67.
   - If you land well below this, look for a porting bug before tuning
     anything.
2. **Live definition: W4.** Keep the change only if vcv improves by at least
   0.01, or improves by less while `c_nextdue` also rises.
3. **Rule: W5.** Port `timmyo`'s rule. Compare the rule with the best model
   on vcv, and let `predict.py` choose.
4. **Experiment: `Salim`'s `classify()` as the cluster voter.** Keep it only
   under the same rule as step 2.
5. **Submission.** Write `submission.csv` with `python -m src.predict`. Then
   update the README's reproduce section and its results table using the
   `experiments.csv` rows.

## Rules
- With 1,000 valid clients, differences below about 0.01–0.02 can be noise.
  Decide with vcv, not with a single valid split.
- **Do not add per-class threshold or probability reweighting.** It was
  negative under nested evaluation, between −0.005 and −0.03.
- Seed everything with `random_state=42`. Set `OMP_NUM_THREADS=1` if you run
  fits in parallel.
- Don't commit `__pycache__/`, the processed CSVs, or the unzipped data.
- When you finish, report a table with one row per step: valid and vcv
  macro-F1 for each model, plus `c_detect` and `c_nextdue`. Also report
  per-class F1 for the final model, and say which weaknesses you replaced and
  which you kept.
