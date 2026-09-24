# Prompt: consolidate the best of every branch into `Salim`

Paste everything below the line into Claude Code, started in a checkout of
this repo on branch `Salim`.

---

You are working in the `ubs-2026` repo on branch **`Salim`**. Read `README.md`
first: we predict each client's next recurring merchant family (`cloud, gym,
insurance, mobile, music, software, streaming, none`) in the 90 days after
cutoff `2026-01-01`, scored by **macro-F1** over the 8 labels.

Four team branches explored this in parallel. I benchmarked all of them with
one shared evaluator and tested whether their ideas stack. They do. Your job:
port the winning pieces into `Salim`, check that the numbers below reproduce,
ship a submission, then run the gated follow-up experiments. Work in the
phases below, in order, committing after each phase.

## 1. What each branch contributes (`git fetch origin` first)

| Branch | Take | Leave behind |
|---|---|---|
| `shipra1` | Nothing. It is the shared base commit `510d03a` plus committed `.pyc` files. | everything |
| `timmyo` | **Stream-detection rewrite** (its "exp 4") in `src/recurrence.py` + `src/category_map.py`: outgoing transactions only (refunds broke the gap stats); drop types `atm, fee, p2p_transfer`; drop non-subscription phrases (`NON_SUBSCRIPTION_PHRASES`) matched as **substrings on any MCC**; **pool each client's charges across all MCCs** (subscriptions hop MCCs: 20% of valid charges vs 6% of train); amount-cluster everything with tolerance `max(0.3, 0.03 × running mean)`. **Experiment harness** `src/evaluate.py` (rebuild → score → append to `experiments.csv`) with step-C diagnostics `c_detect / c_nextdue / c_none_active / c_fams`. Docs: `PIPELINE.md` (step vocabulary A–I, experiment log), `HANDOVER.md`. | committed `__pycache__/`, `py -3.10` invocations, its `data/*.jsonl` path layout |
| `shipra2` | **Tuned models** from `src/model.py`: `build_logreg_l1` (L1, saga, C=10), `build_logreg_l2` (C=3), `build_hgb_tuned` (`learning_rate=0.02, max_leaf_nodes=63, min_samples_leaf=5, l2_regularization=2.0, max_iter=300, max_depth=5`, class-balanced), `RANDOM_SEED=42` + `set_global_seed`. **Soft-vote ensemble** from `src/ensemble.py` (α·P(LR-L1) + (1−α)·P(HGB-tuned), with `get_proba` aligning columns to `ALL_LABELS`). **`INSIGHTS.md`**, the negative-result log. `.gitignore` entries. | label/submission CSVs committed under `data/`, `inspection.zip`, `uv.lock`, the `HARD_EXCLUDE_MCCS` toggle (tested negative), its old recency-sorted `rule_predict` |
| `Salim` (you are here) | `src/predict.py` (pick model, refit on train+valid, write the submission in `sample_submission.csv` order, assert labels). Features in `src/features.py`: **live vs lapsed** split (a stream is stale if silent for more than 2× its own cadence), `lapsed_<cat>`, `mean_gap_days_<cat>`, `days_to_next_<cat>`. **`src/personas.py`** behavioural persona scores (scorer fit on train only, reused for valid/test). `rule_predict` sorted by `days_to_next`. README reproduce section; Python ≥3.11 and relaxed pins. | **`CADENCE_BANDS` beyond monthly** (bimonthly, quarterly, semiannual, annual). Measured negative, see §2. |

## 2. Benchmark: same evaluator for every branch, macro-F1 over 8 labels

Evaluation protocols:
- **valid**: fit on train (2,000 clients), score valid (1,000). This is what every branch reported.
- **vcv**: use this one to make decisions. Split the *valid* clients into 5 stratified folds (`StratifiedKFold(5, shuffle=True, random_state=0)`). For each fold, fit on all of train plus the other 4/5 of valid, predict the held-out 1/5, then compute macro-F1 once over all 1,000 pooled valid predictions. This mirrors the final refit on train+valid, and it scores only valid clients, which timmyo showed are noisier than train and look like test (median gap_cv 0.11 train, 0.29 valid, 0.33 test). Plain k-fold over train+valid is optimistic.

Model definitions: all models use `class_weight="balanced"` and `random_state=42`. `lr_l2` = median impute → StandardScaler → `LogisticRegression(C=1, max_iter=3000)`. `lr_l1` = the same with `penalty="l1", solver="saga", C=10, max_iter=5000`. `hgb_def` = default HGB. `hgb_tuned` = shipra2's parameters. `ens` = 0.5·lr_l1 + 0.5·hgb_tuned probabilities.

| Feature set | valid lr_l2 | valid lr_l1 | valid hgb_def | valid hgb_tuned | valid ens | **vcv lr_l2** | **vcv lr_l1** | **vcv hgb_def** | **vcv hgb_tuned** | **vcv ens** |
|---|---|---|---|---|---|---|---|---|---|---|
| base / `shipra1` | .384 | .383 | .359 | .382 | .403 | .394 | .388 | .415 | .414 | .419 |
| `shipra2` | .381 | .398 | .348 | .396 | .416 | .386 | .389 | .413 | .411 | .426 |
| `Salim` (current) | .366 | .365 | .387 | .402 | .393 | .372 | .369 | .417 | .419 | .408 |
| `timmyo` | .424 | .422 | .443 | .473 | .454 | .431 | .432 | .460 | .467 | .469 |
| C: timmyo detection + Salim features, no personas | .427 | .434 | .432 | .460 | .462 | .442 | .443 | .452 | .470 | .470 |
| **A: timmyo detection + Salim features + personas (target)** | .440 | .444 | .431 | .461 | **.476** | **.450** | .445 | .460 | **.481** | **.475** |
| B: A + Salim's quarterly/annual cadence bands | .408 | .402 | .452 | .466 | .447 | .427 | .429 | .445 | .481 | .471 |

Combo A is `Salim`'s `src/` with `src/recurrence.py` and `src/category_map.py`
replaced by timmyo's versions. That is the whole change. Salim's `features.py`
runs unchanged on top of it.

What the table shows:
1. timmyo's detection is the biggest single win: +0.04 to +0.06 vcv for every model type.
2. Salim's features stack on top of it: combo A is the best overall.
3. Salim's longer cadence bands hurt once detection pools across MCCs (LR −0.02 vcv, valid ens −0.03). Keep **monthly only (20–45 days)**.
4. Personas are a small but consistent plus (+0.005 to +0.01). Keep them, and re-check once the other changes are in.
5. Tuned HGB beats default HGB on the new features. Ensemble vs tuned HGB alone is within noise, so choose by vcv.
6. Per-class probability reweighting / threshold tuning for macro-F1 is **negative** under nested evaluation (−0.005 to −0.03). Don't add it.

Noise: with 1,000 valid clients, differences under ~0.01–0.02 can be noise.
Keep a change only if vcv improves by ≥ +0.01, or if it improves by less but
valid and the step-C diagnostics also improve.

Weak spots of combo A (vcv ensemble, per-class F1): **music 0.32** (only 24 of
93 recalled), streaming 0.36, software 0.42, cloud 0.51, gym 0.55,
insurance 0.49, mobile 0.52, none 0.63.

## 3. Phases

### Phase 0: housekeeping (no behaviour change)
- Start from a clean `git status`. **Do not `git merge` the other branches.** They carry a different `data/dataset.zip`, committed `.pyc` and CSV clutter, and conflicting versions of the same `src/` files. Port with `git checkout origin/<branch> -- <path>` or merge by hand, and name the source branch and author in each commit message.
- Keep Salim's data layout (`unzip data/dataset.zip -d data/dataset` gives `data/dataset/dataset/*.jsonl`). Put the data directory and cutoff in **one** place (for example `src/config.py`) and make every entry point use it. timmyo's `evaluate.py` currently reads `data/*.jsonl`.
- Add `encoding="utf-8"` to every `open()` call (this is a Windows machine).

### Phase 1: one evaluation harness, then a baseline
Adapt timmyo's `src/evaluate.py`:
- `python -m src.evaluate "<note>" [--submit]` rebuilds train/valid/test features from the raw jsonl. It fits the persona scorer on train only and reuses it for valid and test, then caches the feature CSVs in `data/processed/`.
- Score the model zoo from §2 (`lr_l2`, `lr_l1`, `hgb_def`, `hgb_tuned`, `ens`) under **both** valid and vcv. Also print step-C diagnostics, per-class F1, the confusion matrix of the best model, and predicted vs true label distribution.
- Append one row per run to `experiments.csv`: timestamp, note, git short SHA, every score, every diagnostic.
- Seed everything with `RANDOM_SEED=42`. If you ever run fits in parallel processes, set `OMP_NUM_THREADS=1`: HGB's OpenMP threads oversubscribe the CPU and made my benchmark ~10× slower.
- Run it on the unchanged Salim code: `0 Salim baseline`. Expect about vcv hgb_tuned 0.419, ens 0.408.

### Phase 2: port timmyo's stream detection (expect the big jump)
- Take timmyo's `src/category_map.py` and `src/recurrence.py` as they are: monthly band 20–45 days, `gap_cv ≤ 0.5`, `amount_cv ≤ 0.35`, tolerance 0.3 / 0.03. This deliberately removes Salim's `CADENCE_BANDS`.
- Leave `features.py` alone. It already consumes `mean_gap_days` from the stream table.
- **Acceptance check:** vcv hgb_tuned ≈ 0.48, vcv ens ≈ 0.475, valid ens ≈ 0.476 (±0.01 for library versions), valid `c_detect` ≈ 0.67. With Salim's own `python -m src.model` (default logreg/HGB, fit train → score valid) combo A gives exactly **logreg 0.4403, hgb 0.4312** (re-verified). If you land far off, stop and diff your code against combo A before going further.
- **Ship immediately:** as soon as the acceptance check passes, run `python -m src.predict` and commit + push `submission.csv`. It already beats every branch's current submission, and milestone deadlines are fixed; don't make it wait on Phase 3.

### Phase 3: models (from shipra2)
- Move `build_logreg_l1`, `build_logreg_l2`, `build_hgb_tuned` and `RANDOM_SEED`/`set_global_seed` into `src/model.py`. Keep Salim's `rule_predict`, which sorts by `days_to_next`.
- Extend `src/predict.py` to `--model auto|logreg|logreg_l1|hgb|hgb_tuned|ensemble`. `auto` must select by **vcv**, not by the single valid split.
- Re-tune. shipra2 tuned the HGB parameters and α on the single valid split, on the *old* features. Re-run a ~40-config random search over shipra2's grid (`src/hgb_tune.py` on `origin/shipra2`) and an α sweep over {0.3, …, 0.7}, both scored by vcv. Adopt new values only if they beat shipra2's by ≥ +0.01 vcv. Also try averaging HGB over 3–5 seeds; it is cheap and more stable.

### Phase 4: ship the tuned submission (before any optional experiment)
- Run `python -m src.predict --model <winner>` to write `submission.csv`. Assert in code: header is exactly `client_id,predicted_next_recurring_merchant`; the client IDs are exactly those of `sample_submission.csv`, in the same order, one row each; only the 8 allowed labels appear.
- Print the prediction distribution. Sanity check: `none` should be about 25–30%, and every class should appear.
- Commit and push to `Salim` now (`git push -u origin Salim`), so a milestone-ready file always exists.

### Phase 5: gated experiments (one at a time, each logged, kept only per the rule in §2)
Listed in order of expected value:
1. **Uncategorised recurring streams (targets music, the weakest class).** After Phase 2, 11.5% of valid recurring streams have no category. Their descriptions are generic ("service payment", "merchant charge", "card purchase", "digital plus", "premium plan", "member plan"), and the targets of the clients who own them skew toward music. Median stream amount separates the families: cloud ≈ 8, music ≈ 13, streaming ≈ 16, software ≈ 29, mobile ≈ 44, gym ≈ 62, insurance ≈ 117. Try (a) features for each client's soonest-due uncategorised stream (amount, `days_to_next`, mcc), then (b) imputing a stream's category with a small classifier or kNN trained on categorised streams (amount, mcc, day-of-month, description tokens), used only above a confidence threshold.
2. **Fix D4 `recent_txns_<cat>`.** It currently counts every tagged transaction in the last 90 days, including decoys and transactions already inside a live stream (timmyo verified this). Variant: count only outgoing, non-excluded transactions that are not in any recurring stream, and add days since the last such transaction.
3. **Robust cadence instead of new bands.** After exp 4, the largest remaining C4 miss on train is mean gap < 20 days (82 missed targets). Look at those cases first: timmyo notes some streams bill every ~15 days, but in valid most rejected gaps come from extra off-cycle charges and skipped months. Try the median gap instead of the mean, or collapse charges ≤ 5 days apart before computing gap stats. Don't widen the bands; that was tested 7 ways and all were negative.
4. **Pseudo-labels from `unlabeled_pretrain_transactions.jsonl` (10k clients).** Move the cutoff back to 2025-10-01. Label each client with the family of the first recurring-stream charge in the following 90 days, else `none`. Train on these rows plus the real labels, down-weighting the pseudo rows. Validate the labeller first: applied to train/valid clients at the shifted cutoff, its label mix should resemble the real one (~30% `none`, ~10% per family). This has the highest ceiling and the highest risk; keep it only if vcv improves.

## 4. Do not retry (already measured negative)
- Weekly / biweekly / 14–45 / 14–90-day cadence bands, at any amount tolerance (shipra2, 6 variants: −0.017 to −0.039).
- Bimonthly, quarterly, semiannual and annual bands on top of pooled detection (my combo B above).
- MCC hard exclusion vetoing keyword matches on groceries, transport, etc. (shipra2: −0.016).
- Cross-category rank/share features over `recent_txns_*` (shipra2: −0.023).
- Two-stage model: "adopt anything?" then "which family?" (shipra2: below single-stage).
- Bank-holiday calendar `data/bank_transfer_closed_days.csv` (timmyo: transactions ignore the calendar).
- Amount-cluster tolerance 0.1 / 0.01 or tighter (splits real streams).
- Per-class probability reweighting / threshold tuning for macro-F1 (nested: negative).

## 5. Guardrails
- Features may use only transactions on or before the cutoff. The persona scorer and any other fitted transform are fit on training rows only.
- Make one change per experiment. For each, state the step code from `PIPELINE.md` (for example "C4", "D2"), the hypothesis, and the before/after vcv.
- Never tune on the single valid split alone; decide on vcv.
- Never commit extracted data, `data/processed/*.csv`, `__pycache__/`, `.pyc` or `.DS_Store`. Merge the `.gitignore` entries from shipra2 and timmyo into Salim's.
- Keep `requires-python >=3.11`, `numpy>=1.26`, `pandas>=2.2`, `scikit-learn>=1.5`.
- Commit after each phase with a descriptive message, and push to `Salim` only.

## 6. Docs (last)
- Consolidate into one `INSIGHTS.md`: shipra2's findings and negative results, timmyo's A–I step vocabulary and experiment log, the §2 benchmark, every Phase 5 result (kept or not), and a short model card for the final pipeline.
- Update README's "Reproducing the Submission" to the final command sequence: unzip → `python -m src.evaluate "<note>"` → `python -m src.predict`.
- Fold the still-useful parts of `HANDOVER.md` into `INSIGHTS.md` rather than keeping three overlapping docs.

## 7. Report back
- A table with one row per experiment: note, valid and vcv for each model, step-C diagnostics, kept or reverted.
- The final model, its vcv macro-F1 and per-class F1.
- The submission path, row count and label distribution.
- Anything that did not reproduce the numbers in §2, and why.
