# Insights — Transaction Activity Forecasting

A full record of the approach, data findings, and modeling iterations for predicting
`target_next_recurring_merchant`, including the experiments that **failed** — kept
here deliberately so they aren't retried.

Code lives in `src/`. Final model: **ensemble of `logreg_l1` + `hgb_tuned`**, macro-F1
**0.4163** on `valid_features.csv`.

---

## 1. The task

Predict, per client, which merchant family (`cloud, gym, insurance, mobile, music,
software, streaming`, or `none`) will recur within 90 days after the cutoff date
(`2026-01-01`), using only transaction history up to that date. Scored on macro-F1
(equal weight per class).

---

## 2. Data exploration findings

### MCC mapping (`data/mcc_frequency.json`, `src/category_map.py`)
12 unique MCCs in the transaction data. Mapped to the 7 target categories primarily
by **description keyword**, with MCC as a fallback — keyword-first turned out to be
more reliable than MCC alone:

| Category | Primary MCC | Notes |
|---|---|---|
| insurance | 6300 | clean |
| gym | 7997 | clean |
| software | 5734 | clean |
| mobile | 4814 | clean |
| cloud | 5732 | **overloaded**: shares MCC with real electronics/marketplace spend |
| streaming / music | 5812 | **overloaded**: shares MCC with real dining; streaming and music aren't separable by MCC at all, only by description (`audio` vs `video`/`stream` tokens) |

Two important data quirks found by manual inspection:
- **MCC cross-contamination**: ~8–10% of keyword-matched transactions (e.g. `"gym"`/`"fit"`) carry an MCC other than the expected one. Gating category purely by MCC would silently drop real signal.
- **Rotating descriptions**: a single real recurring subscription can alternate its description text cycle to cycle (e.g. one client's charge rotates through `"digital plus"` → `"premium plan"` → `"media streaming"` → `"video access"` at a near-constant amount). Grouping by literal description text splits what is really one stream.

### `type == "topup"`
Exclusively salary deposits in this dataset (`description="salary"`, `direction="in"`, ~$900–9,000). Used as the income-regularity signal.

---

## 3. Recurring-stream detection (`src/recurrence.py`)

Two real bugs were found and fixed during development — both worth remembering:

1. **Keyword priority bug**: `"audio streaming"` contains both `audio` and `streaming` tokens; checking `streaming` before `music` in the classifier caused every `music` transaction to be absorbed into `streaming` (music detections dropped to zero at one point). Fixed by checking `music` first.
2. **`None` vs `NaN` bug**: pandas silently upcasts `None` → `NaN` (float) in an object column. A `c is not None` filter doesn't catch `NaN`, corrupting majority-vote category assignment. Fixed with `pd.notna()`.

**Design**: transactions are grouped into "streams" by `(client_id, mcc)`. For the two
overloaded MCCs (`5812`, `5732`) only, streams are further split by **1D amount
clustering** rather than literal description matching — this correctly merges
rotating-description cycles of the same real subscription while separating genuinely
different subscriptions at different price points. Known non-subscription literals
(`"coffee shop"`, `"electronics shop"`, etc.) are excluded from the clustering pool
first.

Applying amount-clustering to *all* MCCs (not just the two overloaded ones) was tried
and reverted — it fragmented noisy MCCs (pharmacy, hotel, ATM, ride-share) into tiny
clusters that satisfied the recurring test by pure chance, producing false positives.

A stream counts as **recurring** if it has ≥2 occurrences, mean gap 20–45 days, gap
coefficient-of-variation ≤0.5, and amount CV ≤0.35. Its category is assigned by
majority vote across its own transactions' keyword hits (handles the rotating-text
problem).

### The 20–45 day monthly-only gap band is deliberate, not an oversight
A client can have a very clean, obviously-real recurring subscription that the
detector still misses because its cadence isn't monthly — e.g. client `C000002`'s
software subscription bills every ~13 days (`amount_cv=0.06`, extremely stable; but
`mean_gap=12.8`, below the 20-day floor). **Six variants** were tried to also admit
weekly/biweekly billing, all measured against the `logreg_l1` baseline (macro-F1
0.3982) and all made it worse:

| Variant | macro-F1 |
|---|---|
| Baseline: single monthly band (20–45 days), `amount_cv≤0.35` | **0.3982** |
| Contiguous 14–45 day window | 0.3794 |
| Contiguous 14–90 day window | 0.3594 |
| Disjoint weekly(5–9)/biweekly(10–18)/monthly bands, uniform loose `gap_cv≤0.55` | 0.3673 |
| Same disjoint bands, moderate per-band `amount_cv` (0.10/0.15) | 0.3618 |
| Same disjoint bands, very tight per-band `amount_cv` (0.02/0.05) | 0.3783 |
| Monthly-only band, just `gap_cv` raised 0.5→0.55 (isolating that one change) | 0.3817 |

Even a near-exact-amount-match requirement (`0.02` CV, essentially demanding the same
few cents every time) for the weekly/biweekly bands still net-hurt. The likely
mechanism: a short gap band has far more chances to match two coincidentally-similar,
unrelated transactions across a ~14-month history than the 20–45 day monthly band
does, and that false-positive noise outweighs whatever genuine weekly/biweekly signal
exists — which appears to be rare in this dataset. **Conclusion: keep the single
monthly band.** `C000002`-style cases remain a known, accepted gap.

### MCC "hard exclusion" for obviously non-subscription MCCs — also tested, also worse
Intuition: MCCs like `5411` (groceries), `4111` (transport), `5912` (pharmacy), `7011`
(hotel) should never contribute to a target category, regardless of keyword match —
`mcc_frequency.json` marks all of these `label_category: null`. Checked how many
transactions currently get a category anyway via the keyword-first design: **726**,
concentrated in `5411` (392, e.g. `"gym membership"`, `"audio streaming online"`
literally appearing under the groceries MCC). Tried hard-excluding these MCCs from
ever returning a category (`src/category_map.py`'s `HARD_EXCLUDE_MCCS`, now kept as an
empty-set toggle in the code): macro-F1 **0.3821**, worse than the 0.3982 baseline.
The sample descriptions on these "wrong" MCCs read as genuine subscription text
(matching the templated vocabulary seen everywhere else), not coincidental noise —
i.e. this looks like the MCC field itself is corrupted for these rows, not the
description. Trusting MCC as a veto here loses more recall than it gains precision, so
the keyword-first design (already documented above) stays as-is despite the
real-world intuition pointing the other way.

### Key modeling insight: adoption vs. continuation
Cross-checking each client's target label against whether that category was already
an active recurring stream: **only ~53% of non-`none` labels are continuations of an
existing subscription; ~47% are brand-new category adoptions** the client didn't have
before cutoff. Clients with a `none` target average *more* active categories (1.54)
than clients who adopt something new (1.16) — a "saturation" effect. This shaped the
whole feature design: the task is at least as much about predicting *new* behavior as
continuing existing behavior.

---

## 4. Feature engineering (`src/features.py`)

92 features per client, in four groups:

1. **General financial behavior** (18): tenure, recency, transaction frequency, income (topup count/amount/cadence), transaction-type mix, MCC diversity.
2. **Per-category recurring state** (8 × 7 categories): active flag, occurrence count, recency, tenure-in-category, mean amount, cadence regularity (`gap_cv`), and **periodicity** (`mean_gap_days`, `days_until_next_expected` — see §5.4).
3. **Adoption dynamics** (4): `n_active_categories`, `n_missing_categories`, `days_since_last_new_category`, `adoption_rate_per_year`.
4. **Early-adoption signal** (7): count of category-matching transactions in the last 90 days that *haven't yet* reached the ≥2-occurrence recurring bar. Directly targets the "new adoption" half of the problem — a client's first-ever payment in a category is invisible to the strict recurring detector but is exactly the leading indicator needed for 90-day-ahead prediction. In the coefficient analysis (§6), this was the single strongest positive driver of predicting each category, for every category.

Missing values are left as true `NaN` (not 0) when a category isn't active — 0 would
falsely imply "zero amount charged," whereas `NaN` correctly means "not applicable."

---

## 5. Feature experiments: what worked and what didn't

All measured as change in macro-F1 vs. the standing baseline at the time, on
`valid_features.csv`, model trained on `train_features.csv` only.

| # | Experiment | Result | Kept? |
|---|---|---|---|
| 5.1 | Original 79-feature baseline | LR 0.3843, HGB 0.3591 | baseline |
| 5.2 | + near-recurring clusters + exact last-seen-per-category | LR **+0.0097** (0.3940); HGB unchanged | **Reverted** (user chose to try cross-category ranking instead) |
| 5.3 | + cross-category ranking (rank/share of `recent_txns_*` across categories) | LR **-0.023** (0.3612); HGB **-0.004** | **Reverted** — redundant derived columns increased multicollinearity rather than resolving it |
| 5.4 | + periodicity (`mean_gap_days_{cat}`, `days_until_next_expected_{cat}`) | LR ≈flat (0.3808, -0.0035); HGB **-0.0077** | **Kept** — drop judged not significant enough to revert; became the standing 92/93-column feature set |
| 5.5 | Two-stage model (binary "adopt anything?" → conditional "which category?") | Best-tuned threshold 0.3684, still below single-stage 0.3808 | **Reverted** — stage 2 loses training signal (never sees `none` rows), no cross-task sharing the way a joint softmax gets for free |

### Why 5.2 helped but 5.3 and 5.5 didn't
A diagnostic (splitting new-adoption cases into "has a recent-signal precursor" vs.
"zero prior evidence") found: 77% of new-adoption cases have *some* signal (accuracy
35.8% on those), 23% have none at all (accuracy **3.9%**, worse than random — a true
ceiling, unfixable from transaction history alone). Feature 5.2 filled a genuine blind
spot (near-miss cadences the strict `is_recurring` gate was discarding). Features 5.3
and 5.5 instead just restructured or duplicated information the model already had,
which cost more (via regularization noise / lost training rows) than it added.

---

## 6. Validation deep-dive (`src/analyze.py`, `src/failure_analysis.py`)

**Reproducibility**: a single `RANDOM_SEED = 42` (`src/model.py`) is applied to every
model and to `numpy`/`random` global state. Verified — two independent full runs
produce bit-identical macro-F1.

**Coefficient inspection** (multinomial LR): `recent_txns_{category}` is the top
positive driver for every one of the 7 classes — validates the early-adoption feature
design. Also surfaced real multicollinearity: several cross-category coefficients
have non-trivial magnitude (e.g. `n_occurrences_gym` pushing *toward* a `cloud`
prediction), which motivated the L1 experiment (§7) and the (unsuccessful)
cross-category ranking experiment (§5.3).

**Segment accuracy** — the clearest read on where the model struggles:

| Segment | logreg_l1 | hgb_tuned |
|---|---|---|
| Continuation (already active) | 0.602 | 0.517 |
| New adoption (brand-new category) | 0.298 | 0.312 |
| True `none` | 0.468 | 0.509 |

**The two models fail in opposite directions**: LR over-predicts specific categories
when the truth is `none` (top errors: `none→gym`, `none→software`, `none→insurance`);
HGB under-predicts, defaulting to `none` too often when the truth is a real category
(top errors: `music→none`, `gym→none`, `mobile→none`). LR predicts `none` for 257/1000
clients (precision 0.53, recall 0.47); HGB predicts it for 316/1000 (precision 0.47,
recall 0.51) — same trade-off, opposite direction.

**Cross-model error overlap** — the finding that motivated the ensemble: the two
models agree on only **53.4%** of predictions. 27.6% both correct, 42.6% both wrong,
15.1% only LR correct, 14.7% only HGB correct — balanced, complementary mistakes, not
one model being a strictly worse version of the other.

---

## 7. Hyperparameter tuning (`src/l1_logreg.py`, `src/hgb_tune.py`)

**Logistic regression**: swept `C` under both L1 (`solver="saga"`) and L2. Best: **L1,
C=10** → macro-F1 **0.3982** (89% of coefficients still nonzero — barely sparse). L2
at C=3.0 scored **0.3963**, essentially tied. The real lever was **loosening
regularization strength** (the untuned default `C=1.0` was over-regularizing this
feature set) — L1's sparsity per se contributed only a marginal, within-noise edge
over a comparably-tuned L2. Both kept as options (`build_logreg_l1`,
`build_logreg_l2`) since the choice is close enough to depend on what's built on top.

**HistGradientBoosting**: 40-config random search over `learning_rate`,
`max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`, `max_iter`, `max_depth`.
Default HGB (0.3514–0.3591 depending on feature set) had never been given a fair
comparison against tuned LR. Tuned HGB (`learning_rate=0.02, max_leaf_nodes=63,
min_samples_leaf=5, l2_regularization=2.0, max_iter=300, max_depth=5`) reached
**0.3962** — essentially tied with tuned LR. This also answered a side question:
XGBoost was considered but judged not worth adding, since it's architecturally the
same family as HGB and tuned HGB already lands in the same band as tuned LR.

---

## 8. Ensemble (`src/ensemble.py`) — final model

Soft-voting: average `logreg_l1` and `hgb_tuned` predicted probabilities, weighted by
α (weight on logreg_l1), argmax the result. Swept α from 0.0 to 1.0 on valid:

| α (logreg_l1 weight) | macro-F1 |
|---|---|
| 0.0 (pure hgb_tuned) | 0.3962 |
| 0.3–0.7 | 0.406–0.416 (all beat both individual models) |
| **0.5 (equal weight)** | **0.4163** ← best |
| 1.0 (pure logreg_l1) | 0.3982 |

Stable peak at α=0.5, not a lucky spike — confirms the failure-analysis hypothesis
that low agreement (53.4%) + balanced complementary errors is exactly the condition
under which probability-averaging helps. Per-class gains are broad: `cloud` f1 rose
from ~0.46 to 0.55, `insurance` to 0.46, `streaming` to 0.33 (up from 0.24 alone).

**This is the final model.**

---

## 9. Results summary

| Model | macro-F1 (valid) |
|---|---|
| Majority baseline (`none`) | 0.0567 |
| Rule-based (features, no learning) | 0.3367 |
| Logistic Regression, untuned (L2, C=1.0) | 0.3843 |
| HistGradientBoosting, untuned | 0.3591 |
| Logistic Regression, L2 tuned (C=3.0) | 0.3963 |
| Logistic Regression, L1 tuned (C=10.0) | 0.3982 |
| HistGradientBoosting, tuned | 0.3962 |
| **Ensemble (logreg_l1 + hgb_tuned, α=0.5)** | **0.4163** |

~7.3× improvement over the majority-class floor.

### Submissions generated (train+valid combined refit → predict on `test_transactions.jsonl`)
- `data/submission_logreg_l1.csv`
- `data/submission_hgb_tuned.csv`
- `data/submission_ensemble.csv` — **recommended**, best validated model

All validated against the README's Submission Contract (correct columns, exact
client set/order from `sample_submission.csv`, labels within the allowed set).

---

## 10. Known limitations / what's still unsolved

- **New-adoption accuracy remains the main weak point** (0.30–0.31 vs. 0.52–0.60 for
  continuation), and ~23% of new-adoption cases have zero prior transaction evidence
  — a hard ceiling no amount of feature engineering on this data can close.
- `music` and `streaming` remain the hardest individual classes — both share MCC
  `5812` and near-identical subscription pricing/cadence patterns in this synthetic
  dataset, so the separating signal is genuinely thin.
- `unlabeled_pretrain_transactions.jsonl` (149K rows, no labels) was never used —
  optional per the README, considered but deprioritized as the most speculative-payoff
  option relative to the supervised-data work above.

---

## 11. Code map

| File | Purpose |
|---|---|
| `src/category_map.py` | MCC/keyword → target category classifier |
| `src/recurrence.py` | Recurring-stream detection (amount clustering, cadence/amount thresholds) |
| `src/features.py` | Client-level feature table builder |
| `src/model.py` | Model definitions (`build_logreg_l1/l2`, `build_hgb`/`build_hgb_tuned`), baseline comparison script, `RANDOM_SEED` |
| `src/analyze.py` | Coefficient inspection, segment accuracy, confusion pairs (single model) |
| `src/l1_logreg.py` | L1 vs L2, `C` sweep |
| `src/hgb_tune.py` | HGB hyperparameter random search |
| `src/two_stage.py` | Two-stage model experiment (negative result, kept for reference) |
| `src/failure_analysis.py` | Confusion matrices + segment breakdown + cross-model error overlap for both final models |
| `src/ensemble.py` | Soft-voting ensemble, α sweep, final submission generation |

### Diagnostic CSVs (in `data/processed/`, generated ad hoc for manual review)
- `music_target_descriptions.csv` — all unique descriptions across clients with `target=music`, frequency-ranked. Confirms `music` detection (keyword `"audio"` only) is lossy: filler descriptions like `"premium plan"`/`"digital plus"` outnumber `"audio streaming"` among these clients.
- `valid_new_adoption_clients.csv` / `valid_new_adoption_clients_full.csv` — the 446 valid-set clients in the "new adoption" segment (target category not already active), with and without the full 92-feature row.
- `unresolved_recurring_5812_transactions.csv` — raw transactions behind the 293 streams that pass the recurring cadence test on MCC 5812 but never got a keyword hit (category stays unresolved) — for manually auditing whether the amount-clustering + majority-vote design is behaving as intended.
