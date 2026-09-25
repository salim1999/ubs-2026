# Pipeline Reference (step names for discussing changes)

This is the shared vocabulary for improving the model **one step at a time**.
Say "let's work on **C4**" or "change **D4**" and everyone knows which part of
the code is meant. Challenge spec: [README.md](README.md). Team context:
[HANDOVER.md](HANDOVER.md).

Items marked *(verified)* were checked against the code on 2026-09-24.

## Goal

Predict each client's **next recurring merchant family** within the 90-day
window after the cutoff (`2026-01-01`). Exactly one of 8 classes:

`cloud, gym, insurance, mobile, music, software, streaming, none`

Metric: **macro-F1** (every class weighs equally, including `none`).

### Four different kinds of "label" — don't confuse them

| Term | Level | Meaning |
|---|---|---|
| **Transaction category** | one transaction | Intermediate tag from step B |
| **Stream category** | one recurring stream | Majority vote of tags inside the stream (C5) |
| **Target label** | one client | The true answer (`target_next_recurring_merchant`) |
| **Prediction** | one client | The model's guess for the target |

The target label is never an input feature.

---

## Step overview

```
RAW TRANSACTIONS
  ─▶ A  Transaction preprocessing              src/recurrence.py  load_transactions
  ─▶ B  Transaction category mapping           src/category_map.py
  ─▶ C  Recurring-stream detection             src/recurrence.py  detect_streams
  ─▶ D  Client feature engineering             src/features.py
        D1 general · D2 per-category · D3 adoption · D4 early adoption
  ─▶ E  Training-table construction            (features + labels join)
  ─▶ F  Missing-value handling + scaling        src/model.py  build_logreg
  ─▶ G  Final multiclass classifier            src/model.py
  ─▶ H  Prediction / decision rule             models/rule_based/model.py
  ─▶ I  Validation + macro-F1 evaluation       src/model.py  evaluate
```

| Code | Name |
|---|---|
| **A** | Transaction preprocessing |
| **B** | Transaction category mapping |
| **C** | Recurring-stream detection (C1–C5) |
| **D** | Client feature engineering |
| **D1** | General financial features |
| **D2** | Existing recurring-category features |
| **D3** | Adoption dynamics |
| **D4** | Early-adoption features |
| **E** | Training-table construction |
| **F** | Missing-value handling and scaling |
| **G** | Final multiclass classifier |
| **H** | Prediction / decision rule |
| **I** | Validation and macro-F1 evaluation |

---

## A — Transaction preprocessing

Input fields per transaction: `client_id, timestamp, amount, currency,
direction, type, mcc, description, fee`. All history is before the cutoff.

Current behaviour *(verified)*: `load_transactions` reads the JSONL, parses
`timestamp`, and runs step B on each row. That's all. There is **no**
description normalisation, amount normalisation, or chronological sort at this
stage (sorting happens later where needed).

Output: a transaction-level table with an added `category` column.

## B — Transaction category mapping

`src/category_map.py`. **Not** the final prediction. Each transaction gets one
of the 7 families or `None` ("doesn't look like a subscription").

### B1. Keyword matching first *(verified)*

Descriptions are split into whitespace tokens. The first family with a
matching token wins, checked in this order:

| Order | Family | Tokens |
|---|---|---|
| 1 | music | `audio`, `pass` (since exp 8) |
| 2 | streaming | `stream`, `streaming`, `video` |
| 3 | software | `saas`, `software`, `productivity` |
| 4 | mobile | `phone` |
| 5 | cloud | `cloud`, `backup`, `storage` |
| 6 | gym | `gym`, `fit`, `fitness` |
| 7 | insurance | `insurance`, `cover`, `policy` |

Music comes before streaming so that "audio streaming" becomes music.
Matches are whole tokens only ("fitness" matches, "fitnessclub" doesn't).

### B2. Why keywords override MCC

MCC is contaminated. For example, about 8% of gym-keyword transactions are
not under 7997. Strong description evidence therefore beats MCC.

### B3. MCC fallback (clean MCCs only)

`4814 → mobile` · `5734 → software` · `6300 → insurance` · `7997 → gym`

### B4. Overloaded MCCs

`5812` (restaurants) mixes dining with music/streaming subscriptions. `5732`
(electronics) mixes shopping with cloud subscriptions. They have no MCC
fallback.

Non-subscription merchants are excluded from stream detection
(`NON_SUBSCRIPTION_PHRASES` / `_TYPES`, since exp 4). Matching is by substring
on **any** MCC, because these merchants also rotate prefixes/suffixes
("billing grocery store plus") and hop MCCs: `atm withdrawal, p2p send,
service fee, casual dining, coffee shop, electronics shop, pharmacy,
neighborhood market, ride share, hotel booking, fresh foods, online
marketplace, grocery store`, plus types `atm, fee, p2p_transfer`.

Known weakness: decoys (subscription words under the wrong MCC) are still
tagged, because keywords run first.

## C — Recurring-stream detection

`src/recurrence.py`. Answers the question: "is this sequence a real recurring
payment?" A single tagged transaction is not a stream.

### C1. Grouping
*(Since exp 4)* Outgoing transactions only, minus the B4 non-subscription
phrases/types, **pooled per client across all MCCs**. Reason: the same
subscription hops MCCs from month to month (e.g. a 21.5 streaming charge
alternating 5812/5411; a cloud charge on 5732/4814/5734). Grouping by
`client × MCC` split one stream into fragments that failed `n ≥ 2` or the
45-day gap rule. Before exp 4: `client × MCC`.

### C2. Amount clustering (all candidates)
Sort by amount and cluster greedily: start a new cluster when the next amount
is more than `max(0.3, 0.03 × running mean)` above the running mean. This
handles subscriptions whose description rotates ("Digital Plus" → "Premium
Plan" → "Media Streaming") but whose amount stays the same. The tolerance is
tight because pooling puts all of a client's streams in one pool (charges
vary about 1–2%). On train, 0.5/0.05 to 0.2/0.02 detect the same; 0.1/0.01
splits real streams. Before exp 4: `max(3.0, 0.25 × mean)`, 5812/5732 only.

### C3. Other MCCs
*(Obsolete since exp 4, everything is clustered.)* Previously the whole
client×MCC group was one candidate.

### C4. Recurrence rules *(verified)*
A candidate is recurring if **all** of these hold:

| Rule | Current value |
|---|---|
| occurrences | `n ≥ 2` |
| mean gap between transactions | `20 ≤ mean_gap ≤ 45` days |
| timing regularity | `gap_cv = std(gaps) / mean(gaps) ≤ 0.5` |
| amount stability | `amount_cv = std / mean ≤ 0.35` |

Known suspects: a skipped month pushes the mean gap above 45; a double charge
in one month pulls it down. After exp 4, the biggest remaining C4 failure on
train is `gap < 20` (82 missed targets; some streams bill every ~15 days).
Most other misses are families with < 2 charges or no stream at all (new
adoptions, D4 territory).

### C5. Stream category
Majority vote over the non-`None` transaction categories in the stream.
Streams with no keyword/MCC evidence at all stay `category = None` and are
unused downstream.

Output: one row per candidate stream with `n_occurrences, first_date,
last_date, mean_amount, amount_cv, mean_gap_days, gap_cv, is_recurring,
category`.

## D — Client feature engineering

`src/features.py`. Many transactions become one row per client.
*(Verified)*: **78 features** = 18 (D1) + 49 (D2) + 4 (D3) + 7 (D4).

### D1. General financial features (18)
`tenure_days, recency_days, n_txns, n_txns_per_month, total_out, total_in,
net_flow, avg_txn_amount, avg_out_amount, frac_card_payment, frac_p2p,
frac_atm, frac_fee, n_topups, mean_topup_amount, topup_gap_mean_days,
topup_gap_cv, n_distinct_mcc`

### D2. Existing recurring-category features (7 × 7 = 49)
Only streams with `is_recurring` and a category count. Streams of the same
family are aggregated per client. For each family `<cat>`:

| Feature | Definition |
|---|---|
| `active_<cat>` | 1 if ≥1 recurring stream of that family, else 0 |
| `n_streams_<cat>` | number of recurring streams in that family |
| `n_occurrences_<cat>` | total transactions across those streams |
| `recency_days_<cat>` | cutoff − latest `last_date` |
| `tenure_days_<cat>` | cutoff − earliest `first_date` (subscription age) |
| `mean_amount_<cat>` | mean of the streams' mean amounts |
| `gap_cv_<cat>` | mean of the streams' gap CV (lower = more regular) |

Missing handling: if a family isn't active, its numeric features are **NaN**
(not 0), because 0 recency would falsely mean "paid today". `active_*` is 0.

Not yet included: `mean_gap_days_<cat>` and `days_until_next_due_<cat>`
(draft in `git stash@{0}`).

### D3. Adoption dynamics (4)
`n_active_categories`, `n_missing_categories` (= 7 − active),
`days_since_last_new_category` (min of `tenure_days_<cat>`),
`adoption_rate_per_year` (= active categories / (tenure_days / 365)).

### D4. Early-adoption features (7)
`recent_txns_<cat>` = number of transactions tagged `<cat>` (step B) in the
last 90 days before the cutoff.

*(Verified)*: this counts **all** tagged recent transactions, including ones
that are already part of a recurring stream, and including decoys. The
"not yet recurring" filter is applied only in `rule_predict`
(`active_<cat> == 0`), not in the feature itself.

Motivation from earlier analysis: about 47% of non-`none` targets looked like
new categories. HANDOVER disputes this (about 80% of labelled families recur
in 3+ months of history), so the figure may reflect C missing real streams.

## E — Training-table construction

One row per client: `X` = the 78 features, `y` = `target_next_recurring_merchant`.
Currently read from the prebuilt `data/processed/{train,valid,test}_features.csv`,
where the label is already joined. Nothing in the repo regenerates those CSVs
with labels yet (draft: `src/evaluate.py`).

## F — Missing-value handling and scaling

Logistic regression only: `SimpleImputer(median)` fitted on train, then
`StandardScaler`. HGB takes NaN natively and gets no preprocessing.

## G — Final multiclass classifier

Best so far: `LogisticRegression(class_weight="balanced", max_iter=2000)`
(multinomial). Also tested: `HistGradientBoostingClassifier` (class-balanced),
the hand-written rule, and the majority class.

## H — Prediction / decision rule

Classifiers: argmax of the class probabilities, giving one of 8 labels.

`rule_predict` (no learning):
1. families with `recent_txns > 0` and not active → pick the one with the most recent transactions
2. else active families → pick the smallest `recency_days` (**bug**: that's
   the most recently paid, i.e. due *last*; fix in stash)
3. else `none`

**Example (Spotify → music):** "Spotify Premium" gets tag `music` (B). Several
monthly charges form a music stream (C). That gives `active_music=1,
recency_days_music=27, …` (D2). The model *may* predict `music`, but if there
is also a fresh single cloud payment (D4), the target could be `cloud`.
Historical category and target are related, but not the same thing.

## I — Validation and macro-F1 evaluation

Fit on train (2,000 clients), score on valid (1,000 clients) with macro-F1
over all 8 labels. Noise level with 1,000 clients: treat differences under
~0.01–0.02 as possibly noise.

Train is cleaner than valid/test (see the note under the experiment log), so
valid is the test-like split. CV understates gains from noise-robustness
changes (exp 4: valid +0.05/+0.09 vs CV +0.02/+0.01).

### Baseline (2026-09-24, prebuilt features, `python -m src.model`)

| Model | macro-F1 |
|---|---|
| majority (`none`) | 0.057 |
| rule | 0.337 |
| **logistic regression** | **0.384** |
| HGB (balanced via sample_weight, sklearn 1.0) | 0.350 |

Logistic regression F1 by class: cloud .43 · gym .49 · insurance .44 ·
mobile .35 · music .24 · software .40 · streaming .24 · none .49

Known weaknesses:
- **music vs streaming** (recall about 0.20 each): same MCC 5812, similar
  prices and timing, rotating descriptions.
- **`none`**: `none` clients have about 2.3 recurring streams too.

---

## How to propose a change

Always state:

1. which step changes (e.g. **C4**)
2. current behaviour
3. proposed behaviour
4. why it could improve macro-F1
5. implementation difficulty
6. leakage risk
7. how to test it on its own (score before vs. after on valid)

Change **one component at a time** and keep the rest as the baseline, so that
validation differences stay interpretable. Don't redesign the whole
architecture when only one step needs work.

## Experiment log

Scores: `LogReg / HGB / Rule` = macro-F1 on valid. `CV` = 5-fold on train+valid
(LogReg / HGB). `C det` = share of non-`none` valid clients whose target family
is an active stream. Run with `py -3.10 -m src.evaluate "<note>"`, raw rows in `experiments.csv`.

| # | Step | Change | LogReg | HGB | Rule | CV | C det | Keep? |
|---|---|---|---|---|---|---|---|---|
| 0 | – | Baseline (prebuilt features = regenerated, identical) | 0.384 | 0.350 | 0.337 | 0.408 / 0.435 | 0.369 | ✅ |
| 1 | C4/D2 | Bank calendar (`bank_transfer_closed_days.csv`): shift due dates off closed days | – | – | – | – | – | ❌ Stopped at gate: transactions ignore the calendar (closed-day share = base rate, flat weekdays, no dip on currency-specific holidays). Business-day gaps not more regular (CV 0.278 vs 0.275). New Year window: charges due on Dec 25/26/Jan 1/2 land on the due day as often as on open days (21% vs 19%), no shift. Branch `calendar-test`, `scripts/calendar_check.py` |
| 2 | D2 | + `mean_gap_days_<cat>`, `days_until_next_due_<cat>` (from stash) | 0.382 | 0.369 | 0.337 | 0.407 / 0.438 | 0.369 | ✅ HGB up, LogReg flat |
| 3 | C1 | Streams from `direction == "out"` only (refunds broke gaps) | 0.374 | 0.353 | 0.302 | 0.408 / **0.466** | **0.492** | ✅ kept: detection much better, valid F1 flat (within noise), CV ↑; see note |
| 4 | C1/C2 | Pool each client's charges **across MCCs** (was client × MCC), non-sub exclusion by phrase on all MCCs, amount tol 0.3/0.03 (was 3/0.25, 5812/5732 only). Tol chosen on train. `c_nextdue` 0.36 → 0.46. Rule drops because it still sorts by recency (step H fix next) | 0.424 | **0.441** | 0.229 | 0.428 / **0.479** | **0.673** | ✅ +0.05 / +0.09 on valid |
| 5 | G | Relaxed lasso: L1 LogReg selects 10/20/30/40/50 predictors, L2 LogReg refits (no tuning) | 0.393 / 0.412 / 0.413 / 0.421 / 0.426 | – | – | – | – | ❌ no gain over all 92 (0.424); `none` needs the many weak features. (Side note: L1 LogReg alone, C=0.3 by train CV, gave 0.447; not adopted) |
| 6 | D2 | Live-stream features: `over_days_<cat>`, `is_live_<cat>` (overdue ≤ 5 d), `n_live_streams`, `max_live_n_occurrences`, `short_live_stream` (3–4 charges), `min_over_days` | 0.448 | 0.456 | 0.229 | – | 0.673 | ✅ +0.024 / +0.015 |
| 7 | H | Rule: `none` if no live stream or short 3–4 charge stream, else live family due soonest | 0.448 | 0.456 | **0.472** | – | 0.673 | ✅ rule +0.24, best predictor |
| 8 | B1 | Token `pass` → music ("member pass" was music's untagged description) | **0.460** | 0.455 | **0.484** | – | 0.690 | ✅ LogReg/rule +0.012, HGB flat |
| 9 | C5 | Stream vote: keyword tags first, MCC fallback only if no keyword | 0.455 | 0.449 | 0.487 | – | 0.693 | ❌ dropped: models −0.005, rule +0.003 (noise), extra code |
| 10 | F | LogReg: absent per-family/stream features filled with 0 instead of median (flags carry absence) | 0.461 | – | – | – | – | ❌ +0.001 = noise, dropped |
| 11 | D (scratchpad) | Persona features on top of exp 8, scored with LogReg L2 / L1 (C=0.3): **rule's vote** (`first_due_<cat>`, `rule_says_none`) ✅ +0.007/+0.007; quitter ❌; newcomer ❌ (−0.008); **calendar** (`bill_day_<cat>`, `earliest_bill_day`) ✅ +0.005/+0.006; traveller ❌ | 0.472 | – | – | – | – | ✅ not yet in `src/` |
| 12 | D (scratchpad) | Shrink on top of 11: remove old timing (`recency_days_*`, `days_until_next_due_*`) and D3 adoption → 108 features. Kept (removal hurt): D1 general −0.011, stream stats, stream counts, D4 recent_txns −0.033 | **0.475** (L1 **0.486**) | – | – | – | – | ✅ not yet in `src/` |
| 13 | D | In `src/`: + rule's vote, + calendar, − old timing, − D3, − `mean_gap_days` → 101 features | 0.477 | 0.471 | 0.484 | – | 0.690 | ✅ |
| 14 | G | **Benchmark model 2**: sparse per-label L1 LogReg (one-vs-rest, own C per label by inner CV on train); `src/model.py` `build_sparse_logreg` | 0.477 (sparse **0.484**) | 0.471 | 0.484 | – | 0.690 | ✅ always benchmark both |
| 15 | G (scratchpad) | Personas/segments (rule splits, KMeans, logistic model tree), fixed 5–15 predictor scorecards, last-k payment / Markov features, per-last-family models; date-of-year n/a (shared cutoff) | ≤ 0.482 | – | – | – | – | ❌ none beat the benchmarks |
| 16 | D5 | `late_bro`: late_cycles_<fam> = over_days / mean gap, capped [−1, 3], 3 = no stream; late_min_cycles (8) | 0.477 (sparse 0.479) | – | 0.484 | – | – | ≈ flat |
| 17 | D5 | + `portfolio_bro`: families with ≥2 charges ever / last 90 d / dropped (3) | 0.478 (sparse 0.480) | – | 0.484 | – | – | ≈ flat |
| 18 | D5 | + `refund_bro`: tagged refund per family in last 90 d + count (8); all 19 story cols → 120 features | **0.489** (sparse **0.494**) | – | 0.484 | – | – | ✅ both models beat the rule; none F1 0.57 → 0.62 |
| 19 | D (scratchpad) | 31 ideas from the feature README as 7-col per-family blocks (timing clock, cadence, calendar, competition, client state, income, novelty, kNN priors) | – | – | – | – | – | ❌ none passed; mostly restate existing timing features |
| 20 | D6 | 10 client-level story features (due in 30/90 d, established, dormancy, timing margin, first-due cycle position / age, entropy, dominant share, cooling families) → `full` set 130; `lean` set 101 = −21 redundant cols, + cooling + dormancy only | full 0.474 / lean 0.484 (sparse 0.489 / **0.495**) | 0.485 | 0.484 | – | – | ✅ both sets kept; lean recommended |
| 21 | D7 | Traveller persona (`traveler_intensity`, `days_since_trip`) in full and lean | – | – | 0.484 | – | – | ✅ sparse lean 0.495 → 0.504 |
| 22 | D8 | **Final model**: sparse per-label lean + 3,152 pseudo-labelled pretrain clients (all 3 sources agree; Salim's merged labels). Looser filters / LLM labels: no gain or worse | 0.489 (lean) | – | 0.484 | – | – | ✅ **0.513** |
| 23 | G | Different model per label (L1, L2, HGB, RF, ET; chosen by train CV, pseudo rows as extra training data). All-HGB 0.530, best-per-label 0.528, calibration / meta-LogReg no help. HGB clearly better in train CV for gym, mobile, insurance, software; music prefers L1 | – | **0.530** (all-HGB) | – | – | – | ❌ kept L1 for interpretability (user decision); HGB = upside +0.017 |
| 24 | L2 | Due date = next billing day of the last charge (`days_to_bill_<fam>`, first_due ranked by it). Label-free check on history: picks the actual next charge 68-78% vs 64-67%. Rule 0.484 → 0.496, but final model 0.509 (−0.004), without the days_to_bill columns 0.512 | – | – | 0.496 | – | – | ❌ model already has the timing signal; reverted |
| 25 | L3 | is_live loosened: OR ≥ 2 tagged charges in 90 d over ≥ 2 months. Train-only 0.504 → 0.512 (old timing), with pseudo + lever 2: 0.514 | – | – | – | – | – | ❌ ±0.001 with pseudo data; reverted |
| 26 | H | Per-class log-prob biases for macro-F1 (Lipton et al. 2014), tuned by coordinate ascent on 5-fold train OOF (pseudo rows as extra training data). OOF 0.509 → 0.521, valid −0.009; no shrink factor beats no-bias on valid | – | – | – | – | – | ❌ does not transfer train → valid (distribution shift) |
| 27 | D | Discrete-time competing-risks hazard LogReg (daily charge hazard per live family, trained self-supervised on 6 historical cutoffs of train history; billing-day buckets dominate, OR 2.0 on the day, 0.08 at 8-14 days) -> p_first_<fam>, p_none90. Label-free first-charge hit 66.9% vs 66.6% day-of-month rule. Final model: + all 0.502, + p_none90 0.511, hazard-only rule 0.450 | – | – | 0.450 | – | – | ❌ timing already in the features |

**Note (2026-09-24): valid/test differ from train.** Detected families per
client: train 1.71, valid 1.33, test 1.19. Candidate subscription streams exist
just as often, but timing is noisier: median gap_cv train 0.11, valid 0.29,
test 0.33; share failing `gap_cv ≤ 0.5`: 7% / 20% / 24%. Rejected valid
streams are mostly monthly (most gaps 25–35 d) with **extra off-cycle
charges** (gaps 0–19 d) and **skipped months / pauses** (gaps 55–65, 75+).
So CV (2/3 train clients) is optimistic, and valid is the closer stand-in for test.

**Update after exp 4:** much of that timing noise was **MCC hopping**. Share of
keyword-tagged subscription charges *not* under their family's home MCC:
train 6%, valid 20.5%, test 19.3%. Under `client × MCC` grouping, a hop looks
like a skipped month in one group and an off-cycle charge in another.
Pooling across MCCs (exp 4) removes most of it: valid `c_detect` 0.49 → 0.67.
