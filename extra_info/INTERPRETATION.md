# Interpreting the Sparse Per-Label Logistic Regression

> **Note:** the numbers here come from the sparse per-label model on the `lean` set trained on train only (valid macro-F1 0.504). The final model additionally uses 3,152 pseudo-labelled clients (0.513); the direction of every effect is unchanged.

Model: benchmark 2, sparse per-label logistic regression on the `lean` feature set (101 features).
Valid macro-F1: **0.4945** (fit on train only).

How to read the numbers:
- **×k per SD**: one standard deviation more of a feature multiplies the odds of *that* label (vs all other labels) by k. ×1 = no effect, above 1 = more likely, below 1 = less likely.
- **Raw data**: label rates computed directly from the 2,000 training clients, without the model.
- **Status**: ✅ = re-computed independently and confirmed (`scratchpad/interp/verify.py`, 27 checks). Where a first draft was wrong or imprecise, the corrected statement is shown and marked ✏️.

---

## 1. How the model works

| # | Statement | Verified value | Status |
|---|---|---|---|
| 1.1 | There are 8 separate yes/no models ("is it gym?", …, "is it none?"). Each keeps only the predictors it needs. | music 23, cloud 53, insurance 53, mobile 54, gym 55, streaming 55, software 76, none 80 of 101 | ✅ |
| 1.2 | Each label picks its own penalty strength C by inner cross-validation on train. | music 0.03; cloud, gym, insurance, mobile, streaming 0.1; software, none 0.3 | ✅ |

---

## 2. The core story: holds for every label

> **What a client paid recently and what is still running decides what comes next.**

| # | Statement | Verified value | Status |
|---|---|---|---|
| 2.1 | **Recent payments** in the family (last 90 days) are the strongest driver: ×1.9 to ×2.8 per SD. | mobile ×2.76, cloud ×2.63, gym ×2.48, insurance ×2.35, streaming ×2.29, software ×2.21, music ×1.88 | ✅ |
| 2.2 | **Subscription age** raises the odds: ×1.8 for cloud, gym, insurance and mobile; ×3.1 for software; almost no effect for music and streaming. | cloud ×1.75, gym ×1.78, insurance ×1.76, mobile ×1.80, software ×3.11, music ×1.04, streaming ×1.07 | ✏️ corrected (first draft said "×1.8 for all") |
| 2.3 | **Still live** (next charge not overdue) raises the odds: ×1.3 to ×1.7 per SD. | music ×1.27, cloud ×1.40, gym ×1.40, mobile ×1.48, software ×1.48, insurance ×1.64, streaming ×1.68 | ✅ |
| 2.4 | **Due first** among the client's live subscriptions raises the odds. | gym ×1.14, insurance ×1.32, mobile ×1.38, music ×1.39, software ×1.41, streaming ×1.41, cloud ×1.65 | ✅ |
| 2.5 | Raw data: with many recent cloud payments, the cloud rate rises from about 1% to 30%. | 1.1% (none) → 30.1% (top quartile) | ✏️ precise numbers (first draft: 3% → 31%) |
| 2.6 | Raw data: the older the insurance subscription, the more often insurance comes next. | 19% → 24% → 33% → 41% (age quartiles) | ✅ |
| 2.7 | Stable: "recent payments" is chosen with a positive sign in every bootstrap resample, for every family. | 100% for all 7 families (20 resamples) | ✅ |

---

## 3. Stories per label

| # | Label | Statement | Verified value | Status |
|---|---|---|---|---|
| 3.1 | **Mobile** | The clearest signal of all: mobile is a habit. | recent mobile payments ×2.76 (highest of all families) | ✅ |
| 3.2 | **Gym** | Competition: clients with active insurance get gym next less often, because insurance is due first. | model: recent insurance payments ×0.61 for gym | ✅ |
| 3.3 | **Gym** (raw data) | The gym rate is lower when insurance was active recently. | 11.2% vs 6.4% | ✏️ precise numbers (first draft: 10.7% vs 5.8%) |
| 3.4 | **Music** (raw data) | If music is the next subscription due, music comes next in half the cases. | 51% vs 5% | ✅ |
| 3.5 | **Music** (raw data) | A live music subscription makes music about 9 times more likely. | 37% vs 4% | ✅ |
| 3.6 | **Streaming** | Clients with many different merchant types get streaming next less often. | model ×0.68 per SD; raw data 17% → 14% → 9% → 5% (quartiles) | ✅ |

---

## 4. Checklist for "none": nothing recurs in the next 90 days

| # | Signal | None rate (raw data) | Status |
|---|---|---|---|
| 4.1 | Longest live subscription is a **short trial** (only 3–4 charges) | 67% vs 27% | ✅ |
| 4.2 | The **rule says none** (no live subscription, or only a trial) | 59% vs 20% | ✅ |
| 4.3 | **No live subscription** at all (vs 2 or more) | 56% vs 18% | ✏️ precise numbers (first draft: 38% vs 13%) |
| 4.4 | **Fewer card payments**: more card use lowers the odds of none | model ×0.53 per SD of card-payment share | ✅ |

---

## 5. Do not use in the pitch (misleading signs)

| # | Statement | Verified value | Status |
|---|---|---|---|
| 5.1 | **Number of payments** (`n_occurrences_*`) and **subscription age** (`tenure_days_*`) move together, but the model gives them opposite signs. Only tell the "age" story. | cloud r=0.85, gym 0.87, insurance 0.87, mobile 0.89, software 0.86; always − for payments, + for age | ✅ |
| 5.2 | Proof that the negative sign misleads: more software payments actually *raise* the software rate in the raw data. | 9% → 15% → 39% → 29% (quartiles) | ✅ |
| 5.3 | The same "twin" problem affects `active_*` / `over_days_*` vs `is_live_*`, and `port_families_ever` vs `port_families_last90d`. Use only `is_live_*`. | from the skeptic's collinearity check (46 opposite-sign pairs with r > 0.7) | ✅ |
| 5.4 | **Software** also leans on *other* families' recent payments. That reflects correlated spending, not a software signal, so it needs a footnote. | recent streaming ×0.45, recent gym ×0.57 | ✅ |
| 5.5 | **Insurance refunds** are inconsistent: an insurance refund looks positive while refunds overall look negative. Too few cases, so leave it out. | refund_insurance_90d ×1.33, refund_n_90d ×0.68 | ✅ |

---

## 6. One-slide summary

1. **Recent payments** are the strongest reason a family comes next (×1.9–2.8).
2. **Still live and due first** add to it (×1.3–1.7).
3. **Older subscriptions** renew more reliably, especially software (×3.1) and cloud, gym, insurance and mobile (×1.8).
4. **Families compete**: whatever is due first wins (for example insurance before gym).
5. **"None"** = trials that end, subscriptions that stopped, or no live subscription at all.

Verification: 27 statements re-computed from the training data and the fitted model. 23 were confirmed exactly, 4 were corrected to the precise numbers or wording (✏️). None were disproved in direction.
