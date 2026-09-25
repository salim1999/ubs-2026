# Final Model: Individual Coefficients

> **Note:** the numbers here come from the sparse per-label model on the `lean` set trained on train only (valid macro-F1 0.504). The final model additionally uses 3,152 pseudo-labelled clients (0.513); the direction of every effect is unchanged.

Final model: **sparse per-label logistic regression**, `lean` feature set (103 features), fit on train. Valid macro-F1 **0.5036**.

How to read: *odds ×k per SD* = one standard deviation more of the feature multiplies the odds of that label (vs all other labels) by k, holding the other features fixed. **Stable** = selected in ≥ 80% of 20 bootstrap refits with the same sign in ≥ 95%. Only stable coefficients are listed. Coefficients of twin features (`n_occurrences_*`, `active_*`, `over_days_*`, `late_cycles_*`) are left out because they move together with `tenure_days_*` / `is_live_*` and get offsetting signs. Full table: `(regenerate with src/model.py coef_table)`.

## cloud  (55 of 103 features used, 23 stable)

| Direction | Feature | Odds ×/SD | Sentence | Note |
|---|---|---|---|---|
| ↑ | `recent_txns_cloud` | 2.63 | More recent payments to cloud (last 90 days) increases the probability of cloud (odds x2.63 per SD). |  |
| ↑ | `tenure_days_cloud` | 1.76 | More age of the cloud subscription increases the probability of cloud (odds x1.76 per SD). |  |
| ↑ | `first_due_cloud` | 1.66 | More cloud being the next subscription due increases the probability of cloud (odds x1.66 per SD). |  |
| ↑ | `is_live_cloud` | 1.40 | More the cloud subscription still running increases the probability of cloud (odds x1.40 per SD). |  |
| ↑ | `frac_card_payment` | 1.23 | More share of card payments increases the probability of cloud (odds x1.23 per SD). |  |
| ↓ | `refund_n_90d` | 0.71 | More number of families with a recent refund decreases the probability of cloud (odds x0.71 per SD). | ⚠️ refund story failed the confounder check |
| ↓ | `port_families_ever` | 0.78 | More number of subscription families ever decreases the probability of cloud (odds x0.78 per SD). |  |
| ↓ | `refund_mobile_90d` | 0.78 | More a recent refund from mobile decreases the probability of cloud (odds x0.78 per SD). |  |
| ↓ | `n_distinct_mcc` | 0.84 | More number of different merchant types decreases the probability of cloud (odds x0.84 per SD). |  |

## gym  (58 of 103 features used, 21 stable)

| Direction | Feature | Odds ×/SD | Sentence | Note |
|---|---|---|---|---|
| ↑ | `recent_txns_gym` | 2.46 | More recent payments to gym (last 90 days) increases the probability of gym (odds x2.46 per SD). |  |
| ↑ | `tenure_days_gym` | 1.78 | More age of the gym subscription increases the probability of gym (odds x1.78 per SD). |  |
| ↑ | `is_live_gym` | 1.41 | More the gym subscription still running increases the probability of gym (odds x1.41 per SD). |  |
| ↑ | `n_distinct_mcc` | 1.40 | More number of different merchant types increases the probability of gym (odds x1.40 per SD). |  |
| ↑ | `n_topups` | 1.24 | More number of top-ups increases the probability of gym (odds x1.24 per SD). |  |
| ↓ | `recent_txns_insurance` | 0.60 | More recent payments to insurance (last 90 days) decreases the probability of gym (odds x0.60 per SD). |  |
| ↓ | `tenure_days_insurance` | 0.69 | More age of the insurance subscription decreases the probability of gym (odds x0.69 per SD). |  |
| ↓ | `port_families_ever` | 0.76 | More number of subscription families ever decreases the probability of gym (odds x0.76 per SD). |  |
| ↓ | `recent_txns_mobile` | 0.78 | More recent payments to mobile (last 90 days) decreases the probability of gym (odds x0.78 per SD). |  |

## insurance  (53 of 103 features used, 23 stable)

| Direction | Feature | Odds ×/SD | Sentence | Note |
|---|---|---|---|---|
| ↑ | `recent_txns_insurance` | 2.33 | More recent payments to insurance (last 90 days) increases the probability of insurance (odds x2.33 per SD). |  |
| ↑ | `tenure_days_insurance` | 1.77 | More age of the insurance subscription increases the probability of insurance (odds x1.77 per SD). |  |
| ↑ | `is_live_insurance` | 1.65 | More the insurance subscription still running increases the probability of insurance (odds x1.65 per SD). |  |
| ↑ | `n_distinct_mcc` | 1.35 | More number of different merchant types increases the probability of insurance (odds x1.35 per SD). |  |
| ↑ | `refund_insurance_90d` | 1.35 | More a recent refund from insurance increases the probability of insurance (odds x1.35 per SD). |  |
| ↓ | `refund_n_90d` | 0.68 | More number of families with a recent refund decreases the probability of insurance (odds x0.68 per SD). | ⚠️ refund story failed the confounder check |
| ↓ | `short_live_stream` | 0.76 | More longest live subscription being a 3-4 charge trial decreases the probability of insurance (odds x0.76 per SD). |  |
| ↓ | `recent_txns_mobile` | 0.76 | More recent payments to mobile (last 90 days) decreases the probability of insurance (odds x0.76 per SD). |  |
| ↓ | `recent_txns_software` | 0.77 | More recent payments to software (last 90 days) decreases the probability of insurance (odds x0.77 per SD). |  |

## mobile  (55 of 103 features used, 21 stable)

| Direction | Feature | Odds ×/SD | Sentence | Note |
|---|---|---|---|---|
| ↑ | `recent_txns_mobile` | 2.82 | More recent payments to mobile (last 90 days) increases the probability of mobile (odds x2.82 per SD). |  |
| ↑ | `tenure_days_mobile` | 1.79 | More age of the mobile subscription increases the probability of mobile (odds x1.79 per SD). |  |
| ↑ | `is_live_mobile` | 1.47 | More the mobile subscription still running increases the probability of mobile (odds x1.47 per SD). |  |
| ↑ | `first_due_mobile` | 1.38 | More mobile being the next subscription due increases the probability of mobile (odds x1.38 per SD). |  |
| ↑ | `n_distinct_mcc` | 1.28 | More number of different merchant types increases the probability of mobile (odds x1.28 per SD). |  |
| ↓ | `recent_txns_software` | 0.69 | More recent payments to software (last 90 days) decreases the probability of mobile (odds x0.69 per SD). |  |
| ↓ | `recent_txns_streaming` | 0.79 | More recent payments to streaming (last 90 days) decreases the probability of mobile (odds x0.79 per SD). |  |
| ↓ | `recent_txns_insurance` | 0.80 | More recent payments to insurance (last 90 days) decreases the probability of mobile (odds x0.80 per SD). |  |
| ↓ | `recent_txns_cloud` | 0.82 | More recent payments to cloud (last 90 days) decreases the probability of mobile (odds x0.82 per SD). |  |

## music  (23 of 103 features used, 6 stable)

| Direction | Feature | Odds ×/SD | Sentence | Note |
|---|---|---|---|---|
| ↑ | `recent_txns_music` | 1.88 | More recent payments to music (last 90 days) increases the probability of music (odds x1.88 per SD). |  |
| ↑ | `first_due_music` | 1.39 | More music being the next subscription due increases the probability of music (odds x1.39 per SD). |  |
| ↓ | `port_families_last90d` | 0.84 | More number of subscription families in the last 90 days decreases the probability of music (odds x0.84 per SD). |  |
| ↓ | `short_live_stream` | 0.84 | More longest live subscription being a 3-4 charge trial decreases the probability of music (odds x0.84 per SD). |  |
| ↓ | `bill_day_music` | 0.93 | More later music billing day in the month decreases the probability of music (odds x0.93 per SD). |  |

## software  (60 of 103 features used, 24 stable)

| Direction | Feature | Odds ×/SD | Sentence | Note |
|---|---|---|---|---|
| ↑ | `recent_txns_software` | 2.35 | More recent payments to software (last 90 days) increases the probability of software (odds x2.35 per SD). |  |
| ↑ | `tenure_days_software` | 2.10 | More age of the software subscription increases the probability of software (odds x2.10 per SD). |  |
| ↑ | `bill_day_streaming` | 1.52 | More later streaming billing day in the month increases the probability of software (odds x1.52 per SD). |  |
| ↑ | `first_due_software` | 1.40 | More software being the next subscription due increases the probability of software (odds x1.40 per SD). |  |
| ↑ | `is_live_software` | 1.34 | More the software subscription still running increases the probability of software (odds x1.34 per SD). |  |
| ↓ | `recent_txns_streaming` | 0.60 | More recent payments to streaming (last 90 days) decreases the probability of software (odds x0.60 per SD). |  |
| ↓ | `recent_txns_gym` | 0.79 | More recent payments to gym (last 90 days) decreases the probability of software (odds x0.79 per SD). |  |
| ↓ | `first_due_music` | 0.81 | More music being the next subscription due decreases the probability of software (odds x0.81 per SD). |  |
| ↓ | `recent_txns_insurance` | 0.82 | More recent payments to insurance (last 90 days) decreases the probability of software (odds x0.82 per SD). |  |

## streaming  (56 of 103 features used, 26 stable)

| Direction | Feature | Odds ×/SD | Sentence | Note |
|---|---|---|---|---|
| ↑ | `recent_txns_streaming` | 2.29 | More recent payments to streaming (last 90 days) increases the probability of streaming (odds x2.29 per SD). |  |
| ↑ | `is_live_streaming` | 1.68 | More the streaming subscription still running increases the probability of streaming (odds x1.68 per SD). |  |
| ↑ | `first_due_streaming` | 1.41 | More streaming being the next subscription due increases the probability of streaming (odds x1.41 per SD). |  |
| ↑ | `mean_amount_software` | 1.20 | More average software charge amount increases the probability of streaming (odds x1.20 per SD). |  |
| ↑ | `bill_day_cloud` | 1.15 | More later cloud billing day in the month increases the probability of streaming (odds x1.15 per SD). |  |
| ↓ | `n_distinct_mcc` | 0.67 | More number of different merchant types decreases the probability of streaming (odds x0.67 per SD). |  |
| ↓ | `port_families_last90d` | 0.74 | More number of subscription families in the last 90 days decreases the probability of streaming (odds x0.74 per SD). |  |
| ↓ | `refund_music_90d` | 0.78 | More a recent refund from music decreases the probability of streaming (odds x0.78 per SD). |  |
| ↓ | `refund_gym_90d` | 0.84 | More a recent refund from gym decreases the probability of streaming (odds x0.84 per SD). |  |

## none  (80 of 103 features used, 50 stable)

| Direction | Feature | Odds ×/SD | Sentence | Note |
|---|---|---|---|---|
| ↑ | `port_families_ever` | 1.63 | More number of subscription families ever increases the probability of none (odds x1.63 per SD). | ⚠️ twin of port_families_last90d (r = 0.78); raw none rate falls with it; read together as 'shrinking portfolio' |
| ↑ | `rule_says_none` | 1.46 | More the rule predicting none increases the probability of none (odds x1.46 per SD). |  |
| ↑ | `refund_n_90d` | 1.34 | More number of families with a recent refund increases the probability of none (odds x1.34 per SD). | ⚠️ refund story failed the confounder check (effect comes from recent purchases) |
| ↑ | `max_live_n_occurrences` | 1.28 | More charges of the longest live subscription increases the probability of none (odds x1.28 per SD). | ⚠️ raw relation is U-shaped (44% / 14% / 15% / 22%); only meaningful next to short_live_stream and rule_says_none |
| ↑ | `refund_gym_90d` | 1.22 | More a recent refund from gym increases the probability of none (odds x1.22 per SD). |  |
| ↓ | `tenure_days_software` | 0.41 | More age of the software subscription decreases the probability of none (odds x0.41 per SD). |  |
| ↓ | `is_live_insurance` | 0.43 | More the insurance subscription still running decreases the probability of none (odds x0.43 per SD). |  |
| ↓ | `frac_card_payment` | 0.54 | More share of card payments decreases the probability of none (odds x0.54 per SD). |  |
| ↓ | `is_live_software` | 0.56 | More the software subscription still running decreases the probability of none (odds x0.56 per SD). |  |
