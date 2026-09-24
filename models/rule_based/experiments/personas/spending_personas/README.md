# Spender-persona due-date rules

This experiment asks whether the same transparent due-date rule should use
slightly different settings for different kinds of account holders. It creates
four visual, human-readable spender personas and gives a persona a local rule
only when the train evidence is stable enough.

## Result

| Model | Validation macro-F1 | Delta vs original |
|---|---:|---:|
| Original mean-gap rule | 0.484420 | — |
| Global median-gap rule | **0.496853** | +0.012432 |
| Persona-specific rules | 0.493632 | +0.009211 |

The persona model does **not** beat the global median rule (`-0.003221`). The
useful lesson is that account behavior makes a good explanation layer, but the
available 2,000 labeled clients do not support reliably different prediction
rules. The safest production choice remains the single global median-gap rule.

Only **Low-key everyday users** passed the train-only customization guardrails.
For that group, train CV selected mean gap, three grace days, and no short-stream
gate. On validation this reduced its within-persona macro-F1 from `0.579971` to
`0.535550`. The other three personas used the global fallback and therefore had
identical predictions to the global median rule.

## The four personas

| Persona | Train clients | Typical behavior (medians) |
|---|---:|---|
| High-velocity spenders | 604 | 7.1 transactions/month, CHF 670 outflow/month, 3 active categories |
| Subscription regulars | 586 | 5.0 transactions/month, card-oriented, 2 active categories |
| Transfer & cash movers | 588 | Higher ATM/P2P mix, CHF 2,360 inflow/month, 1 active category |
| Low-key everyday users | 222 | 3.0 transactions/month, CHF 237 outflow/month, 1 active category |

Open [spender_personas.svg](spender_personas.svg) for the presentation-ready
radar view. The chart uses within-feature centroid percentiles: a larger radius
means the persona displays more of that behavior relative to the other personas.
The raw medians live in [persona_raw_profiles.csv](persona_raw_profiles.csv), and
the values behind the chart live in
[persona_profile_cards.csv](persona_profile_cards.csv).

## Method

The segmentation uses seven understandable account-level dimensions:

1. monthly transaction activity;
2. monthly outflow;
3. monthly inflow;
4. card-payment share;
5. ATM plus P2P share;
6. merchant-category breadth;
7. number of active recurring categories.

Skewed count and monetary values are log-transformed, all seven dimensions are
standardized, and K-means is fitted on **training clients only**. Four clusters
were fixed for a communicable persona set. As a sensitivity check, silhouette
scores were `0.2154`, `0.1784`, and `0.1743` for 3, 4, and 5 clusters. These
modest scores say that the personas overlap; they are useful summaries, not
four naturally separated populations.

For each persona, 60 simple due-date configurations are evaluated with
five-fold train-only CV:

- cadence: mean, median, trimmed mean, recent-three median, mean/median blend,
  or calendar day-of-month;
- overdue grace: 0, 3, 5, 7, or 10 days;
- short 3–4-payment stream gate: on or off.

A local rule is accepted only when the persona has at least 200 training
clients, its improvement is at least `0.003` after sample-size shrinkage, and it
beats the global median rule in at least three of five folds. Otherwise the
persona falls back to `median gap + 5 grace days + short-stream gate`. All
choices are frozen before validation data is loaded.

The design follows the classic RFM idea—recency, frequency, and monetary
behavior—but extends it with channel mix, category breadth, subscription
engagement, and payment interval. This is consistent with research that treats
RFM as a base behavioral representation and adds interval or engagement for
richer segmentation:

- [RFM-based repurchase behavior for customer classification and segmentation](https://doi.org/10.1016/j.jretconser.2021.102566)
- [An analytical framework based on RFM and time-series clustering](https://doi.org/10.1016/j.eswa.2021.116373)
- [RFMI-based Customer Segmentation with K-means](https://doi.org/10.1109/BigData62323.2024.10825251)

## Run

From the repository root:

```bash
py -3.10 models/rule_based/experiments/personas/spending_personas/experiment.py
```

Runtime is roughly 2–3 minutes because recurring streams are reconstructed from
raw transactions. The script rewrites all generated CSV, JSON, and SVG
artifacts in this folder deterministically (`random_state=42`).

## Artifacts

- `results.json`: headline scores and locked configurations.
- `spender_personas.svg`: standalone presentation graphic.
- `persona_profile_cards.csv`: radar values and persona sizes.
- `persona_raw_profiles.csv`: readable medians in original units.
- `cluster_sensitivity.csv`: K=3/4/5 silhouette and group-size checks.
- `train_rule_selection.csv`: complete train-only rule-selection audit trail.
- `train_persona_assignments.csv`: training client-to-persona mapping.
- `validation_by_persona.csv`: diagnostic scores by persona.
- `validation_predictions.csv`: paired global/persona predictions.

## Recommendation

Use the personas in the story and visualization, but **do not use their custom
rules for the submission**. Keep the global median-gap predictor. If more labels
become available, the next safe test is to retain these fixed four personas and
re-estimate only their three rule settings; that avoids redefining the segments
after seeing outcomes.
