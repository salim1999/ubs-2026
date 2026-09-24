# Client-persona rule experiments

These experiments ask whether the global due-date rule should become slightly
more individual without becoming opaque:

```text
client transactions -> interpretable persona -> persona-specific due-date rule
```

The motivation is not personalization for its own sake. A persona is useful
only if clients in that group have meaningfully different recurrence behavior
and a simple local rule generalizes better to unseen clients.

## Three independent approaches

1. **Spending personas:** profiles based on interpretable recency, frequency,
   monetary, cash-flow, channel-mix, and subscription-breadth features.
2. **Recurrence personas:** profiles based directly on payment-stream maturity,
   timing regularity, liveness, amount stability, and category diversity.
3. **Supervised personas:** a shallow, visual decision tree routes clients to a
   small set of timing rules.

Every approach must:

- construct and select profiles from training data only;
- lock persona definitions and local rules before reading validation labels;
- compare against the original mean-gap rule (`0.4844`) and global median-gap
  rule (`0.4969`);
- include a global fallback for small or uncertain groups;
- output human-readable persona names, profile tables, and a visualization.

## Research basis

- Transaction-behavior research represents spending using overall, temporal,
  category-related, and customer-profile feature groups:
  [Gladstone et al., EPJ Data Science](https://link.springer.com/article/10.1140/epjds/s13688-021-00281-y).
- RFM-style segmentation provides a compact, interpretable starting point for
  recency, frequency, and monetary profiles:
  [Anitha & Patil, bank-customer segmentation](https://arxiv.org/abs/2008.08662).
- Localized forecasting research tests clusters of related series with local
  specialists while retaining global information:
  [Godahewa et al., 2021](https://arxiv.org/abs/2012.15059).
- The mixture-of-experts framework formalizes routing cases to specialized
  models through a gating mechanism:
  [Jacobs et al., 1991](https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf).

The final decision will prioritize validation macro-F1, stability, minimum
persona size, and whether the visualization tells a truthful business story.

## Results

| Approach | Personas | Validation macro-F1 |
|---|---:|---:|
| Original global mean-gap rule | -- | 0.4844 |
| Recurrence personas | 4 | 0.4878 |
| Supervised persona router | 2 | 0.4917 |
| Spending personas | 4 | 0.4936 |
| **Global median-gap rule** | -- | **0.4969** |

### Spending personas

- High-velocity spenders
- Subscription regulars
- Transfer and cash movers
- Low-key everyday users

The persona rule changed only six validation predictions. Its one local rule
failed to generalize, so this segmentation is best used for presentation and
diagnostics. See [the profile cards and visualization](spending_personas/README.md).

### Recurrence personas

- Mature regular
- Multi-subscription active
- Short or lapsed explorer
- Quiet / no detected cycle

These profiles are the most directly connected to the prediction mechanism and
produce the clearest visual explanation. Their personalized cadence and grace
periods nevertheless underperform the global median rule. See the
[recurrence-persona analysis](recurrence_personas/README.md).

### Supervised personas

The locked shallow router found only one useful split:

```text
average outgoing payment <= CHF 140.61 -> median gap, 7-day grace
average outgoing payment >  CHF 140.61 -> median gap, 3-day grace
```

This is easy to visualize but also trails the global median rule in train-only
out-of-fold scoring and validation. See the
[supervised persona tree](supervised_personas/README.md).

## Decision

Use the personas as an **explanation and monitoring layer**, not as separate
prediction rules. For a client, show the assigned profile alongside the global
median-gap forecast, for example:

```text
Persona: Mature regular
Observed behavior: stable amounts, regular timing, several live streams
Forecast: mobile is due first, based on the global median-gap rule
```

This gives the project a visual, client-oriented story without sacrificing the
best validated prediction score. Revisit personalized rules only when more
labeled clients or multiple historical backtest cutoffs are available.
