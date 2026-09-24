# Simple-model comparison

Date: 2026-09-24

## Evaluation protocol

- Rebuild the current features from the raw transaction files.
- Fit on the 2,000 training clients.
- Select every threshold or architecture setting using stratified five-fold
  out-of-fold predictions on train only.
- Score once on the fixed 1,000-client validation set with the competition's
  eight-class macro-F1.

This keeps validation comparable with the existing experiment log and avoids
choosing hyperparameters on the reported validation score.

## Literature-informed candidates

1. **Two-stage hurdle logistic regression.** First predict `none` versus a
   recurring family, then predict which of the seven families. This mirrors the
   two distinct decisions in the task while keeping both models linear.
2. **Shared family ranker.** Reshape every client into seven client-family rows
   and fit one binary logistic regression with coefficients shared across
   families. This is the strongest pooling/global-model interpretation: a due
   date or live-stream signal has the same role for cloud, gym, and streaming.
3. **Clustered logistic experts.** Cluster clients from general activity and
   subscription-state features, fit a logistic regression per cluster, and
   shrink its probabilities toward one global logistic model. This is a small,
   interpretable mixture of experts rather than a black-box model.
4. **Rule/logistic blend.** Average the global logistic probabilities with a
   one-hot prediction from the due-date rule. This combines a mechanistic view
   of recurrence with weak statistical signals from the full feature table.

The design is motivated by work showing that global models can generalize well
across groups of series, while localized experts can help when genuine regimes
exist. Forecast-combination research also finds simple combinations difficult
to beat. Because macro-F1 is threshold-sensitive, the hurdle thresholds were
selected from train-only out-of-fold predictions.

Classical forward/backward stepwise selection was not prioritized. The present
features are highly correlated within each family, and published resampling
work shows that automated stepwise logistic selection can be unstable. The
experiment log already contains the more defensible sparse alternative: L1
selection followed by L2 refitting did not improve the full logistic model.

## Results

| Model | Validation macro-F1 |
|---|---:|
| Rule + global logistic blend | **0.4867** |
| Due-date rule baseline | **0.4844** |
| Global logistic baseline | 0.4601 |
| Shared family ranker | 0.4530 |
| Clustered logistic experts | 0.4507 |
| Two-stage hurdle logistic | 0.4445 |

Train-only choices:

- hurdle: `none` threshold 0.70;
- shared ranker: `none` threshold 0.70;
- clustered experts: four clusters and 25% local / 75% global probability;
- blend: 40% due-date rule / 60% global logistic probability.

The blend gains only 0.0023 over the rule. That is much smaller than the
roughly 0.01--0.02 validation noise noted in `PIPELINE.md`, so it is not a
meaningful win. The honest conclusion is that the due-date rule and blend are
tied on this validation set.

## Recommendation

Use the **due-date rule as the primary selling point and default submission**:
it is the simplest model, has the clearest causal story, and essentially ties
the best tested system. Keep the **rule/logistic blend** as the optional
accuracy-oriented submission. Do not pursue cluster-specific models yet: the
validation result says the available 2,000 training clients are insufficient
for useful segmentation, and the train/validation shift makes the local
experts less robust.

The next valuable experiment is not another classifier. Improve recurrence
detection and next-due estimation upstream, then rerun the same rule. That
preserves the simplicity narrative and targets the component that actually
drives performance.

## Sources

- Montero-Manso & Hyndman (2021), [Principles and algorithms for forecasting
  groups of time series: locality and globality](https://robjhyndman.com/publications/global-forecasting/).
- Jacobs et al. (1991), [Adaptive mixtures of local
  experts](https://people.eecs.berkeley.edu/~jordan/papers/mixtures-of-experts-bitmap.pdf).
- Wang et al. (2023), [Forecast combinations: an over 50-year
  review](https://robjhyndman.com/publications/combinations/).
- Lipton et al. (2014), [Thresholding classifiers to maximize F1
  score](https://arxiv.org/abs/1402.1892).
- Wallisch et al. (2021), [Selection of variables for multivariable models:
  opportunities and limitations in quantifying model stability by
  resampling](https://doi.org/10.1002/sim.8779).

## Reproduction

```powershell
py -3.10 -m src.compare_simple_models
```

The command writes the detailed per-class results to
`simple_model_comparison.csv`.
