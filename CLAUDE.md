# CLAUDE.md

Hackathon project (UBS challenge, Swiss AI Weeks 2026): predict each client's
**next recurring merchant family** in the 90 days after the `2026-01-01` cutoff.
Full task spec and submission rules are in `README.md`; read it before changing
labels, output schema or the metric.

- Labels: `cloud, gym, insurance, mobile, music, software, streaming, none`
- Metric: **macro-F1** over all 8 labels. Small classes count as much as `none`,
  so recall on rare families matters more than overall accuracy.
- Data is synthetic, and its noise is generated on purpose: description
  affixes and truncations, about 5% of charges on the wrong mcc, refunds, and
  shared generic phrases. Many gains come from auditing that noise in the data.

## Environment

- Windows, Python 3.11+, venv at `.venv/` (`.venv/Scripts/python`).
- Dependencies: numpy, pandas, scikit-learn only (`pyproject.toml`). Ask
  before adding a heavy dependency (torch, lightgbm, ...). It has to go into
  `pyproject.toml` and the README reproduction steps too.
- Run modules from the repo root as `python -m src.<module>`. Imports are
  `from src.x import ...`.

## Pipeline

```bash
python -m src.features   # raw jsonl -> data/processed/{train,valid,test}_features.csv
python -m src.model      # fit on train, score on valid: majority / rule / logreg / hgb / rank
python -m src.evaluate   # 5-fold stratified CV on train+valid (+ nested class-weight tuning)
python -m src.predict    # choose model on valid, refit on train+valid, write submission.csv
```

Any change to `recurrence.py`, `category_map.py`, `features.py` or `personas.py`
means you must re-run `src.features` before a score means anything.

| Module | Role |
| --- | --- |
| `category_map.py` | description/mcc → family, with strength `strong` / `mcc` / `generic` |
| `recurrence.py` | per-client recurring streams: amount clustering, cadence bands, generic-phrase attachment |
| `features.py` | one row per client: per-category stream state, recent (early-adoption) activity, general behaviour, selected persona features |
| `personas.py` | rule-based behavioural features; only `PERSONA_RAW_FEATURES` reach the model |
| `model.py` | baselines, `build_hgb`, `build_logreg`, `RankingModel` (candidate ranker + HGB for `none`) |
| `evaluate.py` | CV harness, the reference for deciding whether a change helps |
| `predict.py` | submission writer |

## Data layout

- Scripts read raw data from `data/dataset/dataset/` (unzipped from
  `data/dataset.zip`). Copies of the same files also sit in `data/`.
- `data/processed/` and `data/dataset/` are gitignored. Rebuild them, don't
  commit them.
- `data/unlabeled_pretrain_transactions.jsonl` is over 100 MB and gitignored.
  Never commit it.
- The cutoff is `2026-01-01`. Features may only use transactions `<= cutoff`.
  Leakage past the cutoff, or from labels into features, invalidates the result.

## How to judge a change

- A single 1000-client valid split is noisy (±~0.02 macro-F1). Don't claim an
  improvement from train→valid alone. Run `python -m src.evaluate` and compare
  the 5-fold CV score, per-class F1 and the nested tuned-weights score with
  the previous run.
- Report the numbers before and after, per class, in your summary. If
  something didn't help, say so and revert it rather than leave dead code.
- Tuning class weights or thresholds on the same data you score on is
  optimistic. Use the nested estimate in `evaluate.py`.
- Keep features that are clearly above noise in permutation importance, and
  drop the rest. That's how `PERSONA_RAW_FEATURES` was chosen.
- NaN in stream features means "no active subscription". It carries
  information, so don't impute it away for tree models.

## Submissions

- `submission.csv` (repo root) is the file to upload. It needs exactly the
  client IDs from `sample_submission.csv`, in that order, with columns
  `client_id,predicted_next_recurring_merchant`, and only allowed labels.
- After writing it, check the row count (1000), the set of IDs, the label set
  and the prediction distribution against the valid label distribution.
- Commit the code together with the `submission.csv` it produced, so every
  submitted file can be reproduced. Mention the CV score in the commit message.
- Submitting through the Google Form is done by the user. Never do it yourself.

## Code conventions

- Module docstrings explain the *why*, backed by data audits ("~5% of
  charges carry an unrelated mcc..."). When you change a heuristic or
  threshold, update the docstring or comment with the evidence behind it.
- Put tunable constants at module top in UPPER_CASE (`CADENCE_BANDS`,
  `AMOUNT_REL_TOL`, ...), not inline.
- Keep everything deterministic: fixed seeds (`SEED = 0`), no dependence on
  row order.
- Use plain scripts with `argparse`. Put exploration in the scratchpad or a
  throwaway script, not in `src/`, unless it becomes part of the pipeline.

## Git

- Work on the `Salim` branch. `main` is the PR target. Don't push or open
  PRs unless asked.
- Large regenerable artifacts stay out of git (see `.gitignore`).
