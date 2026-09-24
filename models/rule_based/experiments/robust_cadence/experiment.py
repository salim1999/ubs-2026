"""Train-only comparison of robust next-charge cadence rules.

Run from the repository root with::

    py -3.10 models/rule_based/experiments/robust_cadence/experiment.py

The validation labels are loaded only after an estimator/grace period has
been selected on the training split.
"""

from __future__ import annotations

import calendar
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.rule_based.model import rule_predict
from src.category_map import (
    NON_SUBSCRIPTION_PHRASES,
    NON_SUBSCRIPTION_TYPES,
    TARGET_CATEGORIES,
)
from src.features import build_features
from src.model import ALL_LABELS, LABEL_COL
from src.recurrence import _cluster_by_amount, _stream_stats, load_transactions


CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
OUT_DIR = Path(__file__).resolve().parent
BOOTSTRAP_SEED = 17
N_BOOTSTRAP = 5_000


def macro_f1(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(
        f1_score(
            y_true, y_pred, labels=ALL_LABELS, average="macro", zero_division=0
        )
    )


def paired_client_bootstrap(
    y_true: np.ndarray,
    selected_pred: np.ndarray,
    baseline_pred: np.ndarray,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Bootstrap clients, keeping the two models paired within each draw."""
    rng = np.random.default_rng(seed)
    n_clients = len(y_true)
    deltas = np.empty(n_resamples, dtype=float)
    for draw in range(n_resamples):
        rows = rng.integers(0, n_clients, size=n_clients)
        deltas[draw] = macro_f1(
            y_true[rows], selected_pred[rows]
        ) - macro_f1(y_true[rows], baseline_pred[rows])
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "n_resamples": n_resamples,
        "seed": seed,
        "delta_mean": float(deltas.mean()),
        "delta_ci95_low": float(low),
        "delta_ci95_high": float(high),
        "p_delta_gt_zero": float(np.mean(deltas > 0)),
        "p_delta_ge_zero": float(np.mean(deltas >= 0)),
    }


def _calendar_due(last: pd.Timestamp, days_of_month: np.ndarray) -> pd.Timestamp:
    """Next monthly due date, using the stream's median day of month."""
    target_day = int(np.rint(np.median(days_of_month)))
    year, month = last.year, last.month + 1
    if month == 13:
        year, month = year + 1, 1
    day = min(target_day, calendar.monthrange(year, month)[1])
    return pd.Timestamp(year=year, month=month, day=day, tz=last.tz)


def build_stream_table(path: Path) -> pd.DataFrame:
    """Return recurring streams with several leakage-free cadence forecasts."""
    transactions = load_transactions(str(path))
    out = transactions[transactions["direction"] == "out"]
    pool = out[
        ~out["type"].isin(NON_SUBSCRIPTION_TYPES)
        & ~out["description"].str.contains("|".join(NON_SUBSCRIPTION_PHRASES))
    ]

    rows: list[dict] = []
    for client_id, group in pool.groupby("client_id"):
        for cluster in _cluster_by_amount(group):
            stats = _stream_stats(cluster)
            if not stats["is_recurring"] or stats["category"] is None:
                continue

            dates = cluster["timestamp"].sort_values()
            gaps = dates.diff().dt.days.dropna().to_numpy(dtype=float)
            recent = gaps[-3:]
            if len(gaps) >= 5:
                trimmed = np.sort(gaps)[1:-1]
            else:
                trimmed = gaps

            cadence = {
                "mean_gap": float(np.mean(gaps)),
                "median_gap": float(np.median(gaps)),
                "trimmed_mean_gap": float(np.mean(trimmed)),
                "recent3_median_gap": float(np.median(recent)),
                "mean_median_blend": float(
                    0.5 * np.mean(gaps) + 0.5 * np.median(gaps)
                ),
            }
            row = {
                "client_id": client_id,
                "category": stats["category"],
                "n_occurrences": stats["n_occurrences"],
                "last_date": stats["last_date"],
            }
            for name, gap in cadence.items():
                row[f"due_{name}"] = stats["last_date"] + pd.Timedelta(days=gap)
            row["due_calendar_dom"] = _calendar_due(
                stats["last_date"], dates.dt.day.to_numpy()
            )
            rows.append(row)
    return pd.DataFrame(rows)


def predict(
    clients: pd.Series,
    streams: pd.DataFrame,
    estimator: str,
    grace_days: int,
) -> np.ndarray:
    """Apply the existing global rule, replacing only next-due estimation."""
    due_col = f"due_{estimator}"
    work = streams.copy()
    work["days_until_due"] = (
        work[due_col] - CUTOFF
    ).dt.total_seconds() / 86_400
    live = work[work["days_until_due"] >= -grace_days]

    # Preserve the current short-stream gate: if the largest live stream has
    # only 3 or 4 observations, predict none for the whole client.
    max_n = live.groupby("client_id")["n_occurrences"].max()
    short_clients = set(max_n[max_n.isin([3, 4])].index)

    # Match the current feature aggregation exactly.  Where amount clustering
    # produces multiple streams for one family, the family's conservative due
    # value is the latest of those stream forecasts; the family with the
    # earliest such date wins.  This keeps cadence estimator as the only
    # substantive change from the official baseline.
    family_due = (
        live.groupby(["client_id", "category"])["days_until_due"]
        .max()
        .reset_index()
    )
    family_due["category_order"] = family_due["category"].map(
        {category: i for i, category in enumerate(TARGET_CATEGORIES)}
    )
    first = (
        family_due.sort_values(
            ["client_id", "days_until_due", "category_order"]
        )
        .groupby("client_id", sort=False)["category"]
        .first()
    )
    return np.asarray(
        [
            "none" if c in short_clients else first.get(c, "none")
            for c in clients
        ]
    )


def main() -> None:
    train_labels = pd.read_csv(ROOT / "data/train_labels.csv")
    train_streams = build_stream_table(ROOT / "data/train_transactions.jsonl")

    # Estimator and liveness grace are selected strictly on train labels.
    estimators = [
        "mean_gap",
        "median_gap",
        "trimmed_mean_gap",
        "recent3_median_gap",
        "mean_median_blend",
        "calendar_dom",
    ]
    rows = []
    for estimator in estimators:
        for grace in (0, 3, 5, 7, 10):
            pred = predict(
                train_labels["client_id"], train_streams, estimator, grace
            )
            rows.append(
                {
                    "estimator": estimator,
                    "grace_days": grace,
                    "train_macro_f1": macro_f1(train_labels[LABEL_COL], pred),
                }
            )
    train_results = pd.DataFrame(rows).sort_values(
        ["train_macro_f1", "estimator", "grace_days"],
        ascending=[False, True, True],
    )
    # Deterministic tie breaking above favours alphabetic name then lower grace.
    choice = train_results.iloc[0]

    # Validation is touched only after model selection is complete.
    valid_labels = pd.read_csv(ROOT / "data/valid_labels.csv")
    valid_streams = build_stream_table(ROOT / "data/valid_transactions.jsonl")
    chosen_pred = predict(
        valid_labels["client_id"],
        valid_streams,
        str(choice["estimator"]),
        int(choice["grace_days"]),
    )
    chosen_valid = macro_f1(valid_labels[LABEL_COL], chosen_pred)

    reconstructed_mean_pred = predict(
        valid_labels["client_id"], valid_streams, "mean_gap", 5
    )

    # Recompute the official feature-based baseline as a sanity check.
    valid_features = build_features(
        str(ROOT / "data/valid_transactions.jsonl"), str(CUTOFF.date())
    )
    aligned = valid_labels[["client_id"]].merge(
        valid_features, on="client_id", how="left"
    )
    baseline_pred = rule_predict(aligned.drop(columns=["client_id"]))
    baseline_valid = macro_f1(valid_labels[LABEL_COL], baseline_pred)
    reconstruction_valid = macro_f1(
        valid_labels[LABEL_COL], reconstructed_mean_pred
    )
    chosen_per_class = f1_score(
        valid_labels[LABEL_COL],
        chosen_pred,
        labels=ALL_LABELS,
        average=None,
        zero_division=0,
    )
    baseline_per_class = f1_score(
        valid_labels[LABEL_COL],
        baseline_pred,
        labels=ALL_LABELS,
        average=None,
        zero_division=0,
    )
    bootstrap = paired_client_bootstrap(
        valid_labels[LABEL_COL].to_numpy(), chosen_pred, baseline_pred
    )

    pd.DataFrame(
        {
            "client_id": valid_labels["client_id"],
            "true_label": valid_labels[LABEL_COL],
            "official_baseline_prediction": baseline_pred,
            "median_gap_prediction": chosen_pred,
        }
    ).to_csv(OUT_DIR / "validation_predictions.csv", index=False)

    train_results.to_csv(OUT_DIR / "train_selection.csv", index=False)
    summary = {
        "selected_estimator": str(choice["estimator"]),
        "selected_grace_days": int(choice["grace_days"]),
        "selected_train_macro_f1": float(choice["train_macro_f1"]),
        "selected_valid_macro_f1": chosen_valid,
        "official_baseline_valid_macro_f1": baseline_valid,
        "reconstructed_mean_valid_macro_f1": reconstruction_valid,
        "mean_reconstruction_disagreements": int(
            np.sum(reconstructed_mean_pred != baseline_pred)
        ),
        "valid_delta": chosen_valid - baseline_valid,
        "selected_valid_per_class_f1": dict(
            zip(ALL_LABELS, map(float, chosen_per_class))
        ),
        "baseline_valid_per_class_f1": dict(
            zip(ALL_LABELS, map(float, baseline_per_class))
        ),
        "paired_client_bootstrap": bootstrap,
    }
    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Train-only ranking (top 15):")
    print(train_results.head(15).to_string(index=False, float_format="%.6f"))
    print("\nFinal untouched-validation comparison:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
