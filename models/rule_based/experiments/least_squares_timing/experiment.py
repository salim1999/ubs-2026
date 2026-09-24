"""Pooled ridge timing model inside the transparent due-date rule.

The model learns *only* the next inter-payment gap.  It is trained on rolling
pseudo-cutoffs made from transaction history, without using validation labels
or any transaction after the prediction cutoff.  The class decision remains:

    predicted due date = last observed date + predicted next gap
    choose the live family with the earliest predicted due date

Run from the repository root with::

    py -3.10 models/rule_based/experiments/least_squares_timing/experiment.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Make the repository imports work when this file is run directly.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.rule_based.model import rule_predict
from src.category_map import (
    NON_SUBSCRIPTION_PHRASES,
    NON_SUBSCRIPTION_TYPES,
    TARGET_CATEGORIES,
)
from src.features import build_features
from src.recurrence import (
    MAX_AMOUNT_CV,
    MAX_GAP_CV,
    MAX_GAP_DAYS,
    MIN_GAP_DAYS,
    _cluster_by_amount,
    load_transactions,
)


CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
LABEL_COL = "target_next_recurring_merchant"
ALL_LABELS = [*TARGET_CATEGORIES, "none"]
LIVE_MAX_OVER_DAYS = 5
OUT_DIR = Path(__file__).resolve().parent


@dataclass
class FittedTimingModel:
    variant: str
    alpha: float
    scaler: StandardScaler
    ridge: Ridge
    columns: list[str]


def candidate_clusters(df: pd.DataFrame):
    """Yield the exact amount clusters used by the production detector."""
    out = df[df["direction"] == "out"]
    pool = out[
        ~out["type"].isin(NON_SUBSCRIPTION_TYPES)
        & ~out["description"].str.contains("|".join(NON_SUBSCRIPTION_PHRASES))
    ]
    for client_id, group in pool.groupby("client_id", sort=False):
        for cluster_no, cluster in enumerate(_cluster_by_amount(group)):
            yield client_id, cluster_no, cluster.sort_values("timestamp")


def _amount_cv(amounts: pd.Series) -> float:
    mean = float(amounts.mean())
    return float(amounts.std() / mean) if len(amounts) > 1 and mean > 0 else 0.0


def _prefix_category(prefix: pd.DataFrame) -> str | None:
    cats = prefix["category"].dropna().tolist()
    return Counter(cats).most_common(1)[0][0] if cats else None


def _is_recurring_prefix(prefix: pd.DataFrame) -> bool:
    gaps = prefix["timestamp"].diff().dt.total_seconds().dropna().to_numpy() / 86400
    if len(gaps) == 0:
        return False
    mean_gap = float(gaps.mean())
    gap_cv = float(gaps.std() / mean_gap) if mean_gap > 0 else np.inf
    return bool(
        MIN_GAP_DAYS <= mean_gap <= MAX_GAP_DAYS
        and gap_cv <= MAX_GAP_CV
        and _amount_cv(prefix["amount"]) <= MAX_AMOUNT_CV
    )


def _days_in_calendar_cycle(last_date: pd.Timestamp) -> float:
    """Days to the same day next month (clamped for short months)."""
    naive = last_date.tz_localize(None)
    next_month = naive + pd.offsets.MonthBegin(1)
    max_day = (next_month + pd.offsets.MonthEnd(0)).day
    target = next_month.replace(day=min(naive.day, max_day))
    return float((target - naive).days)


def timing_features(prefix: pd.DataFrame, family: str) -> dict[str, float | str]:
    dates = prefix["timestamp"].sort_values()
    gaps = dates.diff().dt.total_seconds().dropna().to_numpy() / 86400
    x = np.arange(len(gaps), dtype=float)
    trend = float(np.polyfit(x, gaps, 1)[0]) if len(gaps) >= 2 else 0.0
    last = dates.iloc[-1]
    mean_gap = float(gaps.mean())
    return {
        "family": family,
        "mean_gap": mean_gap,
        "median_gap": float(np.median(gaps)),
        "last_gap": float(gaps[-1]),
        "gap_std": float(gaps.std()),
        "gap_trend": trend,
        "n_prior_gaps": float(len(gaps)),
        "mean_amount": float(prefix["amount"].mean()),
        "amount_cv": _amount_cv(prefix["amount"]),
        "last_day_of_month": float(last.day),
        "last_month_sin": float(np.sin(2 * np.pi * last.month / 12)),
        "last_month_cos": float(np.cos(2 * np.pi * last.month / 12)),
        "calendar_gap": _days_in_calendar_cycle(last),
    }


def make_pseudo_samples(df: pd.DataFrame) -> pd.DataFrame:
    """One rolling-origin row for every observable next payment."""
    rows: list[dict] = []
    for client_id, cluster_no, cluster in candidate_clusters(df):
        # At least two prefix observations and one target observation.
        for end in range(2, len(cluster)):
            prefix = cluster.iloc[:end]
            family = _prefix_category(prefix)
            if family is None or not _is_recurring_prefix(prefix):
                continue
            target_gap = (
                cluster.iloc[end]["timestamp"] - prefix.iloc[-1]["timestamp"]
            ).total_seconds() / 86400
            # A genuine monthly stream can occasionally have a skipped charge;
            # retain it as a hard case, but omit unrelated very-long gaps.
            if not 10 <= target_gap <= 70:
                continue
            rows.append(
                {
                    **timing_features(prefix, family),
                    "target_gap": float(target_gap),
                    "stream_id": f"{client_id}:{cluster_no}",
                }
            )
    return pd.DataFrame(rows)


VARIANT_COLUMNS = {
    # A learned shrinkage version of the historical average.
    "shrinkage": ["mean_gap", "n_prior_gaps"],
    # Adapt to changes and family-specific timing.
    "recent": [
        "mean_gap",
        "median_gap",
        "last_gap",
        "gap_std",
        "gap_trend",
        "n_prior_gaps",
        "mean_amount",
        "amount_cv",
    ],
    # Add an explicit monthly-calendar signal (28/29/30/31-day months).
    "calendar": [
        "mean_gap",
        "median_gap",
        "last_gap",
        "gap_std",
        "gap_trend",
        "n_prior_gaps",
        "mean_amount",
        "amount_cv",
        "last_day_of_month",
        "last_month_sin",
        "last_month_cos",
        "calendar_gap",
    ],
}


def design_matrix(rows: pd.DataFrame, variant: str) -> pd.DataFrame:
    numeric = rows[VARIANT_COLUMNS[variant]].astype(float).reset_index(drop=True)
    families = pd.Categorical(rows["family"], categories=TARGET_CATEGORIES)
    one_hot = pd.get_dummies(families, prefix="family", dtype=float).reset_index(drop=True)
    return pd.concat([numeric, one_hot], axis=1)


def fit_timing_model(samples: pd.DataFrame, variant: str, alpha: float) -> FittedTimingModel:
    X = design_matrix(samples, variant)
    scaler = StandardScaler().fit(X)
    ridge = Ridge(alpha=alpha, solver="lsqr").fit(scaler.transform(X), samples["target_gap"])
    return FittedTimingModel(variant, alpha, scaler, ridge, list(X.columns))


def predict_gaps(model: FittedTimingModel, rows: pd.DataFrame) -> np.ndarray:
    X = design_matrix(rows, model.variant).reindex(columns=model.columns, fill_value=0)
    pred = model.ridge.predict(model.scaler.transform(X))
    return np.clip(pred, MIN_GAP_DAYS, MAX_GAP_DAYS)


def select_timing_model(samples: pd.DataFrame) -> tuple[FittedTimingModel, pd.DataFrame]:
    """Choose features/regularization using grouped pseudo-history CV only."""
    alphas = [0.1, 1.0, 10.0, 100.0]
    groups = samples["stream_id"]
    n_splits = min(5, groups.nunique())
    splitter = GroupKFold(n_splits=n_splits)
    result_rows = []

    # The current mean-gap estimator is the timing baseline.
    result_rows.append(
        {
            "variant": "historical_mean",
            "alpha": np.nan,
            "mae_days": mean_absolute_error(samples["target_gap"], samples["mean_gap"]),
            "rmse_days": mean_squared_error(
                samples["target_gap"], samples["mean_gap"], squared=False
            ),
        }
    )
    for variant in VARIANT_COLUMNS:
        for alpha in alphas:
            oof = np.empty(len(samples))
            for train_idx, test_idx in splitter.split(samples, groups=groups):
                fitted = fit_timing_model(samples.iloc[train_idx], variant, alpha)
                oof[test_idx] = predict_gaps(fitted, samples.iloc[test_idx])
            result_rows.append(
                {
                    "variant": variant,
                    "alpha": alpha,
                    "mae_days": mean_absolute_error(samples["target_gap"], oof),
                    "rmse_days": mean_squared_error(
                        samples["target_gap"], oof, squared=False
                    ),
                }
            )
    results = pd.DataFrame(result_rows).sort_values(["mae_days", "rmse_days"])
    learned = results[results["variant"] != "historical_mean"].iloc[0]
    model = fit_timing_model(samples, str(learned["variant"]), float(learned["alpha"]))
    return model, results


def final_stream_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for client_id, cluster_no, cluster in candidate_clusters(df):
        family = _prefix_category(cluster)
        if family is None or not _is_recurring_prefix(cluster):
            continue
        rows.append(
            {
                **timing_features(cluster, family),
                "client_id": client_id,
                "stream_id": f"{client_id}:{cluster_no}",
                "last_date": cluster["timestamp"].max(),
                "n_occurrences": len(cluster),
            }
        )
    return pd.DataFrame(rows)


def learned_rule_predict(
    df: pd.DataFrame,
    model: FittedTimingModel,
    blend_weight: float = 1.0,
    live_max_over_days: float = LIVE_MAX_OVER_DAYS,
    prepared_streams: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    streams = (
        final_stream_rows(df)
        if prepared_streams is None
        else prepared_streams.copy()
    )
    if streams.empty:
        clients = pd.Index(df["client_id"].unique(), name="client_id")
        return pd.Series("none", index=clients), streams

    learned_gap = predict_gaps(model, streams)
    streams["predicted_gap"] = (
        blend_weight * learned_gap + (1 - blend_weight) * streams["mean_gap"]
    )
    streams["recency"] = (CUTOFF - streams["last_date"]).dt.total_seconds() / 86400
    streams["over"] = streams["recency"] - streams["predicted_gap"]
    predictions: dict[str, str] = {}
    for client_id, client_streams in streams.groupby("client_id"):
        client_live = client_streams[
            client_streams["over"] <= live_max_over_days
        ]
        if client_live.empty:
            continue
        # Preserve the baseline's short-trial guard exactly.
        if client_live["n_occurrences"].max() in (3, 4):
            predictions[client_id] = "none"
        else:
            # Match src.features: represent a family with its minimum over
            # value across streams, then choose the largest family value.
            # This odd-looking aggregation is preserved so the experiment
            # isolates timing estimation rather than silently changing rules.
            family_over = client_streams.groupby("family")["over"].min()
            live_family_over = family_over[family_over <= live_max_over_days]
            predictions[client_id] = str(live_family_over.idxmax())
    clients = pd.Index(df["client_id"].unique(), name="client_id")
    return pd.Series(predictions, index=clients).fillna("none"), streams


def macro_f1(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(
        f1_score(y_true, y_pred, average="macro", labels=ALL_LABELS, zero_division=0)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pseudo-source",
        choices=["train", "pretrain", "both"],
        default="train",
        help="Transaction histories used for unlabeled rolling pseudo-cutoffs.",
    )
    args = parser.parse_args()

    print("Loading training histories and building pseudo-cutoffs ...", flush=True)
    train_tx = load_transactions("data/train_transactions.jsonl")
    sample_frames = [make_pseudo_samples(train_tx)]
    if args.pseudo_source in ("pretrain", "both"):
        if args.pseudo_source == "pretrain":
            sample_frames = []
        pretrain_tx = load_transactions("data/unlabeled_pretrain_transactions.jsonl")
        sample_frames.append(make_pseudo_samples(pretrain_tx))
    samples = pd.concat(sample_frames, ignore_index=True)
    print(
        f"Pseudo samples: {len(samples):,} from {samples['stream_id'].nunique():,} streams",
        flush=True,
    )

    timing_model, cv = select_timing_model(samples)
    cv.to_csv(OUT_DIR / "pseudo_timing_cv.csv", index=False)
    print("\nPseudo-history timing CV (best rows):")
    print(cv.head(8).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(
        f"\nSelected: {timing_model.variant}, alpha={timing_model.alpha:g}", flush=True
    )

    # Decision settings are chosen only on train labels.  A small, declared
    # grid avoids turning this transparent extension into a classifier.
    labels_train = pd.read_csv("data/train_labels.csv").set_index("client_id")[LABEL_COL]
    train_streams = final_stream_rows(train_tx)
    train_candidates = []
    for blend in (0.25, 0.5, 0.75, 1.0):
        for live_over in (3.0, 5.0, 7.0):
            pred, _ = learned_rule_predict(
                train_tx, timing_model, blend, live_over, prepared_streams=train_streams
            )
            score = macro_f1(labels_train, pred.reindex(labels_train.index).fillna("none"))
            train_candidates.append(
                {"blend_weight": blend, "live_max_over_days": live_over, "train_macro_f1": score}
            )
    train_grid = pd.DataFrame(train_candidates).sort_values(
        ["train_macro_f1", "blend_weight"], ascending=[False, True]
    )
    train_grid.to_csv(OUT_DIR / "train_decision_grid.csv", index=False)
    chosen = train_grid.iloc[0]
    print("\nTrain-only decision grid (best rows):")
    print(train_grid.head(8).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # The only use of validation data: one final comparison after settings lock.
    print("\nLocked settings; loading validation for the one final score ...", flush=True)
    valid_tx = load_transactions("data/valid_transactions.jsonl")
    labels_valid = pd.read_csv("data/valid_labels.csv").set_index("client_id")[LABEL_COL]
    learned_pred, valid_streams = learned_rule_predict(
        valid_tx,
        timing_model,
        float(chosen["blend_weight"]),
        float(chosen["live_max_over_days"]),
    )
    learned_score = macro_f1(
        labels_valid, learned_pred.reindex(labels_valid.index).fillna("none")
    )

    valid_features = build_features("data/valid_transactions.jsonl", str(CUTOFF.date()))
    baseline_pred = pd.Series(
        rule_predict(valid_features.drop(columns=["client_id"])),
        index=valid_features["client_id"],
    )
    baseline_score = macro_f1(
        labels_valid, baseline_pred.reindex(labels_valid.index).fillna("none")
    )
    per_class = f1_score(
        labels_valid,
        learned_pred.reindex(labels_valid.index).fillna("none"),
        average=None,
        labels=ALL_LABELS,
        zero_division=0,
    )

    result = {
        "pseudo_source": args.pseudo_source,
        "n_pseudo_samples": int(len(samples)),
        "n_pseudo_streams": int(samples["stream_id"].nunique()),
        "selected_variant": timing_model.variant,
        "selected_alpha": timing_model.alpha,
        "pseudo_cv_mae_days": float(cv.iloc[0]["mae_days"]),
        "historical_mean_mae_days": float(
            cv.loc[cv["variant"] == "historical_mean", "mae_days"].iloc[0]
        ),
        "blend_weight": float(chosen["blend_weight"]),
        "live_max_over_days": float(chosen["live_max_over_days"]),
        "train_macro_f1": float(chosen["train_macro_f1"]),
        "valid_baseline_macro_f1": baseline_score,
        "valid_learned_macro_f1": learned_score,
        "valid_delta": learned_score - baseline_score,
        "valid_per_class_f1": dict(zip(ALL_LABELS, map(float, per_class))),
        "n_valid_recurring_streams": int(len(valid_streams)),
    }
    (OUT_DIR / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("\nFINAL VALIDATION RESULT")
    print(f"  existing mean-gap rule : {baseline_score:.4f}")
    print(f"  ridge timing rule      : {learned_score:.4f}")
    print(f"  delta                  : {learned_score - baseline_score:+.4f}")
    print("  per-class F1:")
    for label, value in zip(ALL_LABELS, per_class):
        print(f"    {label:10s} {value:.4f}")
    print(f"\nWrote {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
