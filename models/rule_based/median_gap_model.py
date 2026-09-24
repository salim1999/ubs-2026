"""Robust due-date rule using the median historical payment interval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.category_map import (
    NON_SUBSCRIPTION_PHRASES,
    NON_SUBSCRIPTION_TYPES,
    TARGET_CATEGORIES,
)
from src.recurrence import _cluster_by_amount, _stream_stats, load_transactions


DEFAULT_CUTOFF = "2026-01-01"
DEFAULT_GRACE_DAYS = 5


def _median_gap_streams(transactions_path: str | Path) -> pd.DataFrame:
    """Build recurring streams and estimate their next median-gap due date."""
    transactions = load_transactions(str(transactions_path))
    outgoing = transactions[transactions["direction"] == "out"]
    candidates = outgoing[
        ~outgoing["type"].isin(NON_SUBSCRIPTION_TYPES)
        & ~outgoing["description"].str.contains(
            "|".join(NON_SUBSCRIPTION_PHRASES)
        )
    ]

    rows = []
    for client_id, client_transactions in candidates.groupby("client_id"):
        for cluster in _cluster_by_amount(client_transactions):
            stats = _stream_stats(cluster)
            if not stats["is_recurring"] or stats["category"] is None:
                continue

            dates = cluster["timestamp"].sort_values()
            gaps = dates.diff().dt.days.dropna().to_numpy(dtype=float)
            median_gap = float(np.median(gaps))
            rows.append(
                {
                    "client_id": client_id,
                    "category": stats["category"],
                    "n_occurrences": stats["n_occurrences"],
                    "next_due": stats["last_date"]
                    + pd.Timedelta(days=median_gap),
                }
            )

    return pd.DataFrame(rows)


def median_gap_predict(
    client_ids: pd.Series,
    transactions_path: str | Path,
    cutoff_date: str = DEFAULT_CUTOFF,
    grace_days: int = DEFAULT_GRACE_DAYS,
) -> np.ndarray:
    """Predict each client's next family using a robust median cadence.

    The rule is identical to the original due-date baseline except for one
    deliberate change:

    ``next due = last payment + median(historical payment gaps)``

    The median is less sensitive than the mean to skipped, early, or delayed
    payments. A stream more than ``grace_days`` overdue is treated as inactive.
    """
    cutoff = pd.Timestamp(cutoff_date, tz="UTC")
    streams = _median_gap_streams(transactions_path)
    streams["days_until_due"] = (
        streams["next_due"] - cutoff
    ).dt.total_seconds() / 86_400
    live = streams[streams["days_until_due"] >= -grace_days]

    # Preserve the original short-trial gate.
    largest_live_stream = live.groupby("client_id")["n_occurrences"].max()
    short_trial_clients = set(
        largest_live_stream[largest_live_stream.isin([3, 4])].index
    )

    # Preserve the baseline's conservative aggregation for multiple streams
    # in the same family, isolating median cadence as the only change.
    family_due = (
        live.groupby(["client_id", "category"])["days_until_due"]
        .max()
        .reset_index()
    )
    family_order = {family: i for i, family in enumerate(TARGET_CATEGORIES)}
    family_due["family_order"] = family_due["category"].map(family_order)
    next_family = (
        family_due.sort_values(
            ["client_id", "days_until_due", "family_order"]
        )
        .groupby("client_id", sort=False)["category"]
        .first()
    )

    return np.asarray(
        [
            "none" if client_id in short_trial_clients else next_family.get(client_id, "none")
            for client_id in client_ids
        ]
    )
