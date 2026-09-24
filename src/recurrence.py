"""Detects recurring merchant streams per client from raw transactions.

A "stream" is a cluster of a client's transactions at the same mcc and a
similar amount, found via 1D amount clustering rather than exact
description matching. This is necessary because some recurring
subscriptions rotate their description text month to month (e.g.
"digital plus" -> "premium plan" -> "media streaming" for what is really
one Netflix-like charge) while staying at a near-constant amount and
cadence - grouping by literal description or by a single per-transaction
category label would split or mis-drop those cycles. See category_map.py
for the details behind this design.

A stream is flagged as recurring if it has >=2 transactions at a roughly
regular ~monthly cadence and a stable amount. Its category is assigned by
majority vote across the keyword-derived per-transaction categories of its
own members (imputing across the rotating-description cycles); a stream
with zero keyword evidence anywhere stays unresolved (category=None).
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from src.category_map import NON_SUBSCRIPTION_LITERALS, classify

# Only these two mccs are known to (a) be overloaded with unrelated real
# merchants and (b) have subscriptions that rotate their description text
# cycle to cycle. Amount-clustering is the fix for that specific problem.
# Applying it to every mcc instead over-fragments mccs that don't have this
# issue (pharmacy, hotel, ATM, ride-share, ...): many small 2-txn amount
# clusters land inside the 20-45 day / low-variance recurring window by
# pure chance, producing false positives. Those mccs keep the simpler
# whole-group test.
AMOUNT_CLUSTERED_MCCS = {"5812", "5732"}

# Single monthly cadence band. Six variants were tried to also admit
# weekly/biweekly billing (contiguous 14-45/14-90, disjoint bands at loose
# uniform/moderate/tight/very-tight amount_cv, monthly-only with loosened
# gap_cv) - all six made macro-F1 worse (0.3982 -> 0.3794 / 0.3594 / 0.3673
# / 0.3618 / 0.3783 / 0.3817 respectively). Whatever weekly/biweekly signal
# exists in this dataset is too rare or too entangled with coincidental
# same-category pairs to extract net-positively under any tolerance tried.
# Keep the single, original, proven monthly band.
GAP_BANDS = [
    {"range": (20, 45), "max_amount_cv": 0.35},
]
MAX_GAP_CV = 0.5

# Amount-clustering tolerance: start a new cluster when the next amount
# (sorted ascending) jumps by more than this relative/absolute margin from
# the running cluster mean.
AMOUNT_REL_TOL = 0.25
AMOUNT_ABS_TOL = 3.0


def load_transactions(path: str) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame.from_records(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["category"] = [
        classify(mcc, desc) for mcc, desc in zip(df["mcc"], df["description"])
    ]
    return df


def _is_excluded_literal(mcc: str, description: str) -> bool:
    return description in NON_SUBSCRIPTION_LITERALS.get(mcc, set())


def _cluster_by_amount(group: pd.DataFrame) -> list[pd.DataFrame]:
    """Greedy 1D clustering of a (client_id, mcc) group's rows by amount."""
    ordered = group.sort_values("amount")
    clusters: list[list[int]] = []
    current_idx: list[int] = []
    running_mean = None

    for idx, amount in zip(ordered.index, ordered["amount"]):
        if current_idx and running_mean is not None:
            tol = max(AMOUNT_ABS_TOL, AMOUNT_REL_TOL * running_mean)
            if amount - running_mean > tol:
                clusters.append(current_idx)
                current_idx = []
                running_mean = None

        current_idx.append(idx)
        cluster_amounts = ordered.loc[current_idx, "amount"]
        running_mean = cluster_amounts.mean()

    if current_idx:
        clusters.append(current_idx)

    return [group.loc[idx_list] for idx_list in clusters]


def _stream_stats(cluster: pd.DataFrame) -> dict:
    dates = cluster["timestamp"].sort_values()
    amounts = cluster["amount"]
    n = len(cluster)

    gaps = dates.diff().dt.days.dropna().to_numpy()
    mean_gap = float(np.mean(gaps)) if len(gaps) else np.nan
    gap_cv = float(np.std(gaps) / mean_gap) if len(gaps) and mean_gap > 0 else np.nan

    mean_amount = float(amounts.mean())
    amount_cv = float(amounts.std() / mean_amount) if n > 1 and mean_amount > 0 else 0.0

    matched_band = None
    if not np.isnan(mean_gap):
        for band in GAP_BANDS:
            lo, hi = band["range"]
            if lo <= mean_gap <= hi:
                matched_band = band
                break

    is_recurring = bool(
        n >= 2
        and matched_band is not None
        and (np.isnan(gap_cv) or gap_cv <= MAX_GAP_CV)
        and amount_cv <= matched_band["max_amount_cv"]
    )

    categories = [c for c in cluster["category"] if pd.notna(c)]
    category = Counter(categories).most_common(1)[0][0] if categories else None
    category_agreement = (
        Counter(categories).most_common(1)[0][1] / len(categories) if categories else np.nan
    )

    return {
        "n_occurrences": n,
        "first_date": dates.iloc[0],
        "last_date": dates.iloc[-1],
        "mean_amount": mean_amount,
        "amount_cv": amount_cv,
        "mean_gap_days": mean_gap,
        "gap_cv": gap_cv,
        "is_recurring": is_recurring,
        "category": category,
        "category_agreement": category_agreement,
        "n_keyword_hits": len(categories),
        "descriptions": sorted(set(cluster["description"])),
    }


def detect_streams(df: pd.DataFrame) -> pd.DataFrame:
    """One row per recurring-candidate amount cluster.

    Clusters within known non-subscription literal descriptions (real
    dining / electronics / marketplace spend on the overloaded mccs) are
    excluded upfront so they can't dilute or merge with genuine
    subscription-amount clusters.
    """
    pool = df[
        ~df.apply(lambda r: _is_excluded_literal(r["mcc"], r["description"]), axis=1)
    ]

    rows = []
    for (client_id, mcc), group in pool.groupby(["client_id", "mcc"]):
        clusters = (
            _cluster_by_amount(group) if mcc in AMOUNT_CLUSTERED_MCCS else [group]
        )
        for cluster in clusters:
            stats = _stream_stats(cluster)
            rows.append({"client_id": client_id, "mcc": mcc, **stats})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/train_transactions.jsonl"
    df = load_transactions(path)
    streams = detect_streams(df)

    print(f"Loaded {len(df):,} transactions for {df['client_id'].nunique():,} clients")
    print(f"Total amount-clusters: {len(streams):,}")
    print(f"Recurring clusters: {streams['is_recurring'].sum():,}")
    print()

    recurring = streams[streams["is_recurring"]]
    print("Recurring clusters by category (None = zero keyword evidence anywhere in cluster):")
    print(recurring["category"].value_counts(dropna=False))
    print()

    unresolved = recurring[recurring["category"].isna()]
    print(f"Unresolved recurring clusters: {len(unresolved)} (was 151 before the fix)")
    print(f"Unique clients affected: {unresolved['client_id'].nunique()}")
