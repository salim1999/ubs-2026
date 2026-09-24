"""Detects recurring merchant streams per client from raw transactions.

A "stream" is a cluster of a client's transactions at a similar amount
(pooled across mccs), found via 1D amount clustering rather than exact
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
from itertools import combinations

import numpy as np
import pandas as pd

from src.category_map import NON_SUBSCRIPTION_PHRASES, NON_SUBSCRIPTION_TYPES, classify

MIN_GAP_DAYS = 20
MAX_GAP_DAYS = 45
MAX_GAP_CV = 0.5
MAX_AMOUNT_CV = 0.35

# Amount-clustering tolerance: start a new cluster when the next amount
# (sorted ascending) jumps by more than this relative/absolute margin from
# the running cluster mean. Subscription charges vary by ~1-2%, and all of a
# client's MCCs are pooled, so the margin has to be tight or neighbouring
# streams (e.g. cloud ~9 and music ~11) merge. On train, 0.5/0.05 .. 0.2/0.02
# give the same detection; 0.1/0.01 starts splitting real streams.
AMOUNT_REL_TOL = 0.03
AMOUNT_ABS_TOL = 0.3


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


def _cluster_by_amount(group: pd.DataFrame) -> list[pd.DataFrame]:
    """Greedy 1D clustering of one client's candidate rows by amount."""
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


def _raw_stream_stats(cluster: pd.DataFrame) -> dict:
    dates = cluster["timestamp"].sort_values()
    amounts = cluster["amount"]
    n = len(cluster)

    gaps = dates.diff().dt.days.dropna().to_numpy()
    mean_gap = float(np.mean(gaps)) if len(gaps) else np.nan
    gap_cv = float(np.std(gaps) / mean_gap) if len(gaps) and mean_gap > 0 else np.nan

    mean_amount = float(amounts.mean())
    amount_cv = float(amounts.std() / mean_amount) if n > 1 and mean_amount > 0 else 0.0

    is_recurring = bool(
        n >= 2
        and not np.isnan(mean_gap)
        and MIN_GAP_DAYS <= mean_gap <= MAX_GAP_DAYS
        and (np.isnan(gap_cv) or gap_cv <= MAX_GAP_CV)
        and amount_cv <= MAX_AMOUNT_CV
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


MAX_OUTLIERS_DROPPED = 1


def _stream_stats(cluster: pd.DataFrame) -> dict:
    """As _raw_stream_stats, but if the cluster fails is_recurring, try
    dropping up to MAX_OUTLIERS_DROPPED category-less members (fewest
    first) and keep the smallest removal that makes it pass.

    Motivation: a client's genuine recurring subscription can get merged,
    purely by amount coincidence within the tight 3% clustering tolerance,
    with an unrelated one-off transaction that has no keyword/category
    evidence at all - e.g. found on validation client C000545, where a
    single untagged ~$75 August charge joining two insurance payments
    pulled mean_gap from 29 to 69 days and gap_cv from 0 to 0.58, failing
    is_recurring even though the insurance pair alone would have passed
    cleanly. Only category-less members are ever considered for removal,
    so this can't discard a transaction that is itself keyword-evidence
    for a *different* real category.

    MAX_OUTLIERS_DROPPED=1 (dropping a single outlier) measured a real
    win: rule 0.4844 -> 0.4918, hgb 0.4344 -> 0.4684, c_detect 0.690 ->
    0.714. Tried extending to 2: worse on both stronger models (rule ->
    0.4883, hgb -> 0.4594) despite c_detect/c_fams both still rising -
    allowing 2 removals loosens the bar enough to manufacture some false
    passes out of noise, not just recover genuine false-negatives. Keep 1.
    """
    stats = _raw_stream_stats(cluster)
    if stats["is_recurring"]:
        return stats

    dropless_candidates = list(cluster[cluster["category"].isna()].index)
    if not dropless_candidates:
        return stats

    max_drop = min(MAX_OUTLIERS_DROPPED, len(dropless_candidates), len(cluster) - 2)
    for n_drop in range(1, max_drop + 1):
        for combo in combinations(dropless_candidates, n_drop):
            trimmed = cluster.drop(index=list(combo))
            trimmed_stats = _raw_stream_stats(trimmed)
            if trimmed_stats["is_recurring"]:
                return trimmed_stats

    return stats


def detect_streams(df: pd.DataFrame) -> pd.DataFrame:
    """One row per recurring-candidate amount cluster.

    Candidates are pooled per client across ALL mccs: the same subscription
    hops between mccs from month to month (e.g. a ~21.5 streaming charge
    alternating 5812 / 5411), so grouping by client x mcc split one stream
    into fragments that failed n>=2 or the 45-day gap rule.

    Known non-subscription merchants (dining, groceries, pharmacy, ATM,
    p2p, ...) are excluded upfront so they can't merge with genuine
    subscription-amount clusters.

    Only outgoing money is considered: refunds share the mcc and amount of
    the charge they reverse, land 1-3 days after it, and would otherwise
    drag the mean gap below the monthly band or inflate gap_cv.
    """
    out = df[df["direction"] == "out"]
    pool = out[
        ~out["type"].isin(NON_SUBSCRIPTION_TYPES)
        & ~out["description"].str.contains("|".join(NON_SUBSCRIPTION_PHRASES))
    ]

    rows = []
    for client_id, group in pool.groupby("client_id"):
        for cluster in _cluster_by_amount(group):
            rows.append({"client_id": client_id, **_stream_stats(cluster)})

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
