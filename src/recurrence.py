"""Detects recurring merchant streams per client from raw transactions.

Detection ported from timmyo. A "stream" is a cluster of one client's
outgoing charges at a similar amount, pooled across *all* mccs and
descriptions, found by 1D amount clustering. Design decisions, each backed
by an audit or a measured experiment:

  - Pool across mccs, label afterwards. The same subscription rotates its
    description month to month ("digital plus" -> "premium plan" -> "media
    streaming") while its mcc alternates (e.g. 5812 / 5411). ~20% of valid
    subscription charges switch mcc (6% in train), so grouping by mcc or by
    a per-row family split streams into fragments that failed n >= 2 or the
    cadence test. Pooling raised c_detect on valid from 0.37 to 0.67
    (timmyo exp 4).
  - Category = majority vote over the members' per-row families
    (category_map.classify, strong / clean-mcc evidence only), which
    imputes a family for the keyword-less cycles of a rotating description. category_agreement / n_keyword_hits
    record how clear the vote was.
  - Only outgoing money defines the stream. Refunds share the amount of the
    charge they reverse and land 1-3 days after it; counted as occurrences
    they dragged the gap below the monthly band. They are attached to the
    nearest-amount stream afterwards as features only.
  - Known non-subscription merchants (dining, groceries, ATM, p2p, ...) are
    dropped before clustering (substring match, any mcc) so they can't merge
    with subscription-amount clusters.
  - Tight amount tolerance, max(0.3, 0.03 * running mean): once all mccs
    share a pool, Salim's former 25% tolerance merged neighbouring streams
    (e.g. cloud ~9 and music ~11). On train 0.5/0.05 .. 0.2/0.02 detect the
    same streams; 0.1/0.01 starts splitting real streams.
  - Monthly cadence only (mean gap 20-45 days, gap_cv <= 0.5). Adding
    weekly..annual bands on top of this detection was negative in a shared
    benchmark (LogReg -0.02, valid ensemble -0.03).
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from src.category_map import POOL_EXCLUDE_PHRASES, POOL_EXCLUDE_TYPES, TARGET_CATEGORIES, classify

MIN_GAP_DAYS = 20
MAX_GAP_DAYS = 45
MAX_GAP_CV = 0.5
MAX_AMOUNT_CV = 0.35

# Amount-clustering tolerance: a new cluster starts when the next amount
# (sorted ascending) exceeds the running cluster mean by more than
# max(AMOUNT_ABS_TOL, AMOUNT_REL_TOL * mean). Also used to match refunds.
AMOUNT_REL_TOL = 0.03
AMOUNT_ABS_TOL = 0.3


def load_transactions(path: str) -> pd.DataFrame:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame.from_records(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Per-row vote: Salim's classify() (canonical phrases, abbreviation
    # expansion, affix stripping). Only family-specific ("strong") and
    # clean-mcc ("mcc") evidence votes; generic / music-or-streaming rows
    # get their family from their cluster siblings.
    classified = [classify(mcc, desc) for mcc, desc in zip(df["mcc"], df["description"])]
    df["category"] = [
        cat if strength in ("strong", "mcc") and cat in TARGET_CATEGORIES else None
        for cat, strength in classified
    ]
    df["is_refund"] = df["type"] == "refund"
    return df


def _tol(mean: float) -> float:
    return max(AMOUNT_ABS_TOL, AMOUNT_REL_TOL * mean)


def _cluster_by_amount(group: pd.DataFrame) -> list[pd.DataFrame]:
    """Greedy 1D clustering of one client's candidate rows by amount."""
    ordered = group.sort_values("amount")
    clusters: list[list] = []
    current: list = []
    total = 0.0
    for idx, amount in zip(ordered.index, ordered["amount"]):
        if current and amount - total / len(current) > _tol(total / len(current)):
            clusters.append(current)
            current, total = [], 0.0
        current.append(idx)
        total += amount
    if current:
        clusters.append(current)
    return [group.loc[idx] for idx in clusters]


def _stream_stats(cluster: pd.DataFrame, refunds: pd.DataFrame, cutoff: pd.Timestamp | None) -> dict:
    ordered = cluster.sort_values("timestamp")
    dates = ordered["timestamp"]
    amounts = ordered["amount"].to_numpy()
    n = len(cluster)

    gaps = dates.diff().dt.days.dropna().to_numpy()
    mean_gap = float(np.mean(gaps)) if len(gaps) else np.nan
    median_gap = float(np.median(gaps)) if len(gaps) else np.nan
    gap_cv = float(np.std(gaps) / mean_gap) if len(gaps) and mean_gap > 0 else np.nan

    mean_amount = float(amounts.mean())
    amount_cv = float(amounts.std(ddof=1) / mean_amount) if n > 1 and mean_amount > 0 else 0.0

    is_recurring = bool(
        n >= 2
        and not np.isnan(mean_gap)
        and MIN_GAP_DAYS <= mean_gap <= MAX_GAP_DAYS
        and (np.isnan(gap_cv) or gap_cv <= MAX_GAP_CV)
        and amount_cv <= MAX_AMOUNT_CV
    )

    votes = Counter(c for c in cluster["category"] if pd.notna(c))
    top = votes.most_common(1)
    n_votes = sum(votes.values())

    last_charge = dates.iloc[-1]
    last_refund = refunds["timestamp"].max() if len(refunds) else pd.NaT
    stats = {
        "n_occurrences": n,
        "first_date": dates.iloc[0],
        "last_date": last_charge,
        "mean_amount": mean_amount,
        "amount_cv": amount_cv,
        "amount_trend": float(amounts[-1] / amounts[0]) if amounts[0] > 0 else 1.0,
        "mean_gap_days": mean_gap,
        "median_gap_days": median_gap,
        "gap_cv": gap_cv,
        "is_recurring": is_recurring,
        "category": top[0][0] if top else None,
        "category_agreement": top[0][1] / n_votes if top else np.nan,
        "n_keyword_hits": n_votes,
        "n_refunds": len(refunds),
        "refund_ratio": len(refunds) / n,
        "last_event_is_refund": int(pd.notna(last_refund) and last_refund >= last_charge),
        "n_mccs": cluster["mcc"].nunique(),
    }
    if cutoff is not None:
        for window in (30, 60, 90):
            stats[f"n_last_{window}d"] = int((dates > cutoff - pd.Timedelta(days=window)).sum())
    return stats


def _assign_refunds(clusters: list[pd.DataFrame], refunds: pd.DataFrame) -> list[list]:
    """Refund rows -> nearest-amount cluster within the clustering tolerance."""
    means = np.array([c["amount"].mean() for c in clusters])
    assigned: list[list] = [[] for _ in clusters]
    for idx, amount in zip(refunds.index, refunds["amount"]):
        d = np.abs(means - amount)
        k = int(d.argmin())
        if d[k] <= _tol(means[k]):
            assigned[k].append(idx)
    return assigned


def detect_streams(df: pd.DataFrame, cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per amount cluster of a client's candidate subscription charges."""
    non_sub = df["description"].str.contains("|".join(POOL_EXCLUDE_PHRASES))
    pool = df[(df["direction"] == "out") & ~df["type"].isin(POOL_EXCLUDE_TYPES) & ~non_sub]
    # Refunds of dining/grocery/... purchases must not land on a
    # subscription cluster by amount coincidence.
    refunds_by_client = dict(tuple(df[df["is_refund"] & ~non_sub].groupby("client_id")))

    rows = []
    for client_id, group in pool.groupby("client_id"):
        clusters = _cluster_by_amount(group)
        client_refunds = refunds_by_client.get(client_id, df.iloc[:0])
        for cluster, ref_idx in zip(clusters, _assign_refunds(clusters, client_refunds)):
            refunds = client_refunds.loc[ref_idx]
            rows.append({"client_id": client_id, **_stream_stats(cluster, refunds, cutoff)})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/dataset/dataset/train_transactions.jsonl"
    df = load_transactions(path)
    streams = detect_streams(df, pd.Timestamp("2026-01-01", tz="UTC"))

    print(f"Loaded {len(df):,} transactions for {df['client_id'].nunique():,} clients")
    print(f"Amount clusters: {len(streams):,}   recurring: {streams['is_recurring'].sum():,}")
    recurring = streams[streams["is_recurring"]]
    print(recurring["category"].value_counts(dropna=False))
