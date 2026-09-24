"""Detects recurring merchant streams per client from raw transactions.

A "stream" is a cluster of one client's subscription charges in the same
merchant family at a similar amount. Design decisions, each backed by an
audit of the train data:

  - Streams are grouped by *category*, not mcc: ~5% of subscription charges
    carry an unrelated mcc, and grouping by mcc split those streams.
  - Only outgoing charges define the stream. Refunds (~8% of subscription
    rows) used to be counted as occurrences, which broke the gap statistics
    (e.g. a monthly charge + refund 3 days later looked like a 15-day
    cadence and fell outside every band). Refunds are kept as features.
  - Generic descriptions ("digital plus", "monthly plan", ...) are attached
    to the client's strong-evidence cluster with the nearest amount; this
    also resolves 5812's music-vs-streaming ambiguity, since both families
    share "digital plus" / "premium plan".
  - Cadence uses the median gap and includes weekly / biweekly bands:
    ~10% of streams bill every ~14 days.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from src.category_map import MEDIA, TARGET_CATEGORIES, classify

CADENCE_BANDS = {
    "weekly": (5, 9),
    "biweekly": (9, 20),
    "monthly": (20, 45),
    "bimonthly": (45, 75),
    "quarterly": (75, 105),
    "semiannual": (165, 200),
    "annual": (330, 400),
}
CADENCE_CODE = {name: i for i, name in enumerate(CADENCE_BANDS)}
MAX_GAP_CV = 0.6
MAX_AMOUNT_CV = 0.35

# Greedy 1D amount clustering: a new cluster starts when the next amount
# (sorted ascending) exceeds the running mean by this relative/absolute
# margin. Loose enough for gradual price increases (e.g. 74 -> 95).
AMOUNT_REL_TOL = 0.25
AMOUNT_ABS_TOL = 3.0
# Generic-description rows join a strong cluster only when this close.
ATTACH_REL_TOL = 0.12
ATTACH_ABS_TOL = 1.5


def load_transactions(path: str) -> pd.DataFrame:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame.from_records(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    classified = [classify(mcc, desc) for mcc, desc in zip(df["mcc"], df["description"])]
    df["category"] = [c for c, _ in classified]
    df["cat_strength"] = [s for _, s in classified]
    df["is_refund"] = df["type"] == "refund"
    return df


def _cluster_by_amount(amounts: pd.Series) -> list[list]:
    """Greedy 1D clustering; returns lists of index labels."""
    ordered = amounts.sort_values()
    clusters, current, total = [], [], 0.0
    for idx, amount in ordered.items():
        if current:
            mean = total / len(current)
            if amount - mean > max(AMOUNT_ABS_TOL, AMOUNT_REL_TOL * mean):
                clusters.append(current)
                current, total = [], 0.0
        current.append(idx)
        total += amount
    if current:
        clusters.append(current)
    return clusters


def _cadence(gap: float) -> str | None:
    for name, (lo, hi) in CADENCE_BANDS.items():
        if lo <= gap < hi:
            return name
    return None


def _stream_stats(charges: pd.DataFrame, refunds: pd.DataFrame, cutoff: pd.Timestamp | None) -> dict:
    dates = charges["timestamp"].sort_values()
    amounts = charges.sort_values("timestamp")["amount"].to_numpy()
    n = len(charges)

    gaps = dates.diff().dt.total_seconds().dropna().to_numpy() / 86400
    median_gap = float(np.median(gaps)) if len(gaps) else np.nan
    mean_gap = float(np.mean(gaps)) if len(gaps) else np.nan
    gap_cv = float(np.std(gaps) / mean_gap) if len(gaps) > 1 and mean_gap > 0 else np.nan

    mean_amount = float(amounts.mean())
    amount_cv = float(amounts.std() / mean_amount) if n > 1 and mean_amount > 0 else 0.0
    cadence = _cadence(median_gap) if not np.isnan(median_gap) else None

    is_recurring = bool(
        n >= 2
        and cadence is not None
        and (np.isnan(gap_cv) or gap_cv <= MAX_GAP_CV)
        and amount_cv <= MAX_AMOUNT_CV
    )

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
        "cadence": cadence,
        "is_recurring": is_recurring,
        "n_refunds": len(refunds),
        "refund_ratio": len(refunds) / n,
        "last_event_is_refund": int(pd.notna(last_refund) and last_refund >= last_charge),
        "n_strong": int((charges["cat_strength"] == "strong").sum()),
        "n_mccs": charges["mcc"].nunique(),
    }
    if cutoff is not None:
        for window in (30, 60, 90):
            stats[f"n_last_{window}d"] = int((dates > cutoff - pd.Timedelta(days=window)).sum())
    return stats


def _client_streams(g: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Split one client's subscription rows into (category, rows) clusters."""
    strong = g[g["cat_strength"].isin(["strong", "mcc"]) & g["category"].isin(TARGET_CATEGORIES)]
    weak = g.drop(strong.index)

    # Strong-evidence clusters per category; "mcc"-strength rows are
    # provisional and can be re-homed below if their amount says otherwise.
    anchors = strong[strong["cat_strength"] == "strong"]
    clusters: list[tuple[str, list]] = []
    for cat, cg in anchors.groupby("category"):
        charges = cg[~cg["is_refund"]]
        for idx in _cluster_by_amount(charges["amount"] if len(charges) else cg["amount"]):
            clusters.append((cat, list(idx)))
    centers = [(cat, g.loc[idx, "amount"].median()) for cat, idx in clusters]

    def nearest(amount: float, allowed: set[str] | None, prefer: str | None) -> int | None:
        best, best_d = None, None
        for i, (cat, center) in enumerate(centers):
            if allowed is not None and cat not in allowed:
                continue
            d = abs(amount - center)
            if d > max(ATTACH_ABS_TOL, ATTACH_REL_TOL * center):
                continue
            # same-category match wins ties within tolerance
            key = (cat != prefer, d)
            if best_d is None or key < best_d:
                best, best_d = i, key
        return best

    leftovers: dict[str, list] = {}
    for idx, row in pd.concat([strong[strong["cat_strength"] == "mcc"], weak]).iterrows():
        cat = row["category"]
        allowed = {"music", "streaming"} if cat == MEDIA else None
        hit = nearest(row["amount"], allowed, cat)
        if hit is not None:
            clusters[hit][1].append(idx)
        elif cat in TARGET_CATEGORIES:
            leftovers.setdefault(cat, []).append(idx)

    # Generic rows on a clean mcc with no strong sibling still form streams
    # of their own (e.g. a mobile plan billed only as "monthly plan").
    for cat, idx_list in leftovers.items():
        sub = g.loc[idx_list]
        for idx in _cluster_by_amount(sub["amount"]):
            clusters.append((cat, list(idx)))

    return [(cat, g.loc[idx]) for cat, idx in clusters]


def detect_streams(df: pd.DataFrame, cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per (client, category, amount-cluster) subscription stream."""
    pool = df[df["cat_strength"].notna() & (df["category"].notna())]
    pool = pool[(pool["direction"] == "out") | pool["is_refund"]]

    rows = []
    for client_id, g in pool.groupby("client_id"):
        for cat, rows_df in _client_streams(g):
            charges = rows_df[~rows_df["is_refund"]]
            if charges.empty:
                continue
            refunds = rows_df[rows_df["is_refund"]]
            rows.append({"client_id": client_id, "category": cat, **_stream_stats(charges, refunds, cutoff)})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/dataset/dataset/train_transactions.jsonl"
    df = load_transactions(path)
    streams = detect_streams(df, pd.Timestamp("2026-01-01", tz="UTC"))

    print(f"Loaded {len(df):,} transactions for {df['client_id'].nunique():,} clients")
    print(f"Streams: {len(streams):,}   recurring: {streams['is_recurring'].sum():,}")
    recurring = streams[streams["is_recurring"]]
    print(recurring["category"].value_counts())
    print(recurring["cadence"].value_counts())
    check = ["C000005", "C000007", "C000009", "C000012", "C000014", "C000024"]
    cols = ["client_id", "category", "n_occurrences", "mean_amount", "median_gap_days", "cadence", "is_recurring", "last_date"]
    print(streams[streams["client_id"].isin(check)][cols].to_string(index=False))
