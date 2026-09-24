"""Builds the client-level feature table used for modeling.

One row per client, built purely from transactions up to the cutoff date
(no leakage). Combines three groups of signal:

  1. Per-category state, from recurring streams (src.recurrence): does the
     client already have an active recurring subscription in each of the 7
     target categories, and what does that stream look like (recency,
     count, amount, cadence regularity).
  2. Early-adoption signal: transactions in a category within the last 90
     days that DON'T yet meet the >=2-occurrence bar to count as
     "recurring". Important because ~47% of non-'none' targets are brand
     new categories the client didn't have before cutoff (see recurrence.py
     write-up) - a single recent transaction in a category can be the only
     visible precursor of a subscription that will recur in the 90-day
     prediction window.
  3. General financial behavior: tenure, transaction mix, income
     regularity, adoption pace - context features that don't tie to one
     category but describe the client's overall situation (e.g. clients
     with many active categories are less likely to add another - the
     "saturation" effect found during exploration).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.category_map import TARGET_CATEGORIES
from src.recurrence import detect_streams, load_transactions

RECENT_WINDOW_DAYS = 90
# A stream counts as live if its next charge is overdue by at most this many days.
LIVE_MAX_OVER_DAYS = 5


def _gap_stats(dates: pd.Series) -> tuple[float, float]:
    dates = dates.sort_values()
    gaps = dates.diff().dt.days.dropna().to_numpy()
    if len(gaps) == 0:
        return np.nan, np.nan
    mean_gap = float(np.mean(gaps))
    cv = float(np.std(gaps) / mean_gap) if mean_gap > 0 else np.nan
    return mean_gap, cv


def _general_financial_features(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for client_id, g in df.groupby("client_id"):
        g = g.sort_values("timestamp")
        tenure_days = (cutoff - g["timestamp"].min()).days
        recency_days = (cutoff - g["timestamp"].max()).days

        out = g[g["direction"] == "out"]
        inc = g[g["direction"] == "in"]

        type_counts = g["type"].value_counts()
        n_txns = len(g)

        topup = g[g["type"] == "topup"]
        topup_gap_mean, topup_gap_cv = _gap_stats(topup["timestamp"])

        rows.append(
            {
                "client_id": client_id,
                "tenure_days": tenure_days,
                "recency_days": recency_days,
                "n_txns": n_txns,
                "n_txns_per_month": n_txns / max(tenure_days / 30, 1),
                "total_out": float(out["amount"].sum()),
                "total_in": float(inc["amount"].sum()),
                "net_flow": float(inc["amount"].sum() - out["amount"].sum()),
                "avg_txn_amount": float(g["amount"].mean()),
                "avg_out_amount": float(out["amount"].mean()) if len(out) else np.nan,
                "frac_card_payment": type_counts.get("card_payment", 0) / n_txns,
                "frac_p2p": type_counts.get("p2p_transfer", 0) / n_txns,
                "frac_atm": type_counts.get("atm", 0) / n_txns,
                "frac_fee": type_counts.get("fee", 0) / n_txns,
                "n_topups": len(topup),
                "mean_topup_amount": float(topup["amount"].mean()) if len(topup) else np.nan,
                "topup_gap_mean_days": topup_gap_mean,
                "topup_gap_cv": topup_gap_cv,
                "n_distinct_mcc": g["mcc"].nunique(),
            }
        )
    return pd.DataFrame(rows).set_index("client_id")


def _category_stream_features(streams: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    active = streams[streams["is_recurring"] & streams["category"].notna()].copy()

    agg = (
        active.groupby(["client_id", "category"])
        .agg(
            n_streams=("category", "size"),
            n_occurrences=("n_occurrences", "sum"),
            last_date=("last_date", "max"),
            first_date=("first_date", "min"),
            mean_amount=("mean_amount", "mean"),
            gap_cv=("gap_cv", "mean"),
        )
        .reset_index()
    )
    agg["tenure_days"] = (cutoff - agg["first_date"]).dt.days
    # Timing (recency, mean gap, days until next due) lives in the live-stream
    # features (over_days_, first_due_, bill_day_); the raw versions added
    # nothing on top of them (exp 12, 13).

    wide_frames = []
    for cat in TARGET_CATEGORIES:
        sub = agg[agg["category"] == cat].set_index("client_id")
        sub = sub[
            [
                "n_streams",
                "n_occurrences",
                "tenure_days",
                "mean_amount",
                "gap_cv",
            ]
        ]
        sub.columns = [f"{c}_{cat}" for c in sub.columns]
        sub[f"active_{cat}"] = 1
        wide_frames.append(sub)

    wide = pd.concat(wide_frames, axis=1)
    for cat in TARGET_CATEGORIES:
        wide[f"active_{cat}"] = wide[f"active_{cat}"].fillna(0).astype(int)

    return wide


def _live_stream_features(streams: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Which streams are still running at the cutoff.

    over = days since last charge - mean gap (> 0: next charge is overdue).
    A stream overdue by more than LIVE_MAX_OVER_DAYS has most likely stopped;
    days_until_next_due alone can't show that (negative = "due now" or
    "cancelled months ago"). Streams with only 3-4 charges look like trials
    that end: those clients are mostly 'none' (train and valid).
    """
    active = streams[streams["is_recurring"] & streams["category"].notna()].copy()
    active["over"] = (cutoff - active["last_date"]).dt.days - active["mean_gap_days"]
    live = active[active["over"] <= LIVE_MAX_OVER_DAYS]

    fam = active.groupby(["client_id", "category"])["over"].min().unstack()
    fam = fam.reindex(columns=TARGET_CATEGORIES)
    out = fam.add_prefix("over_days_")
    for cat in TARGET_CATEGORIES:
        out[f"is_live_{cat}"] = (fam[cat] <= LIVE_MAX_OVER_DAYS).astype(int)

    per_client = pd.DataFrame(
        {
            "n_live_streams": live.groupby("client_id").size(),
            "max_live_n_occurrences": live.groupby("client_id")["n_occurrences"].max(),
            "min_over_days": active.groupby("client_id")["over"].min(),
        }
    )
    out = out.join(per_client, how="outer")
    out["n_live_streams"] = out["n_live_streams"].fillna(0)
    out["short_live_stream"] = out["max_live_n_occurrences"].isin([3, 4]).astype(int)

    # "The rule's vote": the live family due first (largest over = closest to
    # its next charge), or 'none' when nothing is live / it's a short trial.
    first = live.sort_values("over", ascending=False).groupby("client_id")["category"].first()
    first = first.reindex(out.index)
    for cat in TARGET_CATEGORIES:
        out[f"first_due_{cat}"] = (first == cat).astype(int)
    out["rule_says_none"] = (first.isna() | (out["short_live_stream"] == 1)).astype(int)
    return out


def _billing_day_features(df: pd.DataFrame, streams: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """"The calendar": usual billing day of month of each live family.

    Subscriptions charge on a fixed day, so with a Jan 1 cutoff the family
    with the earliest billing day is usually charged first.
    """
    active = streams[streams["is_recurring"] & streams["category"].notna()].copy()
    active["over"] = (cutoff - active["last_date"]).dt.days - active["mean_gap_days"]
    live = active[active["over"] <= LIVE_MAX_OVER_DAYS][["client_id", "category"]].drop_duplicates()

    charges = df[df["direction"] == "out"][["client_id", "category", "timestamp"]].dropna()
    charges = charges.merge(live, on=["client_id", "category"])
    day = charges.assign(day=charges["timestamp"].dt.day).groupby(["client_id", "category"])["day"].median()
    out = day.unstack().reindex(columns=TARGET_CATEGORIES).add_prefix("bill_day_")
    out["earliest_bill_day"] = out.min(axis=1)
    return out


def _early_adoption_features(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    recent = df[
        (df["category"].notna())
        & (df["timestamp"] > cutoff - pd.Timedelta(days=RECENT_WINDOW_DAYS))
    ]
    counts = (
        recent.groupby(["client_id", "category"]).size().unstack(fill_value=0)
    )
    for cat in TARGET_CATEGORIES:
        if cat not in counts.columns:
            counts[cat] = 0
    counts = counts[TARGET_CATEGORIES]
    counts.columns = [f"recent_txns_{c}" for c in TARGET_CATEGORIES]
    return counts


def build_features(transactions_path: str, cutoff_date: str) -> pd.DataFrame:
    df = load_transactions(transactions_path)
    cutoff = pd.Timestamp(cutoff_date, tz="UTC")

    streams = detect_streams(df)

    general = _general_financial_features(df, cutoff)
    cat_features = _category_stream_features(streams, cutoff)
    recent = _early_adoption_features(df, cutoff)

    live = _live_stream_features(streams, cutoff)
    billing = _billing_day_features(df, streams, cutoff)

    features = (
        general.join(cat_features, how="left")
        .join(recent, how="left")
        .join(live, how="left")
        .join(billing, how="left")
    )

    active_cols = [f"active_{c}" for c in TARGET_CATEGORIES]
    features[active_cols] = features[active_cols].fillna(0).astype(int)
    recent_cols = [f"recent_txns_{c}" for c in TARGET_CATEGORIES]
    features[recent_cols] = features[recent_cols].fillna(0).astype(int)
    live_flags = (
        [f"is_live_{c}" for c in TARGET_CATEGORIES]
        + [f"first_due_{c}" for c in TARGET_CATEGORIES]
        + ["n_live_streams", "short_live_stream"]
    )
    features[live_flags] = features[live_flags].fillna(0).astype(int)
    # No recurring stream at all -> the rule says none.
    features["rule_says_none"] = features["rule_says_none"].fillna(1).astype(int)

    return features.reset_index()


if __name__ == "__main__":
    import sys

    txn_path = sys.argv[1] if len(sys.argv) > 1 else "data/train_transactions.jsonl"
    cutoff = sys.argv[2] if len(sys.argv) > 2 else "2026-01-01"

    feats = build_features(txn_path, cutoff)
    print(f"Feature table shape: {feats.shape}")
    print(f"Columns ({len(feats.columns)}): {list(feats.columns)}")
    print()
    print(feats.head(3).to_string())
    print()
    print("Missing-value fraction per column (top 15):")
    print((feats.isna().mean().sort_values(ascending=False)).head(15))
