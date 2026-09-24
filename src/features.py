"""Builds the client-level feature table used for modeling.

One row per client, built purely from transactions up to the cutoff date
(no leakage). Combines three groups of signal:

  1. Per-category state, from recurring streams (src.recurrence): does the
     client already have an active recurring subscription in each of the 7
     target categories, and what does that stream look like (recency,
     count, amount, cadence regularity).
  2. Early-adoption signal: transactions in a category within the last 90
     days that DON'T yet meet the >=2-occurrence bar to count as
     "recurring". Important because ~65% of non-'none' validation targets
     are in a category with no active detected stream at cutoff. Most of
     those (~350 of ~460) do have recent transactions in that category, so
     they are largely existing subscriptions the stream detector misses
     (see CADENCE_BANDS in recurrence.py), not true new adoptions. Recent
     category activity is the signal that recovers them.
  3. General financial behavior: tenure, transaction mix, income
     regularity, adoption pace - context features that don't tie to one
     category but describe the client's overall situation (e.g. clients
     with many active categories are less likely to add another - the
     "saturation" effect found during exploration).
"""

from __future__ import annotations

import pathlib
import pickle

import numpy as np
import pandas as pd

from src.category_map import TARGET_CATEGORIES
from src.recurrence import CADENCE_CODE, detect_streams, load_transactions
from src import personas

PERSONA_NAMES = list(personas.PERSONA_RULES)
DATA_DIR = "data/dataset/dataset"
PROCESSED_DIR = "data/processed"
CUTOFF = "2026-01-01"
RECENT_WINDOW_DAYS = 90
STALE_GAP_MULTIPLE = 2.0

# Raw persona-module features kept as model inputs: the ones whose permutation
# importance on valid macro-F1 clearly exceeded noise. The rest (salary,
# savings, FX, timing, ...) showed zero or negative importance.
PERSONA_RAW_FEATURES = [
    "mcc_gym", "mcc_insurance", "mcc_software", "mcc_dining",
    "mcc_atm", "mcc_telecom", "tx_per_month", "share_refund",
]
# The 10 rule-based persona scores (+ top margin) average strong and useless
# features together and added ~nothing in permutation importance.
USE_PERSONA_SCORES = False


def _persona_features(df, cutoff, scorer=None):
    hist = personas.augment(df[df["timestamp"] <= cutoff])
    X = personas.build_features(hist)
    out = X[PERSONA_RAW_FEATURES].add_prefix("persona_")
    if not USE_PERSONA_SCORES:
        return out, None
    if scorer is None:
        scorer = personas.PersonaScorer().fit(X)
    S = scorer.scores(X)                      # one column per persona, 0..1
    S.columns = [f"persona_{c.lower().replace(' / ', '_').replace(' ', '_').replace('-', '_')}"
                 for c in S.columns]
    top2 = np.sort(S.values, axis=1)[:, -2:]
    S["persona_top_margin"] = top2[:, 1] - top2[:, 0]
    return out.join(S), scorer



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


STREAM_FEATURES = [
    "n_streams", "n_occurrences", "recency_days", "tenure_days", "mean_amount", "gap_cv",
    "median_gap_days", "days_to_next", "due_rank", "days_to_next_vs_min", "is_soonest",
    "n_last_30d", "n_last_60d", "n_last_90d", "n_refunds", "refund_ratio",
    "last_event_is_refund", "amount_trend", "cadence_code", "n_mccs",
]


def recurring_streams(streams: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Recurring target-family streams with recency, days_to_next and is_live.

    A recurring stream only counts as live if it is still charging at the
    cutoff: silent for more than STALE_GAP_MULTIPLE of its own cadence
    means it was most likely cancelled. Lapsed streams are kept as a
    separate flag - "had it, dropped it" is a different state from "never
    had it". Shared with the step-C diagnostics in src.evaluate.
    """
    recurring = streams[streams["is_recurring"] & streams["category"].isin(TARGET_CATEGORIES)].copy()
    recurring["recency_days"] = (cutoff - recurring["last_date"]).dt.total_seconds() / 86400
    recurring["days_to_next"] = recurring["median_gap_days"] - recurring["recency_days"]
    recurring["is_live"] = recurring["recency_days"] <= STALE_GAP_MULTIPLE * recurring["median_gap_days"] + 5
    return recurring


def _category_stream_features(streams: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    recurring = recurring_streams(streams, cutoff)
    recurring["cadence_code"] = recurring["cadence"].map(CADENCE_CODE)
    is_live = recurring["is_live"]
    active = recurring[is_live]
    lapsed = recurring[~is_live]

    # One row per (client, category): the stream due soonest represents it.
    active = active.sort_values("days_to_next")
    agg = active.groupby(["client_id", "category"]).agg(
        n_streams=("category", "size"),
        n_occurrences=("n_occurrences", "sum"),
        first_date=("first_date", "min"),
        **{c: (c, "first") for c in [
            "recency_days", "mean_amount", "gap_cv", "median_gap_days", "days_to_next",
            "n_last_30d", "n_last_60d", "n_last_90d", "n_refunds", "refund_ratio",
            "last_event_is_refund", "amount_trend", "cadence_code", "n_mccs",
        ]},
    ).reset_index()
    agg["tenure_days"] = (cutoff - agg["first_date"]).dt.days

    # Cross-category context: trees can't easily learn an argmin over seven
    # separate days_to_next columns, so rank the client's live categories.
    by_client = agg.groupby("client_id")["days_to_next"]
    agg["due_rank"] = by_client.rank(method="first")
    agg["days_to_next_vs_min"] = agg["days_to_next"] - by_client.transform("min")
    agg["is_soonest"] = (agg["due_rank"] == 1).astype(int)

    wide_frames = []
    for cat in TARGET_CATEGORIES:
        sub = agg[agg["category"] == cat].set_index("client_id")[STREAM_FEATURES]
        sub.columns = [f"{c}_{cat}" for c in sub.columns]
        sub[f"active_{cat}"] = 1
        wide_frames.append(sub)

    lapsed_cats = lapsed.groupby(["client_id", "category"]).size().unstack()
    wide_frames.append(
        lapsed_cats.reindex(columns=TARGET_CATEGORIES).notna().astype(int).add_prefix("lapsed_")
    )

    wide = pd.concat(wide_frames, axis=1)
    for cat in TARGET_CATEGORIES:
        wide[f"active_{cat}"] = wide[f"active_{cat}"].fillna(0).astype(int)
        wide[f"lapsed_{cat}"] = wide[f"lapsed_{cat}"].fillna(0).astype(int)

    wide["n_active_categories"] = wide[[f"active_{c}" for c in TARGET_CATEGORIES]].sum(axis=1)
    wide["n_missing_categories"] = len(TARGET_CATEGORIES) - wide["n_active_categories"]

    tenure_cols = [f"tenure_days_{c}" for c in TARGET_CATEGORIES]
    wide["days_since_last_new_category"] = wide[tenure_cols].min(axis=1)
    due_cols = [f"days_to_next_{c}" for c in TARGET_CATEGORIES]
    wide["min_days_to_next"] = wide[due_cols].min(axis=1)
    wide["n_overdue_streams"] = (wide[due_cols] < 0).sum(axis=1)
    wide["n_live_streams"] = active.groupby("client_id").size().reindex(wide.index).fillna(0)

    return wide


def _early_adoption_features(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    recent = df[
        (df["category"].isin(TARGET_CATEGORIES))
        & (~df["is_refund"])
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


def build_features(
    transactions_path: str,
    cutoff_date: str,
    persona_scorer: "personas.PersonaScorer | None" = None,
) -> pd.DataFrame:
    df = load_transactions(transactions_path)
    cutoff = pd.Timestamp(cutoff_date, tz="UTC")

    streams = detect_streams(df, cutoff)

    general = _general_financial_features(df, cutoff)
    cat_features = _category_stream_features(streams, cutoff)
    recent = _early_adoption_features(df, cutoff)

    features = general.join(cat_features, how="left").join(recent, how="left")

    persona_cols, fitted = _persona_features(df, cutoff, persona_scorer)
    features = features.join(persona_cols, how="left")
    features[persona_cols.columns] = features[persona_cols.columns].fillna(0.0)

    active_cols = [f"active_{c}" for c in TARGET_CATEGORIES]
    features[active_cols] = features[active_cols].fillna(0).astype(int)
    lapsed_cols = [f"lapsed_{c}" for c in TARGET_CATEGORIES]
    features[lapsed_cols] = features[lapsed_cols].fillna(0).astype(int)
    features["n_active_categories"] = features["n_active_categories"].fillna(0)
    features["n_missing_categories"] = features["n_missing_categories"].fillna(len(TARGET_CATEGORIES))
    recent_cols = [f"recent_txns_{c}" for c in TARGET_CATEGORIES]
    features[recent_cols] = features[recent_cols].fillna(0).astype(int)

    features["adoption_rate_per_year"] = features["n_active_categories"] / (
        features["tenure_days"] / 365
    ).clip(lower=1 / 365)

    out = features.reset_index()
    out.attrs["persona_scorer"] = fitted
    out.attrs["streams"] = streams
    return out


def build_all(
    data_dir: str = DATA_DIR, cutoff: str = CUTOFF, out_dir: str = PROCESSED_DIR
) -> dict[str, pd.DataFrame]:
    """Rebuild train/valid/test features from raw jsonl and write them to out_dir.

    The persona scorer is fit on train only, then reused so valid/test
    percentiles are ranked against the same reference population. Returns
    the labelled feature tables plus the valid streams (for diagnostics)
    under key "valid_streams".
    """
    data_dir, out_dir = pathlib.Path(data_dir), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = build_features(str(data_dir / "train_transactions.jsonl"), cutoff)
    scorer = train.attrs["persona_scorer"]
    splits = {
        "train": train,
        "valid": build_features(str(data_dir / "valid_transactions.jsonl"), cutoff, scorer),
        "test": build_features(str(data_dir / "test_transactions.jsonl"), cutoff, scorer),
    }
    result = {"valid_streams": splits["valid"].attrs["streams"]}

    for name, feats in splits.items():
        labels_path = data_dir / f"{name}_labels.csv"
        if labels_path.exists():
            labels = pd.read_csv(labels_path)[["client_id", "target_next_recurring_merchant"]]
            feats = feats.merge(labels, on="client_id", how="inner")
        feats.attrs = {}
        feats.to_csv(out_dir / f"{name}_features.csv", index=False)
        result[name] = feats
        print(f"{name}: {feats.shape}")

    if scorer is not None:
        with open(out_dir / "persona_scorer.pkl", "wb") as f:
            pickle.dump(scorer, f)
    return result


if __name__ == "__main__":
    import sys

    build_all(*sys.argv[1:3])
