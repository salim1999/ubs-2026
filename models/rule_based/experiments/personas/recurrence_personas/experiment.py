"""Recurrence-persona rules for next recurring-payment prediction.

The personas are learned without labels from recurrence behaviour only.  Rule
choices are then selected with train labels and a five-fold stability check.
Validation labels are read only when ``--final-validation`` is explicitly
passed, so exploratory persona work cannot accidentally tune on validation.

Run from the repository root::

    py -3.10 models/rule_based/experiments/personas/recurrence_personas/experiment.py
    py -3.10 models/rule_based/experiments/personas/recurrence_personas/experiment.py --final-validation
"""

from __future__ import annotations

import argparse
import calendar
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.category_map import (  # noqa: E402
    NON_SUBSCRIPTION_PHRASES,
    NON_SUBSCRIPTION_TYPES,
    TARGET_CATEGORIES,
)
from src.model import ALL_LABELS, LABEL_COL  # noqa: E402
from src.recurrence import _cluster_by_amount, _stream_stats, load_transactions  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent
CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
N_PERSONAS = 4
RANDOM_STATE = 31

PROFILE_FEATURES = [
    "stream_count",
    "live_count",
    "category_diversity",
    "live_share",
    "mature_share",
    "regularity",
    "amount_stability",
    "recentness",
]


@dataclass(frozen=True)
class Rule:
    cadence: str
    grace_days: int
    short_gate: bool = True

    @property
    def key(self) -> str:
        return f"{self.cadence}|grace={self.grace_days}|short_gate={int(self.short_gate)}"


GLOBAL_FALLBACK = Rule("median_gap", 5, True)


def macro_f1(y_true: np.ndarray | pd.Series, y_pred: np.ndarray) -> float:
    return float(
        f1_score(y_true, y_pred, labels=ALL_LABELS, average="macro", zero_division=0)
    )


def _calendar_due(last: pd.Timestamp, days_of_month: np.ndarray) -> pd.Timestamp:
    target_day = int(np.rint(np.median(days_of_month)))
    year, month = last.year, last.month + 1
    if month == 13:
        year, month = year + 1, 1
    day = min(target_day, calendar.monthrange(year, month)[1])
    return pd.Timestamp(year=year, month=month, day=day, tz=last.tz)


def build_stream_table(path: Path) -> pd.DataFrame:
    """Create one row per detected, categorized recurring amount stream."""
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
            trimmed = np.sort(gaps)[1:-1] if len(gaps) >= 5 else gaps
            cadences = {
                "mean_gap": float(np.mean(gaps)),
                "median_gap": float(np.median(gaps)),
                "trimmed_mean_gap": float(np.mean(trimmed)),
                "recent3_median_gap": float(np.median(recent)),
            }
            row = {
                "client_id": client_id,
                "category": stats["category"],
                "n_occurrences": int(stats["n_occurrences"]),
                "first_date": stats["first_date"],
                "last_date": stats["last_date"],
                "gap_cv": float(stats["gap_cv"]),
                "amount_cv": float(stats["amount_cv"]),
            }
            for name, gap in cadences.items():
                row[f"due_{name}"] = stats["last_date"] + pd.Timedelta(days=gap)
            row["due_calendar_dom"] = _calendar_due(
                stats["last_date"], dates.dt.day.to_numpy()
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_persona_features(clients: pd.Series, streams: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stream behaviour without using spending totals or labels."""
    work = streams.copy()
    work["recency_days"] = (CUTOFF - work["last_date"]).dt.days.clip(lower=0)
    work["overdue_median"] = work["recency_days"] - (
        (work["due_median_gap"] - work["last_date"]).dt.total_seconds() / 86_400
    )
    work["is_live"] = work["overdue_median"] <= GLOBAL_FALLBACK.grace_days
    work["is_mature"] = work["n_occurrences"] >= 5

    rows = []
    for client_id in clients:
        g = work[work["client_id"] == client_id]
        n = len(g)
        if n == 0:
            rows.append(
                {
                    "client_id": client_id,
                    "stream_count": 0.0,
                    "live_count": 0.0,
                    "category_diversity": 0.0,
                    "live_share": 0.0,
                    "mature_share": 0.0,
                    "regularity": 0.0,
                    "amount_stability": 0.0,
                    "recentness": 0.0,
                }
            )
            continue
        rows.append(
            {
                "client_id": client_id,
                "stream_count": float(n),
                "live_count": float(g["is_live"].sum()),
                "category_diversity": float(g["category"].nunique()),
                "live_share": float(g["is_live"].mean()),
                "mature_share": float(g["is_mature"].mean()),
                # Detector caps gap CV at 0.5 and amount CV at 0.35.
                "regularity": float(1 - np.clip(g["gap_cv"].median() / 0.5, 0, 1)),
                "amount_stability": float(
                    1 - np.clip(g["amount_cv"].median() / 0.35, 0, 1)
                ),
                "recentness": float(np.exp(-g["recency_days"].median() / 90)),
            }
        )
    return pd.DataFrame(rows).set_index("client_id")


def _cluster_input(features: pd.DataFrame) -> pd.DataFrame:
    """Compress counts before scaling so one heavy user cannot dominate."""
    x = features[PROFILE_FEATURES].copy()
    for col in ("stream_count", "live_count", "category_diversity"):
        x[col] = np.log1p(x[col])
    return x


def fit_personas(train_features: pd.DataFrame) -> tuple[StandardScaler, KMeans]:
    scaler = StandardScaler()
    x = scaler.fit_transform(_cluster_input(train_features))
    clusterer = KMeans(
        n_clusters=N_PERSONAS, random_state=RANDOM_STATE, n_init=50
    ).fit(x)
    return scaler, clusterer


def assign_raw_personas(
    features: pd.DataFrame, scaler: StandardScaler, clusterer: KMeans
) -> pd.Series:
    x = scaler.transform(_cluster_input(features))
    return pd.Series(clusterer.predict(x), index=features.index, name="cluster_id")


def name_personas(
    features: pd.DataFrame, raw_personas: pd.Series
) -> tuple[pd.Series, dict[int, str]]:
    """Attach reproducible human names based on relative centroid behaviour."""
    means = features.join(raw_personas).groupby("cluster_id").mean()
    remaining = set(means.index.tolist())
    quiet = int(means["stream_count"].idxmin())
    remaining.remove(quiet)

    # Multi-stream activity is the clearest second archetype.
    active_score = (
        means.loc[list(remaining), "live_count"]
        + means.loc[list(remaining), "category_diversity"]
        + means.loc[list(remaining), "live_share"]
    )
    multi = int(active_score.idxmax())
    remaining.remove(multi)

    # Of the two remaining groups, mature/regular users contrast with short,
    # less settled streams.  This naming is semantic only; it does not affect
    # predictions or parameter selection.
    settled_score = (
        means.loc[list(remaining), "mature_share"]
        + means.loc[list(remaining), "regularity"]
        + means.loc[list(remaining), "amount_stability"]
    )
    settled = int(settled_score.idxmax())
    remaining.remove(settled)
    transitional = int(next(iter(remaining)))

    names = {
        quiet: "Quiet / no detected cycle",
        multi: "Multi-subscription active",
        settled: "Mature regular",
        transitional: "Short or lapsed explorer",
    }
    return raw_personas.map(names).rename("persona"), names


def candidate_rules() -> list[Rule]:
    return [
        Rule(cadence, grace, short_gate)
        for cadence in (
            "median_gap",
            "mean_gap",
            "trimmed_mean_gap",
            "recent3_median_gap",
            "calendar_dom",
        )
        for grace in (0, 3, 5, 7, 10)
        for short_gate in (True, False)
    ]


def predict_one_rule(
    clients: pd.Series, streams: pd.DataFrame, rule: Rule
) -> np.ndarray:
    due_col = f"due_{rule.cadence}"
    work = streams.copy()
    work["days_until_due"] = (
        work[due_col] - CUTOFF
    ).dt.total_seconds() / 86_400
    live = work[work["days_until_due"] >= -rule.grace_days]

    max_n = live.groupby("client_id")["n_occurrences"].max()
    short_clients = set(max_n[max_n.isin([3, 4])].index) if rule.short_gate else set()
    family_due = (
        live.groupby(["client_id", "category"])["days_until_due"].max().reset_index()
    )
    family_due["category_order"] = family_due["category"].map(
        {category: i for i, category in enumerate(TARGET_CATEGORIES)}
    )
    first = (
        family_due.sort_values(["client_id", "days_until_due", "category_order"])
        .groupby("client_id", sort=False)["category"]
        .first()
    )
    return np.asarray(
        ["none" if c in short_clients else first.get(c, "none") for c in clients]
    )


def prediction_cache(clients: pd.Series, streams: pd.DataFrame) -> dict[str, np.ndarray]:
    return {rule.key: predict_one_rule(clients, streams, rule) for rule in candidate_rules()}


def select_persona_rules(
    clients: pd.Series,
    y: pd.Series,
    personas: pd.Series,
    cache: dict[str, np.ndarray],
) -> tuple[dict[str, Rule], pd.DataFrame]:
    """Select stable persona rules using train labels only.

    Starting from the global median-gap fallback, coordinate updates are
    accepted only if they improve mean macro-F1 over five fixed folds and are
    positive on at least three folds.  Tiny gains below 0.0005 are ignored.
    """
    rules = {p: GLOBAL_FALLBACK for p in sorted(personas.unique())}
    client_personas = personas.reindex(clients).to_numpy()
    y_np = y.to_numpy()
    folds = list(
        StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE).split(
            np.zeros(len(y_np)), y_np
        )
    )

    def compose(choice: dict[str, Rule]) -> np.ndarray:
        pred = cache[GLOBAL_FALLBACK.key].copy()
        for persona, rule in choice.items():
            mask = client_personas == persona
            pred[mask] = cache[rule.key][mask]
        return pred

    audit_rows: list[dict] = []
    for iteration in range(2):
        changed = False
        for persona in sorted(rules):
            base_pred = compose(rules)
            base_fold = np.array(
                [macro_f1(y_np[test], base_pred[test]) for _, test in folds]
            )
            best_rule = rules[persona]
            best_delta = 0.0
            best_positive = 0
            for candidate in candidate_rules():
                trial_rules = dict(rules)
                trial_rules[persona] = candidate
                trial_pred = compose(trial_rules)
                fold_scores = np.array(
                    [macro_f1(y_np[test], trial_pred[test]) for _, test in folds]
                )
                deltas = fold_scores - base_fold
                mean_delta = float(deltas.mean())
                positive = int((deltas > 0).sum())
                audit_rows.append(
                    {
                        "iteration": iteration + 1,
                        "persona": persona,
                        **asdict(candidate),
                        "mean_fold_delta": mean_delta,
                        "positive_folds": positive,
                    }
                )
                if (
                    mean_delta > best_delta + 1e-12
                    and positive >= 3
                    and mean_delta >= 0.0005
                ):
                    best_rule = candidate
                    best_delta = mean_delta
                    best_positive = positive
            if best_rule != rules[persona]:
                rules[persona] = best_rule
                changed = True
            audit_rows.append(
                {
                    "iteration": iteration + 1,
                    "persona": persona,
                    **asdict(rules[persona]),
                    "mean_fold_delta": best_delta,
                    "positive_folds": best_positive,
                    "selected": True,
                }
            )
        if not changed:
            break
    return rules, pd.DataFrame(audit_rows)


def predict_persona_rules(
    clients: pd.Series,
    personas: pd.Series,
    rules: dict[str, Rule],
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    # Every unknown or underspecified persona receives the global fallback.
    pred = cache[GLOBAL_FALLBACK.key].copy()
    client_personas = personas.reindex(clients).to_numpy()
    for persona, rule in rules.items():
        mask = client_personas == persona
        pred[mask] = cache.get(rule.key, cache[GLOBAL_FALLBACK.key])[mask]
    return pred


def profile_tables(
    features: pd.DataFrame, personas: pd.Series, rules: dict[str, Rule]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = features.join(personas)
    centroids = joined.groupby("persona")[PROFILE_FEATURES].mean()
    counts = joined["persona"].value_counts().rename("n_clients")
    profiles = centroids.join(counts)
    profiles["share_clients"] = profiles["n_clients"] / len(joined)
    profiles["cadence"] = [rules[p].cadence for p in profiles.index]
    profiles["grace_days"] = [rules[p].grace_days for p in profiles.index]
    profiles["short_gate"] = [rules[p].short_gate for p in profiles.index]

    # Within-feature min/max normalization supports a readable radar chart.
    normalized = centroids.copy()
    for col in normalized:
        lo, hi = normalized[col].min(), normalized[col].max()
        normalized[col] = (normalized[col] - lo) / (hi - lo) if hi > lo else 0.5
    return profiles, normalized


def draw_profiles(profiles: pd.DataFrame, normalized: pd.DataFrame) -> None:
    """Save a presentation-ready radar chart and population bars."""
    labels = [
        "streams",
        "live",
        "category\ndiversity",
        "live\nshare",
        "mature\nshare",
        "regularity",
        "amount\nstability",
        "recentness",
    ]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    closed_angles = angles + angles[:1]
    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1])
    ax = fig.add_subplot(grid[0, 0], polar=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(normalized)))
    for color, (persona, row) in zip(colors, normalized.iterrows()):
        values = row.tolist() + [row.iloc[0]]
        ax.plot(closed_angles, values, linewidth=2.2, label=persona, color=color)
        ax.fill(closed_angles, values, alpha=0.10, color=color)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.set_title("Recurring-payment personas\n(relative profile)", pad=22, fontsize=15)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), frameon=False)

    bx = fig.add_subplot(grid[0, 1])
    ordered = profiles.sort_values("n_clients")
    bx.barh(ordered.index, ordered["n_clients"], color="#24557a")
    bx.set_xlabel("Training clients")
    bx.set_title("Persona size and selected rule", fontsize=15)
    bx.spines[["top", "right"]].set_visible(False)
    for i, (_, row) in enumerate(ordered.iterrows()):
        rule_text = f"  {int(row['n_clients'])}  |  {row['cadence']} +{int(row['grace_days'])}d"
        bx.text(row["n_clients"], i, rule_text, va="center", fontsize=9)
    fig.suptitle(
        "Same transparent due-date logic, personalized by recurrence behaviour",
        fontsize=17,
        fontweight="bold",
    )
    for suffix in ("png", "svg"):
        fig.savefig(OUT_DIR / f"persona_profiles.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final-validation",
        action="store_true",
        help="Read validation labels and perform the one final evaluation.",
    )
    args = parser.parse_args()

    train_labels = pd.read_csv(ROOT / "data/train_labels.csv")
    train_clients = train_labels["client_id"]
    train_streams = build_stream_table(ROOT / "data/train_transactions.jsonl")
    train_features = build_persona_features(train_clients, train_streams)
    scaler, clusterer = fit_personas(train_features)
    raw_train = assign_raw_personas(train_features, scaler, clusterer)
    train_personas, name_map = name_personas(train_features, raw_train)
    train_cache = prediction_cache(train_clients, train_streams)
    rules, audit = select_persona_rules(
        train_clients, train_labels[LABEL_COL], train_personas, train_cache
    )
    train_pred = predict_persona_rules(train_clients, train_personas, rules, train_cache)
    fallback_train = macro_f1(
        train_labels[LABEL_COL], train_cache[GLOBAL_FALLBACK.key]
    )
    persona_train = macro_f1(train_labels[LABEL_COL], train_pred)

    profiles, normalized = profile_tables(train_features, train_personas, rules)
    profiles.to_csv(OUT_DIR / "persona_profiles.csv")
    normalized.to_csv(OUT_DIR / "persona_profiles_normalized.csv")
    audit.to_csv(OUT_DIR / "train_rule_selection.csv", index=False)
    draw_profiles(profiles, normalized)

    config = {
        "n_personas": N_PERSONAS,
        "random_state": RANDOM_STATE,
        "raw_cluster_name_map": {str(k): v for k, v in name_map.items()},
        "global_fallback": asdict(GLOBAL_FALLBACK),
        "persona_rules": {name: asdict(rule) for name, rule in rules.items()},
        "train_global_fallback_macro_f1": fallback_train,
        "train_persona_macro_f1": persona_train,
        "validation_was_read": bool(args.final_validation),
    }

    if args.final_validation:
        # This block is deliberately the only validation-label read in the file.
        valid_labels = pd.read_csv(ROOT / "data/valid_labels.csv")
        valid_clients = valid_labels["client_id"]
        valid_streams = build_stream_table(ROOT / "data/valid_transactions.jsonl")
        valid_features = build_persona_features(valid_clients, valid_streams)
        raw_valid = assign_raw_personas(valid_features, scaler, clusterer)
        valid_personas = raw_valid.map(name_map).rename("persona")
        valid_cache = prediction_cache(valid_clients, valid_streams)
        valid_pred = predict_persona_rules(
            valid_clients, valid_personas, rules, valid_cache
        )
        persona_valid = macro_f1(valid_labels[LABEL_COL], valid_pred)
        fallback_valid = macro_f1(
            valid_labels[LABEL_COL], valid_cache[GLOBAL_FALLBACK.key]
        )
        config.update(
            {
                "validation_persona_macro_f1": persona_valid,
                "validation_global_fallback_macro_f1": fallback_valid,
                "validation_original_mean_rule_macro_f1": 0.4844204458488582,
                "validation_published_median_rule_macro_f1": 0.49685288345096884,
                "delta_vs_original": persona_valid - 0.4844204458488582,
                "delta_vs_median": persona_valid - 0.49685288345096884,
                "validation_persona_counts": valid_personas.value_counts().to_dict(),
                "validation_per_class_f1": dict(
                    zip(
                        ALL_LABELS,
                        map(
                            float,
                            f1_score(
                                valid_labels[LABEL_COL],
                                valid_pred,
                                labels=ALL_LABELS,
                                average=None,
                                zero_division=0,
                            ),
                        ),
                    )
                ),
            }
        )
        pd.DataFrame(
            {
                "client_id": valid_clients,
                "persona": valid_personas.reindex(valid_clients).to_numpy(),
                "true_label": valid_labels[LABEL_COL],
                "persona_prediction": valid_pred,
                "global_median_prediction": valid_cache[GLOBAL_FALLBACK.key],
            }
        ).to_csv(OUT_DIR / "validation_predictions.csv", index=False)

    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(profiles.to_string(float_format=lambda x: f"{x:.3f}"))
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
