"""Interpretable spender personas for the due-date rule.

Run from the repository root::

    py -3.10 models/rule_based/experiments/personas/spending_personas/experiment.py

The experiment deliberately keeps the predictive layer rule based. K-means is
used only to create four descriptive client personas from transaction behavior.
Each persona may choose a cadence estimator, liveness grace, and short-stream
gate using train-only cross-validation. Small or unstable groups retain the
global median-gap rule. Validation is loaded only after every choice is locked.
"""

from __future__ import annotations

import calendar
import json
import math
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, silhouette_score
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
from src.features import build_features  # noqa: E402
from src.model import ALL_LABELS, LABEL_COL  # noqa: E402
from src.recurrence import _cluster_by_amount, _stream_stats, load_transactions  # noqa: E402


CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
OUT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
N_PERSONAS = 4
GLOBAL_CONFIG = ("median_gap", 5, True)
MIN_PERSONA_SIZE = 200
SHRINKAGE_STRENGTH = 300
MIN_SHRUNK_CV_DELTA = 0.003
MIN_FOLDS_WON = 3

PERSONA_FEATURES = [
    "activity",
    "outflow",
    "inflow",
    "card_use",
    "cash_p2p",
    "breadth",
    "subscriptions",
]
DISPLAY_LABELS = {
    "activity": "Activity",
    "outflow": "Outflow",
    "inflow": "Inflow",
    "card_use": "Card use",
    "cash_p2p": "Cash + P2P",
    "breadth": "Spend breadth",
    "subscriptions": "Subscriptions",
}


def macro_f1(y_true: np.ndarray | pd.Series, y_pred: np.ndarray) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=ALL_LABELS,
            average="macro",
            zero_division=0,
        )
    )


def persona_features(features: pd.DataFrame) -> pd.DataFrame:
    """Convert raw client features into seven intuitive behavior dimensions."""
    months = (features["tenure_days"] / 30).clip(lower=1)
    frame = pd.DataFrame(index=features["client_id"])
    frame["activity"] = np.log1p(features["n_txns_per_month"].to_numpy())
    frame["outflow"] = np.log1p((features["total_out"] / months).to_numpy())
    frame["inflow"] = np.log1p((features["total_in"] / months).to_numpy())
    frame["card_use"] = features["frac_card_payment"].fillna(0).to_numpy()
    frame["cash_p2p"] = (
        features["frac_atm"].fillna(0) + features["frac_p2p"].fillna(0)
    ).to_numpy()
    frame["breadth"] = np.log1p(features["n_distinct_mcc"].to_numpy())
    frame["subscriptions"] = features["n_active_categories"].fillna(0).to_numpy()
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0)


def name_personas(centroids: pd.DataFrame) -> dict[int, str]:
    """Assign unique, stable business names from standardized centroids."""
    remaining = set(centroids.index)
    names: dict[int, str] = {}
    semantic_scores = [
        (
            "High-velocity spenders",
            centroids["activity"] + centroids["outflow"] + 0.4 * centroids["breadth"],
        ),
        (
            "Subscription regulars",
            centroids["subscriptions"] + 0.5 * centroids["card_use"],
        ),
        (
            "Transfer & cash movers",
            centroids["cash_p2p"] + 0.4 * centroids["inflow"],
        ),
        (
            "Low-key everyday users",
            -(centroids["activity"] + centroids["outflow"]),
        ),
    ]
    for name, scores in semantic_scores:
        cluster = max(remaining, key=lambda c: (float(scores.loc[c]), -int(c)))
        names[int(cluster)] = name
        remaining.remove(cluster)
    return names


def fit_personas(
    train_behavior: pd.DataFrame,
) -> tuple[StandardScaler, KMeans, dict[int, str], pd.DataFrame]:
    scaler = StandardScaler()
    z = scaler.fit_transform(train_behavior[PERSONA_FEATURES])
    kmeans = KMeans(n_clusters=N_PERSONAS, n_init=50, random_state=RANDOM_STATE)
    clusters = kmeans.fit_predict(z)
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=PERSONA_FEATURES)
    names = name_personas(centroids)
    assignment = pd.DataFrame(
        {
            "client_id": train_behavior.index,
            "cluster": clusters,
            "persona": [names[int(c)] for c in clusters],
        }
    )
    return scaler, kmeans, names, assignment


def assign_personas(
    behavior: pd.DataFrame,
    scaler: StandardScaler,
    kmeans: KMeans,
    names: dict[int, str],
) -> pd.DataFrame:
    clusters = kmeans.predict(scaler.transform(behavior[PERSONA_FEATURES]))
    return pd.DataFrame(
        {
            "client_id": behavior.index,
            "cluster": clusters,
            "persona": [names[int(c)] for c in clusters],
        }
    )


def _calendar_due(last: pd.Timestamp, days_of_month: np.ndarray) -> pd.Timestamp:
    target_day = int(np.rint(np.median(days_of_month)))
    year, month = last.year, last.month + 1
    if month == 13:
        year, month = year + 1, 1
    day = min(target_day, calendar.monthrange(year, month)[1])
    return pd.Timestamp(year=year, month=month, day=day, tz=last.tz)


def build_stream_table(path: Path) -> pd.DataFrame:
    """Build recurring streams and six transparent due-date forecasts."""
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
            for estimator, gap in cadence.items():
                row[f"due_{estimator}"] = stats["last_date"] + pd.Timedelta(days=gap)
            row["due_calendar_dom"] = _calendar_due(
                stats["last_date"], dates.dt.day.to_numpy()
            )
            rows.append(row)
    return pd.DataFrame(rows)


def rule_predict(
    clients: pd.Series,
    streams: pd.DataFrame,
    estimator: str,
    grace_days: int,
    short_gate: bool,
) -> np.ndarray:
    """Apply one due-date configuration to a client list."""
    work = streams.copy()
    work["days_until_due"] = (
        work[f"due_{estimator}"] - CUTOFF
    ).dt.total_seconds() / 86_400
    live = work[work["days_until_due"] >= -grace_days]
    max_n = live.groupby("client_id")["n_occurrences"].max()
    short_clients = set(max_n[max_n.isin([3, 4])].index) if short_gate else set()
    family_due = (
        live.groupby(["client_id", "category"])["days_until_due"]
        .max()
        .reset_index()
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
        [
            "none" if client in short_clients else first.get(client, "none")
            for client in clients
        ]
    )


def candidate_configs() -> list[tuple[str, int, bool]]:
    estimators = [
        "median_gap",
        "mean_gap",
        "trimmed_mean_gap",
        "recent3_median_gap",
        "mean_median_blend",
        "calendar_dom",
    ]
    return [
        (estimator, grace, short_gate)
        for estimator in estimators
        for grace in (0, 3, 5, 7, 10)
        for short_gate in (True, False)
    ]


def config_name(config: tuple[str, int, bool]) -> str:
    estimator, grace, short_gate = config
    return f"{estimator}|grace={grace}|short_gate={str(short_gate).lower()}"


def all_config_predictions(
    clients: pd.Series, streams: pd.DataFrame
) -> dict[tuple[str, int, bool], np.ndarray]:
    return {
        config: rule_predict(clients, streams, *config)
        for config in candidate_configs()
    }


def select_persona_configs(
    labels: pd.DataFrame,
    assignments: pd.DataFrame,
    predictions: dict[tuple[str, int, bool], np.ndarray],
) -> tuple[dict[str, tuple[str, int, bool]], pd.DataFrame]:
    """Select local rules by train-only CV with size/stability shrinkage."""
    personas = assignments.set_index("client_id").loc[labels["client_id"], "persona"]
    y = labels[LABEL_COL].to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(np.zeros(len(labels)), y))
    records: list[dict] = []
    chosen: dict[str, tuple[str, int, bool]] = {}

    for persona in sorted(personas.unique()):
        mask = personas.to_numpy() == persona
        n_clients = int(mask.sum())
        config_fold_scores: dict[tuple[str, int, bool], list[float]] = {}
        for config, pred in predictions.items():
            scores = []
            for _, test_idx in folds:
                idx = test_idx[mask[test_idx]]
                if len(idx):
                    scores.append(macro_f1(y[idx], pred[idx]))
            config_fold_scores[config] = scores

        global_scores = config_fold_scores[GLOBAL_CONFIG]
        global_mean = float(np.mean(global_scores))
        ranked = sorted(
            predictions,
            key=lambda c: (
                -float(np.mean(config_fold_scores[c])),
                0 if c == GLOBAL_CONFIG else 1,
                config_name(c),
            ),
        )
        best = ranked[0]
        best_scores = config_fold_scores[best]
        raw_delta = float(np.mean(best_scores) - global_mean)
        shrinkage_weight = n_clients / (n_clients + SHRINKAGE_STRENGTH)
        shrunk_delta = shrinkage_weight * raw_delta
        folds_won = int(
            np.sum(np.asarray(best_scores) > np.asarray(global_scores) + 1e-12)
        )
        use_local = bool(
            n_clients >= MIN_PERSONA_SIZE
            and shrunk_delta >= MIN_SHRUNK_CV_DELTA
            and folds_won >= MIN_FOLDS_WON
        )
        final = best if use_local else GLOBAL_CONFIG
        chosen[persona] = final

        for config in ranked:
            scores = config_fold_scores[config]
            records.append(
                {
                    "persona": persona,
                    "n_train_clients": n_clients,
                    "config": config_name(config),
                    "cv_macro_f1_mean": float(np.mean(scores)),
                    "cv_macro_f1_std": float(np.std(scores, ddof=1)),
                    "delta_vs_global": float(np.mean(scores) - global_mean),
                    "shrinkage_weight": shrinkage_weight,
                    "shrunk_delta_vs_global": shrinkage_weight
                    * float(np.mean(scores) - global_mean),
                    "folds_beating_global": int(
                        np.sum(np.asarray(scores) > np.asarray(global_scores) + 1e-12)
                    ),
                    "is_raw_best": config == best,
                    "selected_after_guardrails": config == final,
                    "used_global_fallback": not use_local,
                }
            )
    return chosen, pd.DataFrame(records)


def predict_persona_rules(
    clients: pd.Series,
    assignments: pd.DataFrame,
    predictions: dict[tuple[str, int, bool], np.ndarray],
    choices: dict[str, tuple[str, int, bool]],
) -> np.ndarray:
    persona = assignments.set_index("client_id").loc[clients, "persona"].to_numpy()
    output = np.empty(len(clients), dtype=object)
    for name, config in choices.items():
        mask = persona == name
        output[mask] = predictions[config][mask]
    return output.astype(str)


def percentile_profile_cards(
    train_behavior: pd.DataFrame, assignments: pd.DataFrame
) -> pd.DataFrame:
    """Return persona cards on a presentation-friendly 0--100 scale."""
    joined = train_behavior.join(assignments.set_index("client_id")[["persona"]])
    raw = joined.groupby("persona")[PERSONA_FEATURES].mean()
    counts = joined.groupby("persona").size().rename("n_train_clients")
    percentiles = raw.rank(pct=True, axis=0, method="average") * 100
    cards = percentiles.join(counts)
    cards["share_of_train"] = counts / len(joined)
    return cards.reset_index()


def raw_profile_table(
    raw_features: pd.DataFrame, assignments: pd.DataFrame
) -> pd.DataFrame:
    joined = raw_features.set_index("client_id").join(
        assignments.set_index("client_id")[["persona"]]
    )
    months = (joined["tenure_days"] / 30).clip(lower=1)
    joined = joined.assign(
        outflow_per_month=joined["total_out"] / months,
        inflow_per_month=joined["total_in"] / months,
        cash_p2p_share=joined["frac_atm"] + joined["frac_p2p"],
    )
    columns = [
        "n_txns_per_month",
        "outflow_per_month",
        "inflow_per_month",
        "frac_card_payment",
        "cash_p2p_share",
        "n_distinct_mcc",
        "n_active_categories",
        "n_live_streams",
    ]
    out = joined.groupby("persona")[columns].median().reset_index()
    out.insert(1, "n_train_clients", joined.groupby("persona").size().loc[out["persona"]].to_numpy())
    return out


def write_radar_svg(cards: pd.DataFrame, destination: Path) -> None:
    """Write a dependency-free, standalone SVG persona radar chart."""
    width, height = 1100, 720
    cx, cy, radius = 390, 360, 250
    n_axes = len(PERSONA_FEATURES)
    angles = [-math.pi / 2 + 2 * math.pi * i / n_axes for i in range(n_axes)]
    colors = ["#E60028", "#14A098", "#FFB000", "#4464AD"]

    def point(angle: float, r: float) -> tuple[float, float]:
        return cx + r * math.cos(angle), cy + r * math.sin(angle)

    def polygon(values: list[float]) -> str:
        return " ".join(
            f"{point(angle, radius * value / 100)[0]:.1f},{point(angle, radius * value / 100)[1]:.1f}"
            for angle, value in zip(angles, values)
        )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FBFAF8"/>',
        '<text x="48" y="55" font-family="Arial" font-size="30" font-weight="700" fill="#171717">Spender personas</text>',
        '<text x="48" y="85" font-family="Arial" font-size="15" fill="#666">Train-only behavioral clusters · axis values are relative percentiles</text>',
    ]
    for level in (25, 50, 75, 100):
        lines.append(
            f'<polygon points="{polygon([level] * n_axes)}" fill="none" stroke="#D9D5CE" stroke-width="1"/>'
        )
    for angle, feature in zip(angles, PERSONA_FEATURES):
        x, y = point(angle, radius)
        lx, ly = point(angle, radius + 35)
        anchor = "middle" if abs(lx - cx) < 30 else ("start" if lx > cx else "end")
        lines.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#D9D5CE"/>')
        lines.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-family="Arial" font-size="14" fill="#333">{DISPLAY_LABELS[feature]}</text>'
        )
    for i, row in cards.sort_values("persona").reset_index(drop=True).iterrows():
        values = [float(row[f]) for f in PERSONA_FEATURES]
        color = colors[i % len(colors)]
        lines.append(
            f'<polygon points="{polygon(values)}" fill="{color}" fill-opacity="0.10" stroke="{color}" stroke-width="3"/>'
        )
        y = 180 + i * 88
        lines.extend(
            [
                f'<rect x="760" y="{y - 15}" width="22" height="22" rx="4" fill="{color}"/>',
                f'<text x="798" y="{y}" font-family="Arial" font-size="18" font-weight="700" fill="#222">{escape(str(row["persona"]))}</text>',
                f'<text x="798" y="{y + 27}" font-family="Arial" font-size="14" fill="#666">{int(row["n_train_clients"])} clients · {100 * float(row["share_of_train"]):.1f}%</text>',
            ]
        )
    lines.append(
        '<text x="760" y="575" font-family="Arial" font-size="13" fill="#666">Larger radius = more of that behavior</text>'
    )
    lines.append("</svg>")
    destination.write_text("\n".join(lines), encoding="utf-8")


def silhouette_sensitivity(behavior: pd.DataFrame) -> pd.DataFrame:
    z = StandardScaler().fit_transform(behavior[PERSONA_FEATURES])
    rows = []
    for k in (3, 4, 5):
        model = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE)
        clusters = model.fit_predict(z)
        sizes = pd.Series(clusters).value_counts()
        rows.append(
            {
                "n_personas": k,
                "silhouette": float(silhouette_score(z, clusters)),
                "smallest_cluster": int(sizes.min()),
                "largest_cluster": int(sizes.max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    # Phase 1: everything through rule selection uses train data only.
    train_labels = pd.read_csv(ROOT / "data/train_labels.csv")
    train_raw_features = build_features(
        str(ROOT / "data/train_transactions.jsonl"), str(CUTOFF.date())
    )
    train_behavior = persona_features(train_raw_features)
    scaler, kmeans, persona_names, train_assignments = fit_personas(train_behavior)
    train_streams = build_stream_table(ROOT / "data/train_transactions.jsonl")
    train_predictions = all_config_predictions(train_labels["client_id"], train_streams)
    choices, selection = select_persona_configs(
        train_labels, train_assignments, train_predictions
    )
    persona_train_pred = predict_persona_rules(
        train_labels["client_id"], train_assignments, train_predictions, choices
    )
    global_train_pred = train_predictions[GLOBAL_CONFIG]

    cards = percentile_profile_cards(train_behavior, train_assignments)
    raw_profiles = raw_profile_table(train_raw_features, train_assignments)
    sensitivity = silhouette_sensitivity(train_behavior)
    cards.to_csv(OUT_DIR / "persona_profile_cards.csv", index=False)
    raw_profiles.to_csv(OUT_DIR / "persona_raw_profiles.csv", index=False)
    train_assignments.to_csv(OUT_DIR / "train_persona_assignments.csv", index=False)
    selection.to_csv(OUT_DIR / "train_rule_selection.csv", index=False)
    sensitivity.to_csv(OUT_DIR / "cluster_sensitivity.csv", index=False)
    write_radar_svg(cards, OUT_DIR / "spender_personas.svg")

    # Phase 2: configurations are frozen; touch validation exactly once.
    valid_labels = pd.read_csv(ROOT / "data/valid_labels.csv")
    valid_raw_features = build_features(
        str(ROOT / "data/valid_transactions.jsonl"), str(CUTOFF.date())
    )
    valid_behavior = persona_features(valid_raw_features)
    valid_assignments = assign_personas(valid_behavior, scaler, kmeans, persona_names)
    valid_streams = build_stream_table(ROOT / "data/valid_transactions.jsonl")
    valid_predictions = all_config_predictions(valid_labels["client_id"], valid_streams)
    persona_valid_pred = predict_persona_rules(
        valid_labels["client_id"], valid_assignments, valid_predictions, choices
    )
    global_valid_pred = valid_predictions[GLOBAL_CONFIG]

    validation_output = valid_labels[["client_id", LABEL_COL]].copy()
    validation_output = validation_output.merge(
        valid_assignments[["client_id", "persona"]], on="client_id", how="left"
    )
    validation_output["global_median_prediction"] = global_valid_pred
    validation_output["persona_rule_prediction"] = persona_valid_pred
    validation_output.to_csv(OUT_DIR / "validation_predictions.csv", index=False)

    per_persona = []
    for persona in sorted(valid_assignments["persona"].unique()):
        mask = validation_output["persona"].to_numpy() == persona
        per_persona.append(
            {
                "persona": persona,
                "n_valid_clients": int(mask.sum()),
                "selected_config": config_name(choices[persona]),
                "global_median_macro_f1": macro_f1(
                    valid_labels.loc[mask, LABEL_COL], global_valid_pred[mask]
                ),
                "persona_rule_macro_f1": macro_f1(
                    valid_labels.loc[mask, LABEL_COL], persona_valid_pred[mask]
                ),
            }
        )
    pd.DataFrame(per_persona).to_csv(
        OUT_DIR / "validation_by_persona.csv", index=False
    )

    summary = {
        "method": "four train-fitted RFM-plus spender personas with guarded local rules",
        "n_personas": N_PERSONAS,
        "persona_features": PERSONA_FEATURES,
        "global_config": config_name(GLOBAL_CONFIG),
        "selection_guardrails": {
            "minimum_persona_size": MIN_PERSONA_SIZE,
            "shrinkage_strength": SHRINKAGE_STRENGTH,
            "minimum_shrunk_cv_delta": MIN_SHRUNK_CV_DELTA,
            "minimum_folds_won": MIN_FOLDS_WON,
        },
        "selected_persona_configs": {
            persona: config_name(config) for persona, config in choices.items()
        },
        "train_macro_f1": {
            "global_median": macro_f1(train_labels[LABEL_COL], global_train_pred),
            "persona_rules_apparent": macro_f1(
                train_labels[LABEL_COL], persona_train_pred
            ),
        },
        "validation_macro_f1": {
            "original_mean_baseline_reference": 0.4844204458488582,
            "global_median_rule": macro_f1(valid_labels[LABEL_COL], global_valid_pred),
            "persona_rules": macro_f1(valid_labels[LABEL_COL], persona_valid_pred),
        },
        "validation_delta_vs_original": macro_f1(
            valid_labels[LABEL_COL], persona_valid_pred
        )
        - 0.4844204458488582,
        "validation_delta_vs_global_median": macro_f1(
            valid_labels[LABEL_COL], persona_valid_pred
        )
        - macro_f1(valid_labels[LABEL_COL], global_valid_pred),
        "silhouette_k4": float(
            sensitivity.loc[sensitivity["n_personas"] == 4, "silhouette"].iloc[0]
        ),
    }
    (OUT_DIR / "results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("\nPersona raw profiles:")
    print(raw_profiles.to_string(index=False, float_format="%.3f"))
    print("\nValidation by persona:")
    print(pd.DataFrame(per_persona).to_string(index=False, float_format="%.6f"))


if __name__ == "__main__":
    main()
