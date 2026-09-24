"""Leakage-controlled supervised persona router for transparent due-date rules.

The model is deliberately a *policy tree*, not a black-box classifier.  A
shallow tree partitions clients using a small set of spend/recurrence
features.  Each terminal leaf chooses one of a finite menu of auditable rules:

    next due = last charge + cadence estimate, with a fixed overdue grace.

Selection protocol
------------------
1. Candidate tree depth/minimum-leaf settings are compared using stratified
   out-of-fold predictions on train only.
2. The winning setting is refit on all train clients.
3. The fixed validation labels are loaded exactly once, after the tree and its
   leaf actions are locked.

Run from the repository root:

    py -3.10 models/rule_based/experiments/personas/supervised_personas/experiment.py
"""

from __future__ import annotations

import calendar
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.rule_based.model import rule_predict
from src.category_map import (
    NON_SUBSCRIPTION_PHRASES,
    NON_SUBSCRIPTION_TYPES,
    TARGET_CATEGORIES,
)
from src.features import build_features
from src.model import ALL_LABELS, LABEL_COL
from src.recurrence import _cluster_by_amount, _stream_stats, load_transactions


CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
HERE = Path(__file__).resolve().parent
SEED = 20260924
N_FOLDS = 5

# Restricted, presentation-friendly feature vocabulary.  Each variable has an
# intuitive banking interpretation and all are observable before the cutoff.
FEATURES = [
    "monthly_outflow",
    "avg_out_amount",
    "card_share",
    "transactions_per_month",
    "active_categories",
    "live_streams",
    "subscription_share",
    "cadence_irregularity",
]

# The menu is intentionally compact.  Grace is varied only where train-only
# robust-cadence work showed a plausible boundary.
ACTION_SPECS = [
    ("mean_g5", "mean_gap", 5),
    ("median_g3", "median_gap", 3),
    ("median_g5", "median_gap", 5),
    ("median_g7", "median_gap", 7),
    ("recent_g5", "recent3_median_gap", 5),
    ("calendar_g5", "calendar_dom", 5),
]
ACTION_ORDER = [a[0] for a in ACTION_SPECS]


def macro_f1(y_true: np.ndarray | pd.Series, y_pred: np.ndarray) -> float:
    return float(
        f1_score(y_true, y_pred, labels=ALL_LABELS, average="macro", zero_division=0)
    )


def _calendar_due(last: pd.Timestamp, days: np.ndarray) -> pd.Timestamp:
    day = int(np.rint(np.median(days)))
    year, month = last.year, last.month + 1
    if month == 13:
        year, month = year + 1, 1
    return pd.Timestamp(
        year=year,
        month=month,
        day=min(day, calendar.monthrange(year, month)[1]),
        tz=last.tz,
    )


def build_stream_table(path: Path) -> pd.DataFrame:
    transactions = load_transactions(str(path))
    outgoing = transactions[transactions["direction"] == "out"]
    pool = outgoing[
        ~outgoing["type"].isin(NON_SUBSCRIPTION_TYPES)
        & ~outgoing["description"].str.contains("|".join(NON_SUBSCRIPTION_PHRASES))
    ]
    rows: list[dict] = []
    for client_id, group in pool.groupby("client_id"):
        for cluster in _cluster_by_amount(group):
            stats = _stream_stats(cluster)
            if not stats["is_recurring"] or stats["category"] is None:
                continue
            dates = cluster["timestamp"].sort_values()
            gaps = dates.diff().dt.days.dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "client_id": client_id,
                    "category": stats["category"],
                    "n_occurrences": int(stats["n_occurrences"]),
                    "last_date": stats["last_date"],
                    "mean_gap": float(np.mean(gaps)),
                    "median_gap": float(np.median(gaps)),
                    "recent3_median_gap": float(np.median(gaps[-3:])),
                    "calendar_due": _calendar_due(
                        stats["last_date"], dates.dt.day.to_numpy()
                    ),
                    "gap_cv": float(stats["gap_cv"]),
                }
            )
    return pd.DataFrame(rows)


def build_persona_features(transaction_path: Path, streams: pd.DataFrame) -> pd.DataFrame:
    """Eight stable, cutoff-safe features used only for persona routing."""
    base = build_features(str(transaction_path), str(CUTOFF.date())).set_index("client_id")
    stream_summary = streams.groupby("client_id").agg(
        subscription_occurrences=("n_occurrences", "sum"),
        cadence_irregularity=("gap_cv", "median"),
    )
    profile = pd.DataFrame(index=base.index)
    tenure_months = (base["tenure_days"] / 30).clip(lower=1)
    profile["monthly_outflow"] = base["total_out"] / tenure_months
    profile["avg_out_amount"] = base["avg_out_amount"]
    profile["card_share"] = base["frac_card_payment"]
    profile["transactions_per_month"] = base["n_txns_per_month"]
    profile["active_categories"] = base["n_active_categories"]
    profile["live_streams"] = base["n_live_streams"]
    profile["subscription_share"] = (
        stream_summary["subscription_occurrences"].reindex(base.index).fillna(0)
        / base["n_txns"].clip(lower=1)
    )
    profile["cadence_irregularity"] = stream_summary["cadence_irregularity"].reindex(
        base.index
    )
    return profile[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def rule_predictions(
    clients: pd.Index | pd.Series, streams: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Precompute each transparent action's prediction for every client."""
    outputs: dict[str, np.ndarray] = {}
    category_order = {c: i for i, c in enumerate(TARGET_CATEGORIES)}
    for action, estimator, grace in ACTION_SPECS:
        work = streams.copy()
        due = work["calendar_due"] if estimator == "calendar_dom" else (
            work["last_date"] + pd.to_timedelta(work[estimator], unit="D")
        )
        work["days_until_due"] = (due - CUTOFF).dt.total_seconds() / 86_400
        live = work[work["days_until_due"] >= -grace]
        max_n = live.groupby("client_id")["n_occurrences"].max()
        short = set(max_n[max_n.isin([3, 4])].index)
        family_due = (
            live.groupby(["client_id", "category"])["days_until_due"]
            .max()
            .reset_index()
        )
        family_due["category_order"] = family_due["category"].map(category_order)
        first = (
            family_due.sort_values(["client_id", "days_until_due", "category_order"])
            .groupby("client_id", sort=False)["category"]
            .first()
        )
        outputs[action] = np.asarray(
            ["none" if c in short else first.get(c, "none") for c in clients]
        )
    return outputs


@dataclass
class PolicyNode:
    indices: np.ndarray
    action: str
    depth: int
    feature: str | None = None
    threshold: float | None = None
    left: "PolicyNode | None" = None
    right: "PolicyNode | None" = None
    node_id: int = -1
    persona: str | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None


def _weighted_reward(
    indices: np.ndarray,
    action: str,
    y: np.ndarray,
    action_predictions: dict[str, np.ndarray],
    weights: np.ndarray,
) -> float:
    return float(
        weights[indices]
        @ (action_predictions[action][indices] == y[indices]).astype(float)
    )


def _best_action(
    indices: np.ndarray,
    y: np.ndarray,
    action_predictions: dict[str, np.ndarray],
    weights: np.ndarray,
) -> tuple[str, float]:
    scored = [
        (_weighted_reward(indices, action, y, action_predictions, weights), action)
        for action in ACTION_ORDER
    ]
    # Deterministic tie break prefers the globally strong median-gap action.
    priority = {a: i + 1 for i, a in enumerate(ACTION_ORDER)}
    priority["median_g5"] = 0
    score, action = max(scored, key=lambda x: (x[0], -priority[x[1]]))
    return action, score


def _candidate_thresholds(values: np.ndarray) -> np.ndarray:
    unique = np.unique(values)
    if len(unique) <= 12:
        return (unique[:-1] + unique[1:]) / 2
    return np.unique(np.quantile(values, np.linspace(0.15, 0.85, 15)))


def fit_policy_tree(
    X: pd.DataFrame,
    y: np.ndarray,
    action_predictions: dict[str, np.ndarray],
    max_depth: int,
    min_leaf: int,
) -> PolicyNode:
    """Greedy shallow policy tree maximizing class-balanced rule correctness."""
    counts = pd.Series(y).value_counts()
    weights = np.asarray([1.0 / counts[v] for v in y], dtype=float)

    def grow(indices: np.ndarray, depth: int) -> PolicyNode:
        action, parent_reward = _best_action(indices, y, action_predictions, weights)
        node = PolicyNode(indices=indices, action=action, depth=depth)
        if depth >= max_depth or len(indices) < 2 * min_leaf:
            return node
        best: tuple[float, str, float, np.ndarray, np.ndarray, str, str] | None = None
        for feature in FEATURES:
            values = X[feature].to_numpy()[indices]
            for threshold in _candidate_thresholds(values):
                left = indices[values <= threshold]
                right = indices[values > threshold]
                if len(left) < min_leaf or len(right) < min_leaf:
                    continue
                left_action, left_reward = _best_action(
                    left, y, action_predictions, weights
                )
                right_action, right_reward = _best_action(
                    right, y, action_predictions, weights
                )
                gain = left_reward + right_reward - parent_reward
                proposal = (
                    gain,
                    feature,
                    float(threshold),
                    left,
                    right,
                    left_action,
                    right_action,
                )
                if best is None or proposal[:3] > best[:3]:
                    best = proposal
        # A split must improve balanced reward, not merely partition clients.
        if best is None or best[0] <= 1e-12:
            return node
        _, node.feature, node.threshold, left, right, _, _ = best
        node.left = grow(left, depth + 1)
        node.right = grow(right, depth + 1)
        return node

    return grow(np.arange(len(X)), 0)


def _route_one(node: PolicyNode, row: pd.Series) -> PolicyNode:
    while not node.is_leaf:
        node = node.left if row[node.feature] <= node.threshold else node.right  # type: ignore[index,assignment,operator]
    return node


def assign_ids_and_names(root: PolicyNode) -> list[PolicyNode]:
    leaves: list[PolicyNode] = []
    counter = 0

    def visit(node: PolicyNode) -> None:
        nonlocal counter
        node.node_id = counter
        counter += 1
        if node.is_leaf:
            leaves.append(node)
        else:
            visit(node.left)  # type: ignore[arg-type]
            visit(node.right)  # type: ignore[arg-type]

    visit(root)
    # Give the simplest two-leaf result slide-ready names.  Other learned
    # structures retain neutral profile names rather than post-hoc storytelling.
    if root.feature == "avg_out_amount" and len(leaves) == 2:
        leaves[0].persona = "Everyday spenders"
        leaves[1].persona = "High-ticket spenders"
    else:
        for i, leaf in enumerate(leaves, 1):
            leaf.persona = f"Profile {i}"
    return leaves


def routed_predictions(
    root: PolicyNode,
    X: pd.DataFrame,
    action_predictions: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[PolicyNode]]:
    leaves = [_route_one(root, row) for _, row in X.iterrows()]
    pred = np.asarray([action_predictions[leaf.action][i] for i, leaf in enumerate(leaves)])
    return pred, leaves


def _subset_actions(predictions: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, np.ndarray]:
    return {action: pred[rows] for action, pred in predictions.items()}


def cross_validate_configs(
    X: pd.DataFrame,
    y: np.ndarray,
    actions: dict[str, np.ndarray],
) -> pd.DataFrame:
    configs = [(1, 120), (1, 200), (2, 120), (2, 200), (3, 120), (3, 200)]
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows: list[dict] = []
    for depth, min_leaf in configs:
        oof = np.full(len(y), "", dtype=object)
        leaf_counts = []
        for fit_idx, test_idx in folds.split(X, y):
            fit_tree = fit_policy_tree(
                X.iloc[fit_idx].reset_index(drop=True),
                y[fit_idx],
                _subset_actions(actions, fit_idx),
                depth,
                min_leaf,
            )
            assign_ids_and_names(fit_tree)
            fold_pred, fold_leaves = routed_predictions(
                fit_tree,
                X.iloc[test_idx].reset_index(drop=True),
                _subset_actions(actions, test_idx),
            )
            oof[test_idx] = fold_pred
            leaf_counts.append(len({leaf.node_id for leaf in fold_leaves}))
        rows.append(
            {
                "max_depth": depth,
                "min_leaf": min_leaf,
                "oof_macro_f1": macro_f1(y, oof),
                "mean_test_leaf_count": float(np.mean(leaf_counts)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["oof_macro_f1", "max_depth", "min_leaf"],
        ascending=[False, True, False],
    )


def _paths(root: PolicyNode) -> dict[int, str]:
    result: dict[int, str] = {}

    def walk(node: PolicyNode, clauses: list[str]) -> None:
        if node.is_leaf:
            result[node.node_id] = " AND ".join(clauses) if clauses else "all clients"
            return
        label = DISPLAY_NAMES[node.feature]  # type: ignore[index]
        walk(node.left, clauses + [f"{label} <= {node.threshold:.2f}"])  # type: ignore[arg-type,union-attr]
        walk(node.right, clauses + [f"{label} > {node.threshold:.2f}"])  # type: ignore[arg-type,union-attr]

    walk(root, [])
    return result


DISPLAY_NAMES = {
    "monthly_outflow": "Monthly outflow",
    "avg_out_amount": "Average outgoing payment",
    "card_share": "Card-payment share",
    "transactions_per_month": "Transactions / month",
    "active_categories": "Active categories",
    "live_streams": "Live recurring streams",
    "subscription_share": "Recurring-payment share",
    "cadence_irregularity": "Cadence irregularity",
}

ACTION_LABELS = {
    "mean_g5": "Mean cadence, 5-day grace",
    "median_g3": "Median cadence, 3-day grace",
    "median_g5": "Median cadence, 5-day grace",
    "median_g7": "Median cadence, 7-day grace",
    "recent_g5": "Recent-3 median cadence, 5-day grace",
    "calendar_g5": "Calendar day-of-month, 5-day grace",
}


def export_tree_svg(root: PolicyNode, output: Path) -> None:
    """Dependency-free diagram suitable for slides and the README."""
    levels: dict[int, list[PolicyNode]] = {}

    def collect(node: PolicyNode) -> None:
        levels.setdefault(node.depth, []).append(node)
        if not node.is_leaf:
            collect(node.left)  # type: ignore[arg-type]
            collect(node.right)  # type: ignore[arg-type]

    collect(root)
    width, x_gap, y_gap = 1200, 270, 180
    height = (max(levels) + 1) * y_gap + 100
    positions: dict[int, tuple[float, float]] = {}
    leaves = [n for nodes in levels.values() for n in nodes if n.is_leaf]
    for i, leaf in enumerate(leaves):
        positions[leaf.node_id] = ((i + 1) * width / (len(leaves) + 1), height - 100)

    def place(node: PolicyNode) -> tuple[float, float]:
        if node.node_id in positions:
            return positions[node.node_id]
        lx, _ = place(node.left)  # type: ignore[arg-type]
        rx, _ = place(node.right)  # type: ignore[arg-type]
        positions[node.node_id] = ((lx + rx) / 2, 70 + node.depth * y_gap)
        return positions[node.node_id]

    place(root)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f8fc"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:16px;font-weight:700}.small{font-size:13px}.edge{font-size:12px;fill:#5f6b7a}</style>',
    ]
    for nodes in levels.values():
        for node in nodes:
            if node.is_leaf:
                continue
            x, y = positions[node.node_id]
            for child, mark in ((node.left, "YES"), (node.right, "NO")):
                cx, cy = positions[child.node_id]  # type: ignore[union-attr]
                parts.append(f'<line x1="{x}" y1="{y+34}" x2="{cx}" y2="{cy-38}" stroke="#8090a4" stroke-width="2"/>')
                parts.append(f'<text class="edge" x="{(x+cx)/2}" y="{(y+cy)/2-4}" text-anchor="middle">{mark}</text>')
    for nodes in levels.values():
        for node in nodes:
            x, y = positions[node.node_id]
            fill = "#e8f1ff" if not node.is_leaf else "#e5f7ee"
            stroke = "#3976c5" if not node.is_leaf else "#20845b"
            parts.append(f'<rect x="{x-125}" y="{y-38}" width="250" height="76" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
            if node.is_leaf:
                line1 = node.persona
                line2 = ACTION_LABELS[node.action]
                line3 = f"n={len(node.indices)}"
            else:
                line1 = DISPLAY_NAMES[node.feature]  # type: ignore[index]
                line2 = f"at most {node.threshold:.2f}?"
                line3 = f"n={len(node.indices)}"
            for dy, text, cls in ((-13, line1, "title"), (8, line2, "small"), (27, line3, "small")):
                parts.append(f'<text class="{cls}" x="{x}" y="{y+dy}" text-anchor="middle">{html.escape(text)}</text>')
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def serialize_tree(node: PolicyNode) -> dict:
    out = {
        "node_id": node.node_id,
        "n_train_clients": len(node.indices),
        "default_action": node.action,
    }
    if node.is_leaf:
        out.update({"persona": node.persona, "rule": ACTION_LABELS[node.action]})
    else:
        out.update(
            {
                "feature": node.feature,
                "feature_label": DISPLAY_NAMES[node.feature],  # type: ignore[index]
                "threshold": node.threshold,
                "left": serialize_tree(node.left),  # type: ignore[arg-type]
                "right": serialize_tree(node.right),  # type: ignore[arg-type]
            }
        )
    return out


def main() -> None:
    # Everything through model locking uses train only.
    train_labels = pd.read_csv(ROOT / "data/train_labels.csv")
    train_streams = build_stream_table(ROOT / "data/train_transactions.jsonl")
    train_X = build_persona_features(
        ROOT / "data/train_transactions.jsonl", train_streams
    ).reindex(train_labels["client_id"]).reset_index(drop=True)
    train_y = train_labels[LABEL_COL].to_numpy()
    train_actions = rule_predictions(train_labels["client_id"], train_streams)

    cv = cross_validate_configs(train_X, train_y, train_actions)
    cv.to_csv(HERE / "train_oof_selection.csv", index=False)
    choice = cv.iloc[0]
    tree = fit_policy_tree(
        train_X,
        train_y,
        train_actions,
        int(choice["max_depth"]),
        int(choice["min_leaf"]),
    )
    train_leaves = assign_ids_and_names(tree)
    train_pred, train_routes = routed_predictions(tree, train_X, train_actions)
    locked = serialize_tree(tree)
    (HERE / "locked_tree.json").write_text(json.dumps(locked, indent=2), encoding="utf-8")
    export_tree_svg(tree, HERE / "persona_tree.svg")

    # Only now load validation data/labels for one final assessment.
    valid_labels = pd.read_csv(ROOT / "data/valid_labels.csv")
    valid_streams = build_stream_table(ROOT / "data/valid_transactions.jsonl")
    valid_X = build_persona_features(
        ROOT / "data/valid_transactions.jsonl", valid_streams
    ).reindex(valid_labels["client_id"]).reset_index(drop=True)
    valid_actions = rule_predictions(valid_labels["client_id"], valid_streams)
    valid_pred, valid_routes = routed_predictions(tree, valid_X, valid_actions)

    # Official original baseline, and the global median-gap rule comparator.
    official_features = build_features(
        str(ROOT / "data/valid_transactions.jsonl"), str(CUTOFF.date())
    )
    aligned = valid_labels[["client_id"]].merge(
        official_features, on="client_id", how="left"
    )
    original_pred = rule_predict(aligned.drop(columns="client_id"))
    median_pred = valid_actions["median_g5"]

    valid_true = valid_labels[LABEL_COL].to_numpy()
    scores = {
        "original_mean_rule": macro_f1(valid_true, original_pred),
        "global_median_rule": macro_f1(valid_true, median_pred),
        "supervised_persona_router": macro_f1(valid_true, valid_pred),
    }
    per_class = pd.DataFrame(
        {
            "label": ALL_LABELS,
            "original_mean_rule": f1_score(valid_true, original_pred, labels=ALL_LABELS, average=None, zero_division=0),
            "global_median_rule": f1_score(valid_true, median_pred, labels=ALL_LABELS, average=None, zero_division=0),
            "supervised_persona_router": f1_score(valid_true, valid_pred, labels=ALL_LABELS, average=None, zero_division=0),
        }
    )
    per_class.to_csv(HERE / "per_class_results.csv", index=False)

    paths = _paths(tree)
    profile_rows = []
    for leaf in train_leaves:
        valid_idx = np.asarray([i for i, routed in enumerate(valid_routes) if routed.node_id == leaf.node_id])
        train_idx = leaf.indices
        profile_rows.append(
            {
                "persona": leaf.persona,
                "definition": paths[leaf.node_id],
                "selected_rule": ACTION_LABELS[leaf.action],
                "action_id": leaf.action,
                "n_train": len(train_idx),
                "n_valid": len(valid_idx),
                "median_monthly_outflow_train": float(train_X.iloc[train_idx]["monthly_outflow"].median()),
                "median_transactions_per_month_train": float(train_X.iloc[train_idx]["transactions_per_month"].median()),
                "median_active_categories_train": float(train_X.iloc[train_idx]["active_categories"].median()),
                "median_live_streams_train": float(train_X.iloc[train_idx]["live_streams"].median()),
                "valid_macro_f1_within_persona": macro_f1(valid_true[valid_idx], valid_pred[valid_idx]) if len(valid_idx) else np.nan,
            }
        )
    profiles = pd.DataFrame(profile_rows)
    profiles.to_csv(HERE / "persona_profiles.csv", index=False)

    pd.DataFrame(
        {
            "client_id": valid_labels["client_id"],
            "true_label": valid_true,
            "persona": [leaf.persona for leaf in valid_routes],
            "selected_action": [leaf.action for leaf in valid_routes],
            "persona_prediction": valid_pred,
            "global_median_prediction": median_pred,
            "original_prediction": original_pred,
        }
    ).to_csv(HERE / "validation_predictions.csv", index=False)

    summary = {
        "selection_protocol": f"{N_FOLDS}-fold stratified train-only OOF",
        "selected_max_depth": int(choice["max_depth"]),
        "selected_min_leaf": int(choice["min_leaf"]),
        "selected_oof_macro_f1": float(choice["oof_macro_f1"]),
        "global_median_train_macro_f1": macro_f1(
            train_y, train_actions["median_g5"]
        ),
        "final_train_macro_f1_resubstitution": macro_f1(train_y, train_pred),
        "n_personas": len(train_leaves),
        "validation_scores": scores,
        "delta_vs_original": scores["supervised_persona_router"] - scores["original_mean_rule"],
        "delta_vs_global_median": scores["supervised_persona_router"] - scores["global_median_rule"],
    }
    (HERE / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Train-only OOF configuration ranking:")
    print(cv.to_string(index=False, float_format="%.6f"))
    print("\nLocked tree:")
    print(json.dumps(locked, indent=2))
    print("\nOne-time validation results:")
    print(json.dumps(summary, indent=2))
    print("\nPersonas:")
    print(profiles.to_string(index=False))


if __name__ == "__main__":
    main()
