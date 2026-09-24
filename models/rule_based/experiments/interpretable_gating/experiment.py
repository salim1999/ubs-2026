"""Train-only search over small, interpretable due-date rule variants.

Run from the repository root with::

    py -3.10 models/rule_based/experiments/interpretable_gating/experiment.py

The fixed validation labels are not read until every rule and hyperparameter
has been selected from the training split. Stream caches and outputs are kept
next to this file so the experiment does not modify shared project artifacts.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.category_map import TARGET_CATEGORIES  # noqa: E402
from src.recurrence import detect_streams, load_transactions  # noqa: E402

CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
LABEL_COL = "target_next_recurring_merchant"
ALL_LABELS = TARGET_CATEGORIES + ["none"]
CACHE = HERE / "cache"


@dataclass(frozen=True)
class RuleConfig:
    """A deliberately small set of human-readable rule controls."""

    max_overdue_days: float = 5.0
    family_aggregation: str = "min"  # "min" (project baseline) or "closest_live"
    trial_gate: str = "baseline"  # baseline, none, chosen_3_4, or all_le4
    gap_cv_penalty_days: float = 0.0
    weak_keyword_penalty_days: float = 0.0
    per_label_overdue: tuple[tuple[str, float], ...] = ()

    def threshold_for(self, family: str) -> float:
        return dict(self.per_label_overdue).get(family, self.max_overdue_days)


def macro_f1(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            labels=ALL_LABELS,
            zero_division=0,
        )
    )


def load_labels(split: str) -> pd.Series:
    path = REPO / "data" / f"{split}_labels.csv"
    return pd.read_csv(path).set_index("client_id")[LABEL_COL]


def load_stream_rows(split: str) -> pd.DataFrame:
    """Load recurring categorized streams, caching only inside this experiment."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{split}_recurring_streams.csv"
    if path.exists():
        rows = pd.read_csv(path, parse_dates=["first_date", "last_date"])
    else:
        tx_path = REPO / "data" / f"{split}_transactions.jsonl"
        rows = detect_streams(load_transactions(str(tx_path)))
        rows = rows[rows["is_recurring"] & rows["category"].notna()].copy()
        keep = [
            "client_id",
            "category",
            "n_occurrences",
            "first_date",
            "last_date",
            "mean_gap_days",
            "gap_cv",
            "amount_cv",
            "category_agreement",
            "n_keyword_hits",
        ]
        rows[keep].to_csv(path, index=False)
        rows = rows[keep]

    # pandas only preserves timezone consistently after a cache round-trip if
    # normalized explicitly.
    rows["first_date"] = pd.to_datetime(rows["first_date"], utc=True)
    rows["last_date"] = pd.to_datetime(rows["last_date"], utc=True)
    rows["recency_days"] = (CUTOFF - rows["last_date"]).dt.days.astype(float)
    rows["over"] = rows["recency_days"] - rows["mean_gap_days"]
    rows["keyword_coverage"] = (
        rows["n_keyword_hits"] / rows["n_occurrences"].clip(lower=1)
    ).clip(0, 1)
    rows["gap_cv"] = rows["gap_cv"].fillna(0.0)
    return rows


def stream_rule_predict(
    client_ids: pd.Index, streams: pd.DataFrame, config: RuleConfig
) -> np.ndarray:
    """Apply a transparent liveness -> family rank -> trial-gate rule."""
    work = streams.copy()
    thresholds = work["category"].map(
        {family: config.threshold_for(family) for family in TARGET_CATEGORIES}
    )
    work = work[work["over"] <= thresholds].copy()
    work["rank_score"] = (
        work["over"]
        - config.gap_cv_penalty_days * work["gap_cv"]
        - config.weak_keyword_penalty_days * (1.0 - work["keyword_coverage"])
    )

    if work.empty:
        return np.full(len(client_ids), "none")

    family_keys = ["client_id", "category"]
    if config.family_aggregation == "min":
        # Exact semantics of the current project features: existence is based
        # on any live stream, but the family value is the minimum over-days.
        family_idx = work.groupby(family_keys)["over"].idxmin()
        family_rows = work.loc[family_idx].copy()
        family_rows["rank_score"] = family_rows["over"]
    elif config.family_aggregation == "closest_live":
        # Corrected semantics: closest/confident live stream represents family.
        family_idx = work.groupby(family_keys)["rank_score"].idxmax()
        family_rows = work.loc[family_idx]
    else:
        raise ValueError(f"Unknown aggregation: {config.family_aggregation}")

    chosen_idx = family_rows.groupby("client_id")["rank_score"].idxmax()
    chosen = family_rows.loc[chosen_idx].set_index("client_id")
    prediction = chosen["category"].astype(str).reindex(client_ids).fillna("none")

    if config.trial_gate == "none":
        reject = pd.Series(False, index=chosen.index)
    elif config.trial_gate in ("baseline", "all_le4"):
        max_occ = work.groupby("client_id")["n_occurrences"].max().reindex(chosen.index)
        reject = max_occ.isin([3, 4]) if config.trial_gate == "baseline" else max_occ.le(4)
    elif config.trial_gate == "chosen_3_4":
        reject = chosen["n_occurrences"].isin([3, 4])
    else:
        raise ValueError(f"Unknown trial gate: {config.trial_gate}")
    prediction.loc[reject[reject].index] = "none"
    return prediction.to_numpy()


def score_config(
    labels: pd.Series, streams: pd.DataFrame, config: RuleConfig
) -> float:
    return macro_f1(labels, stream_rule_predict(labels.index, streams, config))


def paired_bootstrap_delta(
    y: pd.Series,
    candidate: np.ndarray,
    baseline: np.ndarray,
    repetitions: int = 2000,
) -> tuple[float, float, float]:
    """Client bootstrap interval and P(delta > 0), for uncertainty only."""
    rng = np.random.default_rng(20260924)
    y_array = y.to_numpy()
    deltas = np.empty(repetitions)
    for i in range(repetitions):
        sample = rng.integers(0, len(y_array), size=len(y_array))
        deltas[i] = macro_f1(pd.Series(y_array[sample]), candidate[sample]) - macro_f1(
            pd.Series(y_array[sample]), baseline[sample]
        )
    low, high = np.quantile(deltas, [0.025, 0.975])
    return float(low), float(high), float(np.mean(deltas > 0))


def select_global_rule(train_y: pd.Series, train_streams: pd.DataFrame) -> tuple[RuleConfig, pd.DataFrame]:
    """Predeclared compact search; only training labels are used."""
    rows = []
    for aggregation in ("min", "closest_live"):
        for overdue in (-5, 0, 3, 5, 7, 10, 15, 20):
            for trial in ("none", "baseline", "chosen_3_4", "all_le4"):
                config = RuleConfig(
                    max_overdue_days=overdue,
                    family_aggregation=aggregation,
                    trial_gate=trial,
                )
                rows.append({**asdict(config), "train_macro_f1": score_config(train_y, train_streams, config)})
    table = pd.DataFrame(rows).sort_values(
        ["train_macro_f1", "family_aggregation", "max_overdue_days"],
        ascending=[False, True, True],
    )
    best = table.iloc[0]
    config = RuleConfig(
        max_overdue_days=float(best["max_overdue_days"]),
        family_aggregation=str(best["family_aggregation"]),
        trial_gate=str(best["trial_gate"]),
    )
    return config, table


def select_confidence_rule(
    train_y: pd.Series, train_streams: pd.DataFrame, base: RuleConfig
) -> tuple[RuleConfig, pd.DataFrame]:
    """Tune small day-equivalent penalties for irregular/weak streams."""
    rows = []
    for gap_penalty in (0, 2, 5, 10, 15):
        for keyword_penalty in (0, 2, 5, 10, 15):
            config = replace(
                base,
                family_aggregation="closest_live",
                gap_cv_penalty_days=float(gap_penalty),
                weak_keyword_penalty_days=float(keyword_penalty),
            )
            rows.append({**asdict(config), "train_macro_f1": score_config(train_y, train_streams, config)})
    table = pd.DataFrame(rows).sort_values(
        ["train_macro_f1", "gap_cv_penalty_days", "weak_keyword_penalty_days"],
        ascending=[False, True, True],
    )
    best = table.iloc[0]
    config = replace(
        base,
        family_aggregation="closest_live",
        gap_cv_penalty_days=float(best["gap_cv_penalty_days"]),
        weak_keyword_penalty_days=float(best["weak_keyword_penalty_days"]),
    )
    return config, table


def select_per_label_thresholds(
    train_y: pd.Series, train_streams: pd.DataFrame, base: RuleConfig
) -> tuple[RuleConfig, pd.DataFrame]:
    """One-pass coordinate search for seven readable liveness thresholds."""
    choices = (-5, 0, 3, 5, 7, 10, 15, 20)
    thresholds = {family: base.max_overdue_days for family in TARGET_CATEGORIES}
    rows = []
    # Fixed category order is declared above. One pass intentionally limits
    # flexibility/overfitting and leaves the final rule easy to audit.
    for family in TARGET_CATEGORIES:
        candidates = []
        for value in choices:
            proposal = thresholds | {family: float(value)}
            config = replace(base, per_label_overdue=tuple(proposal.items()))
            score = score_config(train_y, train_streams, config)
            candidates.append((score, value))
            rows.append({"family": family, "threshold": value, "train_macro_f1": score})
        # Prefer the threshold closest to the global setting on an exact tie.
        best_score, best_value = max(
            candidates,
            key=lambda item: (item[0], -abs(item[1] - base.max_overdue_days)),
        )
        thresholds[family] = float(best_value)
    return replace(base, per_label_overdue=tuple(thresholds.items())), pd.DataFrame(rows)


def main() -> None:
    train_y = load_labels("train")
    train_streams = load_stream_rows("train")

    baseline = RuleConfig()
    corrected = replace(baseline, family_aggregation="closest_live")
    global_best, global_search = select_global_rule(train_y, train_streams)
    confidence_best, confidence_search = select_confidence_rule(
        train_y, train_streams, global_best
    )
    per_label_min_best, threshold_min_search = select_per_label_thresholds(
        train_y, train_streams, global_best
    )
    per_label_corrected_best, threshold_corrected_search = select_per_label_thresholds(
        train_y, train_streams, confidence_best
    )

    configs = {
        "stream_reproduction": baseline,
        "corrected_family_aggregation": corrected,
        "train_selected_global_gate": global_best,
        "train_selected_confidence": confidence_best,
        "train_selected_per_label_min": per_label_min_best,
        "train_selected_per_label_corrected": per_label_corrected_best,
    }
    train_scores = {name: score_config(train_y, train_streams, config) for name, config in configs.items()}

    # Validation is deliberately loaded only after all selection is complete.
    valid_y = load_labels("valid")
    valid_streams = load_stream_rows("valid")
    valid_scores = {
        name: score_config(valid_y, valid_streams, config)
        for name, config in configs.items()
    }

    valid_predictions = {
        name: stream_rule_predict(valid_y.index, valid_streams, config)
        for name, config in configs.items()
    }
    baseline_pred = valid_predictions["stream_reproduction"]
    result_rows = [
        {
            "method": "project_baseline",
            "train_macro_f1": train_scores["stream_reproduction"],
            "valid_macro_f1": valid_scores["stream_reproduction"],
            "config": "Exact stream-level reproduction of models.rule_based.model.rule_predict",
            "n_changed_vs_baseline": 0,
            "bootstrap_delta_95_low": 0.0,
            "bootstrap_delta_95_high": 0.0,
            "bootstrap_p_delta_gt_0": 0.0,
        }
    ]
    for name, config in configs.items():
        pred = valid_predictions[name]
        low, high, probability = paired_bootstrap_delta(valid_y, pred, baseline_pred)
        result_rows.append(
            {
                "method": name,
                "train_macro_f1": train_scores[name],
                "valid_macro_f1": valid_scores[name],
                "config": json.dumps(asdict(config), sort_keys=True),
                "n_changed_vs_baseline": int(np.sum(pred != baseline_pred)),
                "bootstrap_delta_95_low": low,
                "bootstrap_delta_95_high": high,
                "bootstrap_p_delta_gt_0": probability,
            }
        )
    results = pd.DataFrame(result_rows)
    results.to_csv(HERE / "results.csv", index=False)
    global_search.to_csv(HERE / "global_search_train_only.csv", index=False)
    confidence_search.to_csv(HERE / "confidence_search_train_only.csv", index=False)
    threshold_min_search.to_csv(HERE / "per_label_min_search_train_only.csv", index=False)
    threshold_corrected_search.to_csv(
        HERE / "per_label_corrected_search_train_only.csv", index=False
    )
    details = pd.DataFrame({"client_id": valid_y.index, "actual": valid_y.to_numpy()})
    for name, pred in valid_predictions.items():
        details[name] = pred
    details.to_csv(HERE / "validation_predictions.csv", index=False)

    per_class_rows = []
    for name, pred in valid_predictions.items():
        scores = f1_score(
            valid_y,
            pred,
            average=None,
            labels=ALL_LABELS,
            zero_division=0,
        )
        per_class_rows.extend(
            {"method": name, "label": label, "valid_f1": score}
            for label, score in zip(ALL_LABELS, scores)
        )
    pd.DataFrame(per_class_rows).to_csv(HERE / "per_class_results.csv", index=False)
    with open(HERE / "selected_configs.json", "w", encoding="utf-8") as handle:
        json.dump({name: asdict(config) for name, config in configs.items()}, handle, indent=2)

    print("Selected configurations (training labels only):")
    for name in (
        "train_selected_global_gate",
        "train_selected_confidence",
        "train_selected_per_label_min",
        "train_selected_per_label_corrected",
    ):
        print(f"  {name}: {configs[name]}")
    print("\nScores:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
