"""Compare simple, interpretable prediction architectures.

The validation split is used once, for the final comparison only. Any model
choice (threshold, cluster count, shrinkage, blend weight) is selected with
five-fold out-of-fold predictions on the training split.

Run:
    py -3.10 -m src.compare_simple_models
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.category_map import TARGET_CATEGORIES
from src.evaluate import labelled_features
from src.model import ALL_LABELS, LABEL_COL, build_logreg, rule_predict


SEED = 17
N_FOLDS = 5


def macro_f1(y_true, y_pred) -> float:
    return f1_score(
        y_true, y_pred, labels=ALL_LABELS, average="macro", zero_division=0
    )


def logistic() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000, class_weight="balanced", random_state=SEED
                ),
            ),
        ]
    )


def aligned_proba(model, X: pd.DataFrame, labels=ALL_LABELS) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = list(model.classes_)
    out = np.zeros((len(X), len(labels)))
    for j, label in enumerate(labels):
        if label in classes:
            out[:, j] = raw[:, classes.index(label)]
    return out


def predictions_from_proba(proba: np.ndarray) -> np.ndarray:
    return np.asarray(ALL_LABELS)[np.argmax(proba, axis=1)]


def tune_scalar(y, predict, values):
    scored = [(float(v), macro_f1(y, predict(float(v)))) for v in values]
    return max(scored, key=lambda item: (item[1], -item[0])), scored


def fit_none_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    return logistic().fit(X, (y == "none").astype(int))


def none_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(1)]


def hurdle_fold(X_train, y_train, X_test):
    none_model = fit_none_model(X_train, y_train)
    mask = y_train != "none"
    family_model = logistic().fit(X_train.loc[mask], y_train.loc[mask])
    family_proba = aligned_proba(family_model, X_test, TARGET_CATEGORIES)
    return none_probability(none_model, X_test), family_proba


def hurdle_predict(p_none, family_proba, threshold):
    family = np.asarray(TARGET_CATEGORIES)[np.argmax(family_proba, axis=1)]
    return np.where(p_none >= threshold, "none", family)


SHARED_LOCAL_FEATURES = [
    "active",
    "n_streams",
    "n_occurrences",
    "recency_days",
    "tenure_days",
    "mean_amount",
    "gap_cv",
    "mean_gap_days",
    "days_until_next_due",
    "recent_txns",
    "over_days",
    "is_live",
]

SHARED_GENERAL_FEATURES = [
    "tenure_days",
    "recency_days",
    "n_txns_per_month",
    "avg_out_amount",
    "frac_card_payment",
    "n_distinct_mcc",
    "n_active_categories",
    "n_live_streams",
    "max_live_n_occurrences",
    "min_over_days",
    "short_live_stream",
    "adoption_rate_per_year",
]


def long_family_table(X: pd.DataFrame, y: pd.Series | None = None):
    """One row per client-family, with coefficients shared across families."""
    blocks = []
    targets = []
    for cat in TARGET_CATEGORIES:
        block = X[SHARED_GENERAL_FEATURES].reset_index(drop=True).copy()
        for feature in SHARED_LOCAL_FEATURES:
            block[f"family_{feature}"] = X[f"{feature}_{cat}"].to_numpy()
        for category_indicator in TARGET_CATEGORIES:
            block[f"category_{category_indicator}"] = int(cat == category_indicator)
        blocks.append(block)
        if y is not None:
            targets.append((y.reset_index(drop=True) == cat).astype(int))
    long_X = pd.concat(blocks, ignore_index=True)
    long_y = pd.concat(targets, ignore_index=True) if y is not None else None
    return long_X, long_y


def shared_family_scores(model, X: pd.DataFrame) -> np.ndarray:
    long_X, _ = long_family_table(X)
    classes = list(model.classes_)
    raw = model.predict_proba(long_X)[:, classes.index(1)]
    return raw.reshape(len(TARGET_CATEGORIES), len(X)).T


ROUTING_FEATURES = [
    "n_txns_per_month",
    "avg_out_amount",
    "frac_card_payment",
    "frac_p2p",
    "n_distinct_mcc",
    "n_active_categories",
    "n_live_streams",
    "max_live_n_occurrences",
    "min_over_days",
    "short_live_stream",
    "adoption_rate_per_year",
]


@dataclass
class ClusterExperts:
    router_prep: Pipeline
    router: KMeans
    global_model: Pipeline
    experts: dict[int, Pipeline]
    alpha: float

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        global_p = aligned_proba(self.global_model, X)
        groups = self.router.predict(self.router_prep.transform(X[ROUTING_FEATURES]))
        local_p = np.zeros_like(global_p)
        for group in np.unique(groups):
            rows = np.flatnonzero(groups == group)
            local_p[rows] = aligned_proba(self.experts[int(group)], X.iloc[rows])
        return (1 - self.alpha) * global_p + self.alpha * local_p


def fit_cluster_experts(X, y, n_clusters: int, alpha: float) -> ClusterExperts:
    prep = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    routing_X = prep.fit_transform(X[ROUTING_FEATURES])
    router = KMeans(n_clusters=n_clusters, n_init=20, random_state=SEED).fit(routing_X)
    groups = router.labels_
    global_model = logistic().fit(X, y)
    experts = {}
    for group in range(n_clusters):
        rows = groups == group
        # Each training cluster is large enough to contain every class in the
        # present data. Falling back remains safer if the data changes.
        experts[group] = (
            logistic().fit(X.loc[rows], y.loc[rows])
            if y.loc[rows].nunique() == len(ALL_LABELS)
            else global_model
        )
    return ClusterExperts(prep, router, global_model, experts, alpha)


def one_hot_predictions(predictions) -> np.ndarray:
    index = {label: j for j, label in enumerate(ALL_LABELS)}
    out = np.zeros((len(predictions), len(ALL_LABELS)))
    for i, label in enumerate(predictions):
        out[i, index[label]] = 1.0
    return out


def main() -> None:
    print("Rebuilding current features from raw transactions...")
    train = labelled_features("train")
    valid = labelled_features("valid")
    y_train = train[LABEL_COL].reset_index(drop=True)
    y_valid = valid[LABEL_COL].reset_index(drop=True)
    X_train = train.drop(columns=["client_id", LABEL_COL]).reset_index(drop=True)
    X_valid = (
        valid.drop(columns=["client_id", LABEL_COL])
        .reindex(columns=X_train.columns)
        .reset_index(drop=True)
    )
    folds = list(
        StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(
            X_train, y_train
        )
    )

    # Baselines and their OOF probabilities.
    baseline_oof = np.zeros((len(X_train), len(ALL_LABELS)))
    for fit_rows, test_rows in folds:
        model = build_logreg().fit(X_train.iloc[fit_rows], y_train.iloc[fit_rows])
        baseline_oof[test_rows] = aligned_proba(model, X_train.iloc[test_rows])
    baseline = build_logreg().fit(X_train, y_train)
    baseline_valid_p = aligned_proba(baseline, X_valid)

    results = {
        "global_logreg_baseline": predictions_from_proba(baseline_valid_p),
        "due_date_rule_baseline": rule_predict(X_valid),
    }
    choices = {}

    # 1. Hurdle: explicitly separate "none" from selection of a family.
    hurdle_none_oof = np.zeros(len(X_train))
    hurdle_family_oof = np.zeros((len(X_train), len(TARGET_CATEGORIES)))
    for fit_rows, test_rows in folds:
        p_none, p_family = hurdle_fold(
            X_train.iloc[fit_rows], y_train.iloc[fit_rows], X_train.iloc[test_rows]
        )
        hurdle_none_oof[test_rows] = p_none
        hurdle_family_oof[test_rows] = p_family
    (hurdle_threshold, hurdle_cv), _ = tune_scalar(
        y_train,
        lambda t: hurdle_predict(hurdle_none_oof, hurdle_family_oof, t),
        np.linspace(0.20, 0.80, 25),
    )
    hurdle_none_model = fit_none_model(X_train, y_train)
    non_none = y_train != "none"
    hurdle_family_model = logistic().fit(X_train.loc[non_none], y_train.loc[non_none])
    results["two_stage_hurdle"] = hurdle_predict(
        none_probability(hurdle_none_model, X_valid),
        aligned_proba(hurdle_family_model, X_valid, TARGET_CATEGORIES),
        hurdle_threshold,
    )
    choices["two_stage_hurdle"] = (
        f"none threshold={hurdle_threshold:.3f}; train OOF={hurdle_cv:.4f}"
    )

    # 2. Family-symmetric ranker: one shared binary regression over 7 rows/client.
    shared_oof = np.zeros((len(X_train), len(TARGET_CATEGORIES)))
    shared_none_oof = np.zeros(len(X_train))
    for fit_rows, test_rows in folds:
        long_X, long_y = long_family_table(
            X_train.iloc[fit_rows], y_train.iloc[fit_rows]
        )
        ranker = logistic().fit(long_X, long_y)
        shared_oof[test_rows] = shared_family_scores(ranker, X_train.iloc[test_rows])
        gate = fit_none_model(X_train.iloc[fit_rows], y_train.iloc[fit_rows])
        shared_none_oof[test_rows] = none_probability(gate, X_train.iloc[test_rows])
    (shared_threshold, shared_cv), _ = tune_scalar(
        y_train,
        lambda t: hurdle_predict(shared_none_oof, shared_oof, t),
        np.linspace(0.20, 0.80, 25),
    )
    long_X, long_y = long_family_table(X_train, y_train)
    shared_model = logistic().fit(long_X, long_y)
    shared_gate = fit_none_model(X_train, y_train)
    results["shared_family_ranker"] = hurdle_predict(
        none_probability(shared_gate, X_valid),
        shared_family_scores(shared_model, X_valid),
        shared_threshold,
    )
    choices["shared_family_ranker"] = (
        f"none threshold={shared_threshold:.3f}; train OOF={shared_cv:.4f}"
    )

    # 3. Semi-global experts: KMeans routing plus shrinkage toward global LogReg.
    cluster_candidates = []
    for n_clusters in (2, 3, 4):
        local_oof = np.zeros_like(baseline_oof)
        for fit_rows, test_rows in folds:
            model = fit_cluster_experts(
                X_train.iloc[fit_rows], y_train.iloc[fit_rows], n_clusters, alpha=1.0
            )
            local_oof[test_rows] = model.predict_proba(X_train.iloc[test_rows])
        for alpha in (0.25, 0.50, 0.75, 1.0):
            combined = (1 - alpha) * baseline_oof + alpha * local_oof
            score = macro_f1(y_train, predictions_from_proba(combined))
            cluster_candidates.append((score, n_clusters, alpha))
    cluster_cv, best_k, best_alpha = max(cluster_candidates)
    clustered = fit_cluster_experts(X_train, y_train, best_k, best_alpha)
    results["clustered_logistic_experts"] = predictions_from_proba(
        clustered.predict_proba(X_valid)
    )
    choices["clustered_logistic_experts"] = (
        f"k={best_k}, local weight={best_alpha:.2f}; train OOF={cluster_cv:.4f}"
    )

    # 4. Simple combination of mechanistic rule and statistical model.
    rule_train = rule_predict(X_train)
    rule_valid = rule_predict(X_valid)
    rule_train_one_hot = one_hot_predictions(rule_train)
    (rule_weight, ensemble_cv), _ = tune_scalar(
        y_train,
        lambda w: predictions_from_proba(
            (1 - w) * baseline_oof + w * rule_train_one_hot
        ),
        np.linspace(0.0, 0.60, 13),
    )
    ensemble_valid_p = (1 - rule_weight) * baseline_valid_p + rule_weight * one_hot_predictions(
        rule_valid
    )
    results["rule_logreg_blend"] = predictions_from_proba(ensemble_valid_p)
    choices["rule_logreg_blend"] = (
        f"rule weight={rule_weight:.2f}; train OOF={ensemble_cv:.4f}"
    )

    print("\nSelections made using train OOF only:")
    for name, detail in choices.items():
        print(f"  {name:28s} {detail}")

    print("\nUntouched validation results:")
    rows = []
    for name, predictions in results.items():
        score = macro_f1(y_valid, predictions)
        per_class = f1_score(
            y_valid,
            predictions,
            labels=ALL_LABELS,
            average=None,
            zero_division=0,
        )
        rows.append({"model": name, "macro_f1": score, **dict(zip(ALL_LABELS, per_class))})
    report = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print(report.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    report.to_csv("simple_model_comparison.csv", index=False)
    print("\nWrote simple_model_comparison.csv")


if __name__ == "__main__":
    main()
