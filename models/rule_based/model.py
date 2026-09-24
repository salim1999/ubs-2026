"""Due-date rule for the next recurring merchant family.

This model has no fitted parameters. It applies the same transparent decision
rule to the recurring-stream features for every client.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.category_map import TARGET_CATEGORIES


def rule_predict(features: pd.DataFrame) -> np.ndarray:
    """Predict the next recurring family from live-stream due dates.

    Predict ``none`` when the client has no live recurring stream or when the
    strongest live stream looks like a short 3--4-payment trial. Otherwise,
    predict the live family closest to its expected next payment date.

    Required columns are ``is_live_<family>``, ``over_days_<family>``, and
    ``short_live_stream`` as produced by :mod:`src.features`.
    """
    predictions = []
    for _, client in features.iterrows():
        live_families = [
            (family, client[f"over_days_{family}"])
            for family in TARGET_CATEGORIES
            if client[f"is_live_{family}"] == 1
        ]

        if not live_families or client["short_live_stream"] == 1:
            predictions.append("none")
            continue

        # over_days = recency - mean gap. The largest value corresponds to
        # the smallest number of days remaining until the expected payment.
        family_due_soonest = max(live_families, key=lambda item: item[1])[0]
        predictions.append(family_due_soonest)

    return np.asarray(predictions)
