"""Pseudo-labelled pretrain clients as extra training rows for the final model.

Labels: data/pretrain_labels_merged.csv, copied from Salim's branch
(origin/Salim, commit 34ae845, data/labeled_pretrain_labels_merged.csv). Each of
the 10,000 unlabeled pretrain clients got a label from three sources: an
unsupervised labeler, an LLM (gpt-4.1-mini) and a model trained on the 2,000
train clients only. None of them saw valid labels, so scoring on valid stays
clean. We keep only clients where all three sources agree (3,152 rows, on
top of the 2,000 train clients): valid macro-F1 of the sparse per-label
model 0.504 -> 0.513, while looser
filters (>= 2 agree, confidence cut-offs) did not help.

Pretrain features take ~5 min to build, so they are cached under
data/processed/, keyed by a hash of the feature code (a code change rebuilds).
"""

from __future__ import annotations

import hashlib
import os

import pandas as pd

from src.features import build_features
from src.model import LABEL_COL

PSEUDO_LABELS_PATH = "data/pretrain_labels_merged.csv"
PRETRAIN_PATH = "data/unlabeled_pretrain_transactions.jsonl"
MIN_SOURCES_AGREE = 3
CUTOFF = "2026-01-01"
_FEATURE_CODE = ("src/features.py", "src/recurrence.py", "src/category_map.py")


def _code_hash() -> str:
    h = hashlib.sha1()
    for path in _FEATURE_CODE:
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:10]


def pretrain_features() -> pd.DataFrame:
    cache = f"data/processed/pretrain_features_{_code_hash()}.pkl"
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    feats = build_features(PRETRAIN_PATH, CUTOFF)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    feats.to_pickle(cache)
    return feats


def pseudo_labelled_features(min_sources_agree: int = MIN_SOURCES_AGREE) -> pd.DataFrame:
    """client_id, LABEL_COL, features for pretrain clients whose pseudo-label sources agree."""
    labels = pd.read_csv(PSEUDO_LABELS_PATH)
    labels = labels.loc[labels["n_sources_agree"] >= min_sources_agree, ["client_id", LABEL_COL]]
    return labels.merge(pretrain_features(), on="client_id", how="inner")
