"""Unsupervised client embeddings learned on the unlabeled pretrain clients.

`unlabeled_pretrain_transactions.jsonl` holds 10,000 client histories (750k
rows, same 2024-11 .. 2025-12 window as train, no labels). The README allows
it "to learn general transaction patterns before training a supervised
model". This module does that without any labels:

  1. Each client becomes a bag of family tokens: one "f:<family>" per
     outgoing, non-refund row that category_map.classify resolves to a
     target family (strong or clean-mcc evidence, as in recurrence.py),
     repeated with a "recent:" prefix for rows in the last RECENT_DAYS, so
     the embedding sees what the client pays for now and not only what it
     used to pay for.
  2. TF-IDF (sublinear counts) -> TruncatedSVD with N_COMPONENTS dimensions:
     a latent "subscription profile" (LSA), fit on the unlabeled clients only.
  3. KMeans with N_CLUSTERS on the L2-normalized embeddings: behavioural
     segments, again fit on the unlabeled clients only. The distance to
     each centroid is a feature, so trees can use soft membership.

Why family tokens only (audit, 2026-09-25): a first version also used raw
descriptions, mccs and transaction types. Its embedding mainly encoded
which split a client came from. Four opaque descriptions ("digital
order", "card purchase", "merchant charge", "service payment") never occur
in the pretrain data, but appear for 16% of train, 94% of valid and 98% of
test clients (4-6% of valid/test rows), replacing phrases like "premium
plan" / "digital plus". The KMeans segment holding those clients was 9% of
train (all 'none') but 37% of valid and 61% of test, adversarial AUC
train-vs-valid was 0.92 and train->valid macro-F1 fell to ~0.16. With
family tokens only, segment shares match across splits and adversarial AUC
drops to 0.77; adding mcc tokens pushes it back to 0.83.

The fitted pipeline is cached in PIPELINE_PATH. train/valid/test/unlabeled
clients are only *transformed*, so no labels and no labelled client ever
reach the fit. Features only use transactions <= the cutoff.

Pseudo-labels from the embedding (--write-labels): a kNN classifier
(LABEL_K neighbours, distance-weighted) in the embedding space, anchored on
the labelled train clients only (LABEL_ANCHOR_SPLITS), labels the pretrain
clients; confidence = its class probability. Valid is kept out of the
anchors so a model trained on these pseudo-labels can still be scored on
valid without having seen its labels through the kNN. On valid (anchored on train only) this
scores macro-F1 0.354 (segment majority 0.354, label spreading 0.331), below
the label-free stream rule (0.497) and the rank-teacher pseudo-labels of
src.pseudo_label: the family profile carries no timing, which is what
"next" depends on.

Usage:
    python -m src.unsupervised [--refit]   # fit (or load) and print a segment summary
    python -m src.unsupervised --write-labels   # -> LABELS_OUT_PATH
"""

from __future__ import annotations

import argparse
import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from src.category_map import TARGET_CATEGORIES

DATA_DIR = "data/dataset/dataset"
UNLABELED_PATH = f"{DATA_DIR}/unlabeled_pretrain_transactions.jsonl"
PIPELINE_PATH = "data/processed/unsup_pipeline.pkl"
CUTOFF = "2026-01-01"
SEED = 0

LABELS_OUT_PATH = "data/labeled_pretrain_clients_unsupervised.csv"
LABEL_K = 50
LABEL_ANCHOR_SPLITS = ("train",)  # never valid: it must stay a clean holdout

RECENT_DAYS = 90
N_COMPONENTS = 8  # the vocabulary is only 14 tokens (7 families x all/recent)
N_CLUSTERS = 10


def _client_documents(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """One space-separated family-token string per client, from rows <= cutoff.

    df is the output of recurrence.load_transactions (per-row "category").
    Clients without any family-resolved charge get no document (NaN features).
    """
    out = df[
        (df["direction"] == "out") & ~df["is_refund"]
        & df["category"].isin(TARGET_CATEGORIES) & (df["timestamp"] <= cutoff)
    ]
    base = "f:" + out["category"]
    recent = out["timestamp"] > cutoff - pd.Timedelta(days=RECENT_DAYS)
    tokens = base.where(~recent, base + " recent:" + base)
    return tokens.groupby(out["client_id"]).agg(" ".join)


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True)),
        ("svd", TruncatedSVD(n_components=N_COMPONENTS, random_state=SEED)),
        ("norm", Normalizer()),
    ])


def fit(unlabeled_path: str = UNLABELED_PATH, cutoff: str = CUTOFF) -> dict:
    from src.recurrence import load_transactions  # heavy; only needed when fitting

    df = load_transactions(unlabeled_path)
    docs = _client_documents(df, pd.Timestamp(cutoff, tz="UTC"))
    embed = _build_pipeline()
    Z = embed.fit_transform(docs)
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED).fit(Z)
    model = {"embed": embed, "kmeans": kmeans, "n_clients": len(docs)}
    pathlib.Path(PIPELINE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(PIPELINE_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"unsupervised: fit on {len(docs)} unlabeled clients -> {PIPELINE_PATH}")
    return model


def load(refit: bool = False) -> dict:
    if refit or not pathlib.Path(PIPELINE_PATH).exists():
        return fit()
    with open(PIPELINE_PATH, "rb") as f:
        return pickle.load(f)


def client_features(df: pd.DataFrame, cutoff: pd.Timestamp, model: dict | None = None) -> pd.DataFrame:
    """Embedding + centroid distances + segment id, indexed by client_id."""
    model = model or load()
    docs = _client_documents(df, cutoff)
    Z = model["embed"].transform(docs)
    dist = model["kmeans"].transform(Z)
    out = pd.DataFrame(Z, index=docs.index, columns=[f"unsup_svd{i:02d}" for i in range(Z.shape[1])])
    for j in range(dist.shape[1]):
        out[f"unsup_dist{j:02d}"] = dist[:, j]
    out["unsup_segment"] = dist.argmin(axis=1)
    return out


def _embedding(df: pd.DataFrame, clients, model: dict) -> np.ndarray:
    """SVD embedding per client (zeros for clients without family charges)."""
    cols = [f"unsup_svd{i:02d}" for i in range(N_COMPONENTS)]
    feats = client_features(df, pd.Timestamp(CUTOFF, tz="UTC"), model)
    return feats.reindex(clients)[cols].fillna(0.0).to_numpy()


def write_labels(model: dict, out_path: str = LABELS_OUT_PATH) -> pd.DataFrame:
    from sklearn.neighbors import KNeighborsClassifier

    from src.recurrence import load_transactions

    X, y = [], []
    for split in LABEL_ANCHOR_SPLITS:
        labels = pd.read_csv(f"{DATA_DIR}/{split}_labels.csv")
        X.append(_embedding(load_transactions(f"{DATA_DIR}/{split}_transactions.jsonl"), labels["client_id"], model))
        y.append(labels["target_next_recurring_merchant"])
    knn = KNeighborsClassifier(LABEL_K, weights="distance").fit(np.vstack(X), pd.concat(y))

    unl = load_transactions(UNLABELED_PATH)
    clients = np.sort(unl["client_id"].unique())
    proba = knn.predict_proba(_embedding(unl, clients, model))
    out = pd.DataFrame({
        "client_id": clients,
        "cutoff_date": CUTOFF,
        "target_next_recurring_merchant": knn.classes_[proba.argmax(1)],
        "confidence": proba.max(1).round(4),
    })
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out)} labels to {out_path}")
    print(out["target_next_recurring_merchant"].value_counts(normalize=True).round(3).to_string())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refit", action="store_true", help="refit even if a cached pipeline exists")
    parser.add_argument("--write-labels", action="store_true", help=f"pseudo-label the pretrain clients -> {LABELS_OUT_PATH}")
    args = parser.parse_args()

    from src.recurrence import load_transactions

    model = load(args.refit)
    if args.write_labels:
        write_labels(model)
        return
    svd = model["embed"].named_steps["svd"]
    print(f"explained variance ({N_COMPONENTS} comps): {svd.explained_variance_ratio_.sum():.3f}")

    # Segment sizes, and how the labelled train clients' targets fall into them.
    # Labels are read here for the printout only; they never touch the fit.
    train = load_transactions(f"{DATA_DIR}/train_transactions.jsonl")
    feats = client_features(train, pd.Timestamp(CUTOFF, tz="UTC"), model)
    labels = pd.read_csv(f"{DATA_DIR}/train_labels.csv").set_index("client_id")["target_next_recurring_merchant"]
    seg = feats["unsup_segment"].rename("segment").to_frame().join(labels)
    print(pd.crosstab(seg["segment"], seg["target_next_recurring_merchant"], normalize="index").round(2))


if __name__ == "__main__":
    main()
