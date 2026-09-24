"""Maps transactions to one of the 7 target merchant-family categories.

Description keywords are checked FIRST, across all MCCs - they turned out
to be more precise than MCC alone. Auditing the raw data showed:
  - MCC has real cross-contamination: e.g. ~8% of "fit"/"gym" keyword
    transactions carry an mcc other than 7997 (gym), landing under 5812
    (eating places) or 5411 (groceries) instead. Gating category purely by
    mcc would mis-drop those as non-target spend.
  - The two heaviest-traffic MCCs are overloaded with unrelated real
    merchants: 5812 (Eating Places) mixes real dining with streaming/music
    subscriptions; 5732 (Electronics Stores) mixes real electronics/
    marketplace spend with cloud storage/backup subscriptions.
  - Some transactions rotate their description month to month for the
    SAME recurring subscription (e.g. one client's charges alternate
    between "digital plus", "premium plan", "media streaming", "video
    access" at a near-constant amount) - a fraction of those cycles carry
    no category keyword at all. This can't be resolved per-transaction;
    see recurrence.py's amount-based clustering, which imputes the
    category from sibling transactions in the same recurring stream.

MCC is used only as a fallback for the four "clean" MCCs, where it is
reliable even without a keyword hit (e.g. "monthly plan" under 6300 is
still insurance).
"""

CLEAN_MCC_CATEGORY = {
    "4814": "mobile",
    "5734": "software",
    "6300": "insurance",
    "7997": "gym",
}

TOKEN_CATEGORY = {
    # music checked before streaming: "audio streaming" contains both
    # "audio" and "streaming" tokens, and is a music subscription - music
    # must win that overlap, so it's ordered first.
    "music": {"audio"},
    # "media" added after checking data/processed/unique_train_descriptions.csv:
    # 98% of "media"-containing transactions already carry "stream"/
    # "streaming" too (e.g. "media streaming"); the 41 remaining bare
    # "media"/"media digital"/"media plus" transactions never co-occur
    # with "audio" anywhere in the dataset, so there's no music ambiguity
    # - they're the same rotating-description pattern as everywhere else,
    # just missing the "streaming" suffix in that cycle.
    "streaming": {"stream", "streaming", "video", "media"},
    "software": {"saas", "software", "productivity"},
    "mobile": {"phone"},
    "cloud": {"cloud", "backup", "storage"},
    "gym": {"gym", "fit", "fitness"},
    "insurance": {"insurance", "cover", "policy"},
}

# Tokens confirmed (by manual audit) to belong to real, non-subscription
# merchants that happen to share an overloaded MCC. Matched per-token, not
# as exact full-description phrases: the description generator appends
# template suffixes ("core", "plus", "digital", "online", "service",
# "billing ...", "member ...", "pay ...") to these same base merchants
# (e.g. "casual dining" -> "billing casual dining core"). An exact-phrase
# match against just "casual dining" misses ~100 such variant transactions
# per merchant (~300+ total across dining/grocery alone) - checked via
# data/processed/unique_train_descriptions.csv. These must not be pooled
# with genuine subscription-like charges when clustering by amount/cadence.
NON_SUBSCRIPTION_LITERAL_TOKENS = {
    "5812": {"coffee", "casual", "dining", "fresh", "foods", "grocery", "neighborhood", "market"},
    "5732": {"electronics", "marketplace"},
    "4111": {"ride"},
    # "coffee shop" / "electronics shop" cross-contamination lands on
    # these clean mccs too (found checking the "shop" token). Without
    # this, e.g. a corrupted "coffee shop" transaction landing on 5734
    # gets auto-labeled "software" via the clean-mcc fallback in
    # classify() - excluding it from the pool here stops that.
    "4814": {"shop", "coffee", "electronics"},
    "5411": {"shop", "coffee", "electronics"},
    "5734": {"shop", "coffee", "electronics"},
}
# Tried adding "7011": {"hotel"} (relevant once AMOUNT_CLUSTER_ALL_MCCS is
# enabled in recurrence.py, since one-off hotel bookings could otherwise
# coincidentally form spurious "recurring" amount clusters). Measured:
# logreg_l1 unaffected (0.3784 -> 0.3784), hgb_tuned worse (0.4029 ->
# 0.3956) - reversed part of the "media" keyword gain. Reverted; the best
# config found so far is media-keyword-added + no hotel exclusion.

# MCCs with no plausible real-world subscription meaning (transport,
# groceries, pharmacy, ATM, financial/salary/p2p, hotel - see
# data/mcc_frequency.json, label_category: null). Tested hard-excluding
# these from ever getting a target category (vetoing keyword matches like
# "gym membership" under mcc 5411) on the hypothesis that such matches are
# noise. Measured worse (macro-F1 0.3982 -> 0.3821): the sample
# descriptions on these mccs look like genuine subscription text with a
# corrupted mcc field, not coincidental noise, so trusting mcc as a veto
# loses more recall than it gains precision in this dataset. Keep empty -
# current keyword-first behavior is better here despite the real-world
# intuition that e.g. groceries shouldn't be subscriptions.
HARD_EXCLUDE_MCCS: set[str] = set()

TARGET_CATEGORIES = ["cloud", "gym", "insurance", "mobile", "music", "software", "streaming"]


def classify(mcc: str, description: str) -> str | None:
    """Return the target category for a single transaction, or None.

    None means "no keyword evidence and not a clean-mapped mcc" - the
    transaction may still turn out to be part of a recurring subscription
    stream once merged with same-amount siblings in recurrence.py.
    """
    if mcc in HARD_EXCLUDE_MCCS:
        return None

    tokens = set(description.split())

    for category, keywords in TOKEN_CATEGORY.items():
        if tokens & keywords:
            return category

    if mcc in CLEAN_MCC_CATEGORY:
        return CLEAN_MCC_CATEGORY[mcc]

    return None
