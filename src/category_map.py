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
    "music": {"audio", "pass"},  # "member pass" = untagged music description
    "streaming": {"stream", "streaming", "video"},
    "software": {"saas", "software", "productivity"},
    "mobile": {"phone"},
    "cloud": {"cloud", "backup", "storage"},
    "gym": {"gym", "fit", "fitness"},
    "insurance": {"insurance", "cover", "policy"},
}

# Real, non-subscription merchants. Like subscriptions, they rotate prefixes/
# suffixes ("billing grocery store plus", "coffee shop core") and hop MCCs, so
# they are matched as substrings, on any MCC. They must not be pooled with
# genuine subscription-like charges when clustering by amount/cadence.
NON_SUBSCRIPTION_PHRASES = [
    "atm withdrawal", "p2p send", "service fee", "casual dining", "coffee shop",
    "electronics shop", "pharmacy", "neighborhood market", "ride share",
    "hotel booking", "fresh foods", "online marketplace", "grocery store",
]
NON_SUBSCRIPTION_TYPES = {"atm", "fee", "p2p_transfer"}

TARGET_CATEGORIES = ["cloud", "gym", "insurance", "mobile", "music", "software", "streaming"]


def classify(mcc: str, description: str) -> str | None:
    """Return the target category for a single transaction, or None.

    None means "no keyword evidence and not a clean-mapped mcc" - the
    transaction may still turn out to be part of a recurring subscription
    stream once merged with same-amount siblings in recurrence.py.
    """
    tokens = set(description.split())

    for category, keywords in TOKEN_CATEGORY.items():
        if tokens & keywords:
            return category

    if mcc in CLEAN_MCC_CATEGORY:
        return CLEAN_MCC_CATEGORY[mcc]

    return None
