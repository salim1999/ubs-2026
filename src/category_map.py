"""Maps transactions to one of the 7 target merchant-family categories.

The description vocabulary is generated from a small set of canonical
phrases per family, then corrupted with noise:
  - affixes: "pay/member/billing <phrase>", "<phrase> core/online/plus/
    digital/service";
  - abbreviations / truncations: "prem plan", "insurance mth", "dgtl plus",
    "prod suite", "safe", "urban", "club", ...;
  - ~5% of subscription charges land on an unrelated mcc (e.g. a mobile
    "digital plus" on 5734, "software access" on 5812).

Auditing the train data (description co-occurrence inside same-amount
clusters) gives the canonical phrases below. Four phrases per family, except
that music and streaming (both on 5812) share "digital plus" and
"premium plan", and "digital plus" / "premium plan" are also canonical for
mobile / software respectively. Such generic phrases carry no family on
their own; they are resolved by mcc and, in recurrence.py, by matching
their amount to a sibling stream that does carry a family-specific phrase.

classify() therefore returns (category, strength):
  - "strong":  family-specific phrase or token - trusted regardless of mcc;
  - "mcc":     generic subscription phrase resolved via a clean mcc;
  - "generic": generic subscription phrase on an mcc that can't resolve it
               (category is None, or MEDIA for 5812 = music-or-streaming).
Non-subscription spend (groceries, dining, ATM, ...) returns (None, None).
"""

TARGET_CATEGORIES = ["cloud", "gym", "insurance", "mobile", "music", "software", "streaming"]

# music-or-streaming, undecided: resolved at stream level.
MEDIA = "media"

CANONICAL_PHRASES = {
    "phone contract": "mobile",
    "service bill": "mobile",
    "cloud access": "cloud",
    "storage plan": "cloud",
    "service plan": "cloud",
    "cloud backup": "cloud",
    "saas billing": "software",
    "productivity suite": "software",
    "software access": "software",
    "cover plan": "insurance",
    "insurance monthly": "insurance",
    "safe cover": "insurance",
    "policy premium": "insurance",
    "urban gym": "gym",
    "gym membership": "gym",
    "fit club": "gym",
    "fitness monthly": "gym",
    "media streaming": "streaming",
    "video access": "streaming",
    "audio streaming": "music",
    "member pass": "music",
}

# Family-specific single tokens, used when no canonical phrase survives
# normalisation (truncated descriptions like "safe", "suite", "urban").
STRONG_TOKENS = {
    "phone": "mobile", "contract": "mobile", "bill": "mobile",
    "cloud": "cloud", "storage": "cloud", "backup": "cloud",
    "saas": "software", "software": "software", "productivity": "software", "suite": "software",
    "insurance": "insurance", "cover": "insurance", "policy": "insurance", "safe": "insurance",
    "gym": "gym", "fit": "gym", "fitness": "gym", "club": "gym", "urban": "gym", "membership": "gym",
    "video": "streaming", "media": "streaming",
    "audio": "music", "pass": "music",
}
# "stream(ing)" alone could be either 5812 family.
MEDIA_TOKENS = {"stream", "streaming"}

GENERIC_PHRASES = {
    "digital plus", "premium plan", "monthly plan", "member plan",
    "digital service", "subscription charge",
}
GENERIC_TOKENS = {"plan", "monthly", "premium", "digital", "service", "plus", "access", "member", "subscription", "charge", "billing"}

# Real, non-subscription merchants and one-off charges. Tokens from these
# never count as subscription evidence.
NON_SUBSCRIPTION_PHRASES = {
    "coffee shop", "casual dining", "fresh foods", "grocery store", "neighborhood market",
    "electronics shop", "online marketplace", "ride share", "pharmacy", "hotel booking",
    "atm withdrawal", "salary", "p2p send", "p2p receive", "service fee",
    "digital order", "service payment", "merchant charge", "card purchase",
}
NON_SUBSCRIPTION_TOKENS = {
    "coffee", "shop", "dining", "casual", "fresh", "foods", "grocery", "neighborhood", "market",
    "electronics", "marketplace", "ride", "share", "pharmacy", "hotel", "booking", "atm",
    "withdrawal", "salary", "p2p", "fee", "order", "payment", "purchase",
}

CLEAN_MCC_CATEGORY = {
    "4814": "mobile",
    "5732": "cloud",
    "5734": "software",
    "6300": "insurance",
    "7997": "gym",
    "5812": MEDIA,
}

ABBREVIATIONS = {"prem": "premium", "mth": "monthly", "dgtl": "digital", "prod": "productivity"}
NOISE_PREFIXES = {"pay", "member", "billing"}
NOISE_SUFFIXES = {"core", "online", "plus", "digital", "service"}

_KNOWN = set(CANONICAL_PHRASES) | GENERIC_PHRASES | NON_SUBSCRIPTION_PHRASES


def normalize(description: str) -> str:
    """Expand abbreviations and peel noise affixes until a known phrase remains."""
    tokens = [ABBREVIATIONS.get(t, t) for t in description.lower().split()]
    changed = True
    while changed and " ".join(tokens) not in _KNOWN:
        changed = False
        if len(tokens) > 1 and tokens[-1] in NOISE_SUFFIXES:
            tokens, changed = tokens[:-1], True
        elif len(tokens) > 1 and tokens[0] in NOISE_PREFIXES:
            tokens, changed = tokens[1:], True
    return " ".join(tokens)


def classify(mcc: str, description: str) -> tuple[str | None, str | None]:
    """Return (category, strength) for one transaction; see module docstring."""
    phrase = normalize(description)

    if phrase in CANONICAL_PHRASES:
        return CANONICAL_PHRASES[phrase], "strong"
    if phrase in NON_SUBSCRIPTION_PHRASES:
        return None, None

    tokens = set(phrase.split())
    if tokens & NON_SUBSCRIPTION_TOKENS:
        return None, None

    strong = {STRONG_TOKENS[t] for t in tokens if t in STRONG_TOKENS}
    if len(strong) == 1:
        return strong.pop(), "strong"
    if not strong and tokens & MEDIA_TOKENS:
        return MEDIA, "generic"

    if phrase in GENERIC_PHRASES or (tokens and tokens <= GENERIC_TOKENS):
        if mcc in CLEAN_MCC_CATEGORY and CLEAN_MCC_CATEGORY[mcc] != MEDIA:
            return CLEAN_MCC_CATEGORY[mcc], "mcc"
        return (MEDIA if mcc == "5812" else None), "generic"

    return None, None
