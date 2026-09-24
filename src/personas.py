import numpy as np
import pandas as pd

MCC_NAMES = {
    "4111": "transport", "4814": "telecom", "5411": "grocery", "5732": "electronics",
    "5734": "software", "5812": "dining", "5912": "pharmacy", "6011": "atm",
    "6012": "financial", "6300": "insurance", "7011": "hotel", "7997": "gym",
}
RECURRING_MCCS = ["4814", "6300", "5734", "7997"]


def augment(d: pd.DataFrame) -> pd.DataFrame:
    """Derive the time/currency helper columns client_features expects,
    for a frame that was already loaded elsewhere."""
    d = d.copy()
    d["month"] = d["timestamp"].dt.strftime("%Y-%m")
    d["day"] = d["timestamp"].dt.day
    d["hour"] = d["timestamp"].dt.hour
    d["weekend"] = d["timestamp"].dt.dayofweek >= 5
    d["night"] = (d["hour"] >= 22) | (d["hour"] < 5)
    return d


def load(path):
    d = pd.read_json(path, lines=True, dtype={"mcc": str})
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    return augment(d)


def _cv(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0.0
    m = x.mean()
    return float(x.std() / m) if m > 0 else 0.0


def client_features(g):
    f = {}
    n = len(g)
    months = max(g["month"].nunique(), 1)
    home = g["currency"].value_counts().index[0]
    out = g[g["direction"] == "out"]
    inn = g[g["direction"] == "in"]
    pay = g[g["type"] == "card_payment"]
    # FIX: salary must be an inflow; a description match alone can pick up
    # outgoing rows (e.g. a salary-labelled onward transfer).
    sal = g[(g["description"] == "salary") & (g["direction"] == "in")]
    atm = g[g["type"] == "atm"]
    # FIX: a savings sweep is money leaving the account, so only outgoing
    # transfers count. Incoming transfers were inflating savings_sweep_rate.
    trf_out = g[(g["type"] == "transfer") & (g["direction"] == "out")]
    foreign = g[g["currency"] != home]

    # volume
    f["tx_per_month"] = n / months
    f["log_in_per_month"] = np.log1p(inn["amount"].sum() / months)
    f["log_out_per_month"] = np.log1p(out["amount"].sum() / months)
    tot_in, tot_out = inn["amount"].sum(), out["amount"].sum()
    f["savings_rate"] = float(np.clip((tot_in - tot_out) / tot_in, -1, 1)) if tot_in > 0 else -1.0

    # salary
    f["salary_months_share"] = sal["month"].nunique() / months
    f["log_salary"] = np.log1p(sal["amount"].median()) if len(sal) else 0.0
    f["salary_day_std"] = float(sal["day"].std()) if len(sal) > 1 else 15.0
    f["salary_amount_cv"] = _cv(sal["amount"])
    f["salary_foreign"] = float((sal["currency"] != home).mean()) if len(sal) else 0.0
    other_in = inn[inn["description"] != "salary"]
    f["share_nonsalary_in"] = other_in["amount"].sum() / tot_in if tot_in > 0 else 0.0
    f["inflow_amount_cv"] = _cv(inn["amount"])

    # big transfer shortly after salary (savings sweep)
    ts, ta = trf_out["timestamp"].values, trf_out["amount"].values
    sweeps = sum(bool(((ts > t) & (ts <= t + np.timedelta64(7, "D")) & (ta >= 0.15 * a)).any())
                 for t, a in zip(sal["timestamp"].values, sal["amount"].values))
    f["savings_sweep_rate"] = sweeps / len(sal) if len(sal) else 0.0

    # payment size profile
    amt = pay["amount"] if len(pay) else pd.Series([0.0])
    f["log_median_payment"] = np.log1p(amt.median())
    f["log_p90_payment"] = np.log1p(amt.quantile(0.9))
    f["share_small_pay"] = float((amt < 20).mean())
    f["share_large_pay"] = float((amt > 500).mean())
    f["amount_cv"] = _cv(out["amount"])

    # channel mix
    for t in ["card_payment", "atm", "p2p_transfer", "transfer", "refund"]:
        f[f"share_{t}"] = float((g["type"] == t).mean())
    f["log_median_atm"] = np.log1p(atm["amount"].median()) if len(atm) else 0.0
    f["fee_per_tx"] = g["fee"].sum() / n

    # currency
    f["share_foreign_ccy"] = len(foreign) / n
    fm = foreign.groupby("month").size().reindex(sorted(g["month"].unique()), fill_value=0)
    f["foreign_month_coverage"] = float((fm > 0).mean())
    f["foreign_burstiness"] = _cv(fm) if fm.sum() > 0 else 0.0
    f["share_atm_foreign"] = float((atm["currency"] != home).mean()) if len(atm) else 0.0

    # timing
    f["share_weekend"] = float(g["weekend"].mean())
    f["share_night"] = float(g["night"].mean())
    f["share_weekday_daytime"] = float((~g["weekend"] & g["hour"].between(8, 17)).mean())
    f["share_month_end"] = float((out["day"] >= 25).mean()) if len(out) else 0.0
    f["activity_burstiness"] = _cv(g.groupby(g["timestamp"].dt.tz_localize(None).dt.to_period("W")).size())

    # merchant mix
    # FIX: denominator is now ALL outgoing rows, not just the ones with a
    # non-null MCC. value_counts(normalize=True) silently dropped NaN, so a
    # client with 90% uncategorised spend looked identical to one with 0%.
    n_out = len(out)
    mcc_counts = out["mcc"].value_counts(dropna=True)
    for code, name in MCC_NAMES.items():
        f[f"mcc_{name}"] = float(mcc_counts.get(code, 0)) / n_out if n_out else 0.0
    f["mcc_coverage"] = float(mcc_counts.sum()) / n_out if n_out else 0.0
    # entropy still over the categorised part only (it describes the mix, not the coverage)
    p = (mcc_counts / mcc_counts.sum()).values if mcc_counts.sum() > 0 else np.array([])
    f["mcc_entropy"] = float(-(p * np.log(p)).sum()) if len(p) else 0.0

    # recurring streams: fixed-merchant bills seen in >= 3 distinct months
    rec = out[out["mcc"].isin(RECURRING_MCCS)]
    f["n_recurring_streams"] = int((rec.groupby("mcc")["month"].nunique() >= 3).sum())
    return pd.Series(f)


def build_features(df):
    X = df.groupby("client_id").apply(client_features, include_groups=False)
    return _sanitise(X)


def _sanitise(X):
    """FIX: NaN sorts above every real value in np.searchsorted, so a single
    NaN feature silently scored as the 100th percentile — the strongest
    possible positive evidence. Force non-finite values to 0.0 and shout
    about it rather than letting it through."""
    X = X.astype(float)
    bad = ~np.isfinite(X.values)
    if bad.any():
        cols = X.columns[bad.any(axis=0)].tolist()
        import warnings
        warnings.warn(f"non-finite feature values coerced to 0.0 in: {cols}")
        X = X.mask(~np.isfinite(X), 0.0)
    return X


# persona -> {group_name: (weight, [features])}
# FIX: correlated signals are grouped. Previously share_foreign_ccy,
# foreign_month_coverage and share_atm_foreign each contributed separately,
# so one underlying fact ("this client spends abroad") was counted three
# times while mcc_transport was counted once. Features inside a group are
# averaged first, then the group gets a single weight.
PERSONA_RULES = {
    "Steady Saver": {
        "regular_income": (1, ["salary_months_share"]),
        "saves": (2, ["savings_rate", "savings_sweep_rate"]),
        "stable_spend": (-1, ["amount_cv"]),
        "low_fees": (-1, ["fee_per_tx"]),
        "staples": (1, ["mcc_grocery"]),
        "low_card": (-1, ["share_card_payment"]),
    },
    "Paycheck-to-Paycheck Household": {
        "regular_income": (1, ["salary_months_share"]),
        "no_buffer": (-2, ["savings_rate"]),
        "fixed_bills": (2, ["n_recurring_streams", "mcc_telecom", "mcc_insurance"]),
        "staples": (1, ["mcc_grocery"]),
        "month_end": (1, ["share_month_end"]),
    },
    "Digital Micro-Spender": {
        "high_frequency": (2, ["tx_per_month"]),
        "small_tickets": (2, ["share_small_pay"]),
        "low_median": (-2, ["log_median_payment"]),
        "p2p": (1, ["share_p2p_transfer"]),
        "digital_mcc": (1, ["mcc_electronics", "mcc_software"]),
        "night": (1, ["share_night"]),
    },
    "Frequent Traveller": {
        "abroad": (2, ["share_foreign_ccy", "share_atm_foreign"]),
        "fees": (2, ["fee_per_tx"]),
        "hotels": (2, ["mcc_hotel"]),
        "bursty": (1, ["foreign_burstiness", "activity_burstiness"]),
    },
    "Cross-Border Commuter": {
        "abroad": (1, ["share_foreign_ccy"]),
        "every_month": (2, ["foreign_month_coverage"]),
        "not_bursty": (-1, ["foreign_burstiness"]),
        "foreign_pay": (1, ["salary_foreign"]),
        "transport": (2, ["mcc_transport"]),
        "staples": (1, ["mcc_grocery"]),
        "not_hotels": (-2, ["mcc_hotel"]),
    },
    "Cash-Reliant Client": {
        "atm": (3, ["share_atm", "log_median_atm"]),
        "low_card": (-1, ["share_card_payment"]),
        "narrow_mix": (-1, ["mcc_entropy"]),
        "no_p2p": (-1, ["share_p2p_transfer"]),
        "office_hours": (1, ["share_weekday_daytime"]),
    },
    "Affluent Big Spender": {
        "income": (1, ["log_in_per_month"]),
        "outflow": (2, ["log_out_per_month"]),
        "big_tickets": (2, ["log_p90_payment", "share_large_pay"]),
        "discretionary": (1, ["mcc_electronics", "mcc_dining"]),
        "variable": (1, ["amount_cv"]),
    },
    "Freelancer / Irregular Earner": {
        "no_salary": (-2, ["salary_months_share"]),
        "other_income": (2, ["share_nonsalary_in"]),
        "lumpy": (1, ["inflow_amount_cv", "amount_cv"]),
        "software": (1, ["mcc_software"]),
        "office_hours": (2, ["share_weekday_daytime"]),
    },
    "Weekend Socialiser": {
        "weekend": (2, ["share_weekend"]),
        "night": (2, ["share_night"]),
        "going_out": (1, ["mcc_dining", "mcc_transport"]),
        "p2p": (1, ["share_p2p_transfer"]),
        "no_buffer": (-1, ["savings_rate"]),
    },
    "Serial Returner": {
        "refunds": (3, ["share_refund"]),
        "electronics": (1, ["mcc_electronics"]),
        "variable": (1, ["amount_cv"]),
        "card": (1, ["share_card_payment"]),
    },
}


class PersonaScorer:
    """Scores personas via percentile ranks against a reference (training)
    population, then calibrates each persona's score distribution so the
    argmax across personas is a fair comparison."""

    def __init__(self, calibrate=True):
        self.calibrate = calibrate

    def fit(self, X):
        X = _sanitise(X)
        self.ref_ = {c: np.sort(X[c].values) for c in X.columns}
        self.n_ref_ = len(X)
        # FIX: each persona score is a weighted mean of percentiles, so its
        # spread depends on how many (and how correlated) its rules are.
        # Serial Returner, dominated by one rule, swung far wider than
        # Cross-Border Commuter, whose seven rules averaged out near 0.5 —
        # so the argmax systematically favoured the narrow personas.
        # Standardising each persona column against the reference population
        # puts them on a common scale.
        S = self._raw_scores(X)
        self.mu_ = S.mean()
        self.sigma_ = S.std().replace(0.0, 1.0)
        return self

    def _pct(self, X, col):
        # FIX: was (lo + hi) / (2 * len(ref)) — a midrank. For zero-inflated
        # features (share_refund, mcc_hotel, salary_foreign, ...) a client
        # with 0 landed mid-pack: with 80% zeros in the reference, "has none
        # of this" scored ~0.40 instead of ~0.0, diluting every persona keyed
        # on a rare signal. Using the left insertion point puts ties at the
        # bottom of their block, so absence reads as absence — and, for
        # negative weights, as strong evidence via (1 - p).
        ref = self.ref_[col]
        lo = np.searchsorted(ref, X[col].values, side="left")
        return lo / len(ref)

    def _raw_scores(self, X):
        out = {}
        for persona, rules in PERSONA_RULES.items():
            total = sum(abs(w) for w, _ in rules.values())
            s = np.zeros(len(X))
            for w, cols in rules.values():
                # average within a correlated group, then apply the weight once
                p = np.mean([self._pct(X, c) for c in cols], axis=0)
                s += abs(w) * (p if w > 0 else 1 - p)
            out[persona] = s / total
        return pd.DataFrame(out, index=X.index)

    def scores(self, X, raw=False):
        X = _sanitise(X)
        S = self._raw_scores(X)
        if raw or not self.calibrate:
            return S
        return (S - self.mu_) / self.sigma_

    def assign(self, X):
        S = self.scores(X)
        top2 = np.sort(S.values, axis=1)[:, -2:]
        return pd.DataFrame({"persona": S.idxmax(axis=1),
                             "score": top2[:, 1],
                             "margin": top2[:, 1] - top2[:, 0],
                             "raw_score": self.scores(X, raw=True).max(axis=1)},
                            index=X.index)

    def diagnostics(self, X):
        """Per-persona score distribution and assignment share. If one persona
        takes most of the population, or a persona's sigma is far off the
        others, the rules for it need rebalancing rather than the scorer."""
        raw = self.scores(X, raw=True)
        cal = self.scores(X)
        counts = cal.idxmax(axis=1).value_counts()
        return pd.DataFrame({
            "raw_mean": raw.mean(),
            "raw_std": raw.std(),
            "n_assigned": counts.reindex(raw.columns, fill_value=0),
            "share_assigned": counts.reindex(raw.columns, fill_value=0) / len(X),
        }).sort_values("share_assigned", ascending=False)