"""Pseudo-label the unlabeled pretrain clients with an OpenAI chat model.

`unlabeled_pretrain_transactions.jsonl` holds 10,000 extra client histories
(~750k transactions, same 2024-11-07 .. 2025-12-31 window as train/valid, no
rows past the cutoff) without a target. This script shows an LLM each
client's full pre-cutoff history, plus N_EXAMPLES labelled train clients as
few-shot examples, and asks for the next recurring merchant family. The
result is 5x more (noisy) training rows.

Validation exposure: none in the labels this script writes. Few-shot examples
and the prompt statistics come from train only, and the valid set is loaded
only by `--eval-valid`. Caveat: prompts v1-v3 below were compared on valid,
so that choice is mildly optimistic for valid. Tune further prompts with
`--eval-train` and keep `--eval-valid` for one final check. Pseudo-labels
written before this change (data/labeled_pretrain_*_chatgpt.*) used valid
few-shot examples and must not be used.

Why the prompt looks the way it does:
  - The LLM never sees the future either, so a pseudo-label is a prediction,
    not ground truth.
  - v1 (raw history only) scored macro-F1 0.275 on 198 held-out valid clients:
    it clustered ~75 noisy rows badly and answered `none` once in 198. v2
    therefore puts our detector's view first (recurrence.detect_streams +
    features.recurring_streams: streams with LIVE / STOPPED / trial status and
    next-due date, recent family charges outside any stream, and the
    model.rule_predict answer) followed by the raw history, plus the train-set
    frequencies for when to deviate from the rule (e.g. a single family charge
    15-30 days before the cutoff is the label 49% of the time, 63% with no
    live stream). v2 scored 0.39 on the same clients against 0.405 for the
    rule alone: better on most families, much worse on `none` (15/200 `none`
    answers). Its confidence is uninformative (0.6-0.9, accuracy flat).
  - v3 (this prompt) tells it to keep the rule's answer outside four specific
    cases. Macro-F1 0.375 vs rule 0.406 on the same 198 clients, 0.469 vs
    0.529 on 199 fresh ones. It still overrode the rule on 125/397 clients
    and was right 28 times where the rule was right 56. With gpt-4.1-mini
    the pseudo-labels are worse than model.rule_predict's, which cost nothing.
  - The phrase/mcc tables are generated from src.category_map, and every
    subscription-like row is annotated with its classify() result
    ("[gym]", "[media?]", ...), so the LLM starts from the same audited
    mapping as the feature pipeline and spends its effort on cadence,
    recency and cancellations.
  - The instructions and few-shot examples form one fixed prefix across all
    requests, so OpenAI's automatic prompt caching bills it at a discount.
  - Few-shot examples are stratified (one per label, extra `none`, which is
    ~29% of the labels). They come from the train set and are excluded from
    `--eval-train`. The pseudo-labels carry information from those
    N_EXAMPLES train clients, so strict CV would leave those clients out of
    scoring. At 10 of 3,000 clients, the effect is negligible.

Run `--eval-train 200` first: it scores the LLM on train clients that are
not few-shot examples (macro-F1, per class), which tells you how much to
trust the labels. `--eval-valid 200` does the same on valid clients, which
the prompt never sees.

Only the Python standard library is used for the API call (no `openai`
package), so pyproject.toml is unchanged.

Usage (from the repo root; the key is read from OPENAI_API_KEY):
    python -m src.extend_training_data --dry-run
    python -m src.extend_training_data --eval-train 200
    python -m src.extend_training_data --eval-valid 200   # final check only
    python -m src.extend_training_data --limit 500
    python -m src.extend_training_data            # all clients, resumable

Every result is saved immediately, with a full checkpoint every
CHECKPOINT_EVERY clients. If the API key runs out of credit the run stops
cleanly; re-running the same command continues where it stopped.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.category_map import (
    ABBREVIATIONS,
    CANONICAL_PHRASES,
    CLEAN_MCC_CATEGORY,
    GENERIC_PHRASES,
    MEDIA,
    NOISE_PREFIXES,
    NOISE_SUFFIXES,
    NON_SUBSCRIPTION_PHRASES,
    TARGET_CATEGORIES,
    classify,
)
from src.features import recurring_streams
from src.recurrence import detect_streams, load_transactions

DATA_DIR = "data/dataset/dataset"
UNLABELED_PATH = f"{DATA_DIR}/unlabeled_pretrain_transactions.jsonl"
TRAIN_TXN_PATH = f"{DATA_DIR}/train_transactions.jsonl"
TRAIN_LABELS_PATH = f"{DATA_DIR}/train_labels.csv"
VALID_TXN_PATH = f"{DATA_DIR}/valid_transactions.jsonl"
VALID_LABELS_PATH = f"{DATA_DIR}/valid_labels.csv"
# New file names so labels from the old valid-few-shot runs are never resumed or mixed in.
OUTPUT_PATH = "data/labeled_pretrain_transactions_chatgpt_trainonly.jsonl"
LABELS_CSV_PATH = "data/labeled_pretrain_labels_chatgpt_trainonly.csv"
EVAL_TRAIN_OUTPUT_PATH = "data/llm_train_eval.jsonl"
EVAL_VALID_OUTPUT_PATH = "data/llm_valid_eval_trainonly.jsonl"

CUTOFF = "2026-01-01"
HORIZON_DAYS = 90
LABELS = ["cloud", "gym", "insurance", "mobile", "music", "software", "streaming", "none"]
TARGET_COL = "target_next_recurring_merchant"

SEED = 0
N_EXAMPLES = 10
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_WORKERS = 8
API_URL = "https://api.openai.com/v1/chat/completions"
REQUEST_TIMEOUT_S = 120
MAX_RETRIES = 8
# Longest-live-stream lengths that look like a trial (model.rule_predict).
TRIAL_LENGTHS = {3, 4}
# Family charges outside any detected stream are listed if this recent. On
# train, a family with charges only >60 days before the cutoff was the label
# in ~1-5% of cases, so older ones are left to the raw history.
NEW_CHARGE_WINDOW_DAYS = 60
# Every result is appended to the jsonl (and flushed) as soon as it arrives;
# every CHECKPOINT_EVERY results the file is also fsynced and the labels CSV
# is re-exported, so a crash, Ctrl+C or an empty API balance loses nothing.
CHECKPOINT_EVERY = 500
# OpenAI error codes that no retry can fix: stop the whole run instead of
# burning through every remaining client with the same error.
FATAL_ERROR_CODES = {"insufficient_quota", "invalid_api_key", "billing_hard_limit_reached",
                     "account_deactivated", "model_not_found"}
CHARS_PER_TOKEN = 3.5  # rough, only for the --dry-run cost estimate


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_clients(path: str) -> dict[str, str]:
    """client_id -> prompt text: subscription summary followed by the full history."""
    cutoff = pd.Timestamp(CUTOFF, tz="UTC")
    df = load_transactions(path)
    df = df[df["timestamp"] <= cutoff]
    df = df.sort_values(["client_id", "timestamp", "amount", "description"], kind="mergesort")
    recurring = recurring_streams(detect_streams(df, cutoff), cutoff)
    family_charges = df[df["category"].isin(TARGET_CATEGORIES) & ~df["is_refund"] & (df["direction"] == "out")]
    rec_by_client = dict(tuple(recurring.groupby("client_id")))
    charges_by_client = dict(tuple(family_charges.groupby("client_id")))
    return {
        cid: summarize_client(rec_by_client.get(cid, recurring.iloc[:0]),
                              charges_by_client.get(cid, family_charges.iloc[:0]), cutoff)
             + "\n\n" + format_history(cid, txns)
        for cid, txns in df.groupby("client_id", sort=True)
    }


def rule_label(streams: pd.DataFrame) -> tuple[str, str]:
    """model.rule_predict on one client's recurring streams: (label, why)."""
    live = streams[streams["is_live"]].sort_values("days_to_next")
    if live.empty:
        return "none", "no live subscription stream"
    if live["n_occurrences"].max() in TRIAL_LENGTHS:
        return "none", "the longest live stream has only 3-4 charges (looks like a trial that ends)"
    return live["category"].iloc[0], "the live stream whose next charge is due first"


def summarize_client(streams: pd.DataFrame, family_charges: pd.DataFrame, cutoff: pd.Timestamp) -> str:
    """Our detector's view of one client, so the LLM doesn't have to cluster ~75 rows itself."""
    lines = ["Recurring subscription streams found by our detector (monthly cadence, similar amount):"]
    for s in streams.sort_values("days_to_next").itertuples(index=False):
        next_date = (s.last_date + pd.Timedelta(days=s.median_gap_days)).date()
        if not s.is_live:
            status = f"STOPPED (next charge was due {next_date}, {-s.days_to_next:.0f} days before the cutoff)"
        elif s.n_occurrences in TRIAL_LENGTHS:
            status = f"LIVE but trial length ({s.n_occurrences} charges), next charge due {next_date}"
        else:
            status = f"LIVE, next charge due {next_date}"
        refunds = f", {s.n_refunds} refund(s)" if s.n_refunds else ""
        lines.append(f"  - {s.category}: {s.n_occurrences} charges of ~{s.mean_amount:.2f}, "
                     f"{s.first_date.date()} .. {s.last_date.date()}, every ~{s.median_gap_days:.0f} days"
                     f"{refunds} -> {status}")
    if streams.empty:
        lines.append("  - none")

    lines.append(f"Family charges NOT part of any detected stream, last one within "
                 f"{NEW_CHARGE_WINDOW_DAYS} days of the cutoff (possible new subscriptions):")
    n_new = 0
    for family, g in family_charges[~family_charges["category"].isin(set(streams["category"]))].groupby("category"):
        last = g["timestamp"].max()
        days_before = (cutoff - last).days
        if days_before > NEW_CHARGE_WINDOW_DAYS:
            continue
        n_new += 1
        lines.append(f"  - {family}: {len(g)} charge(s) in total, last on {last.date()} "
                     f"({days_before} days before the cutoff, {g['amount'].iloc[-1]:.2f}), "
                     f"monthly next charge would be ~{(last + pd.Timedelta(days=30)).date()}")
    if not n_new:
        lines.append("  - none")

    label, why = rule_label(streams)
    lines.append(f'Rule-based suggestion: "{label}" ({why}).')
    return "\n".join(lines)


def format_history(client_id: str, txns: pd.DataFrame) -> str:
    """Compact one-line-per-transaction rendering (date only, fee only if > 0).

    Rows that category_map.classify() sees as subscription-like get a hint:
    "[gym]" (family-specific phrase), "[gym via mcc]" (generic phrase on a
    clean mcc), "[media?]" (music or streaming), "[generic]" (no family).
    """
    first, last = txns["timestamp"].min().date(), txns["timestamp"].max().date()
    lines = [f"client {client_id}: {len(txns)} transactions, {first} .. {last}",
             "date | dir | type | amount currency | mcc | description"]
    for r in txns.itertuples(index=False):
        fee = f" (fee {r.fee:.2f})" if r.fee else ""
        lines.append(f"{r.timestamp.date()} | {r.direction} | {r.type} | "
                     f"{r.amount:.2f} {r.currency}{fee} | {r.mcc} | {r.description}{_hint(r.mcc, r.description)}")
    return "\n".join(lines)


def _hint(mcc: str, description: str) -> str:
    category, strength = classify(mcc, description)
    if strength is None:
        return ""
    if category == MEDIA:
        return " [media?]"
    if category is None:
        return " [generic]"
    return f" [{category} via mcc]" if strength == "mcc" else f" [{category}]"


def select_examples(labels: pd.DataFrame, n: int, seed: int) -> list[tuple[str, str]]:
    """One client per label, the remaining slots go to the most frequent labels."""
    rng = random.Random(seed)
    by_label = {lab: sorted(labels.loc[labels[TARGET_COL] == lab, "client_id"]) for lab in LABELS}
    for ids in by_label.values():
        rng.shuffle(ids)
    order = LABELS + sorted(LABELS, key=lambda lab: -len(by_label[lab])) * n
    picked = []
    for lab in order:
        if len(picked) == n:
            break
        if by_label[lab]:
            picked.append((by_label[lab].pop(), lab))
    rng.shuffle(picked)  # don't let the example order hint at a label
    return picked


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

def _phrase_tables() -> str:
    by_family: dict[str, list[str]] = {}
    for phrase, family in CANONICAL_PHRASES.items():
        by_family.setdefault(family, []).append(f'"{phrase}"')
    rows = [f"  - {family}: {', '.join(phrases)}" for family, phrases in sorted(by_family.items())]
    mccs = [f"{mcc}={'music or streaming' if cat == MEDIA else cat}" for mcc, cat in CLEAN_MCC_CATEGORY.items()]
    return "\n".join(rows) + (
        f"\n  Generic phrases with no family of their own: {', '.join(sorted(GENERIC_PHRASES))}."
        f"\n  mcc of a subscription (usually right, ~5% wrong): {', '.join(mccs)}."
        f"\n  Noise: prefixes {sorted(NOISE_PREFIXES)}, suffixes {sorted(NOISE_SUFFIXES)}, "
        f"abbreviations {ABBREVIATIONS}.")


def build_system_prompt(examples: list[tuple[str, str]], clients: dict[str, str]) -> str:
    example_blocks = "\n\n".join(
        f"### Example {i}\n{clients[cid]}\n"
        f'Answer label: "{label}"'
        for i, (cid, label) in enumerate(examples, 1)
    )
    return f"""You are an expert analyst of retail-banking transaction data. You label
synthetic client histories for a machine-learning challenge (UBS, Swiss AI Weeks 2026).

## Task
Each client's history ends at the cutoff date {CUTOFF}. Predict the client's NEXT
RECURRING MERCHANT FAMILY: the family of the first recurring (subscription-like)
payment the client makes in the {HORIZON_DAYS} days after the cutoff. Answer with
exactly one of:
  {", ".join(LABELS)}
Use "none" when no recurring merchant family is expected to recur within
{HORIZON_DAYS} days (no active subscription, or all subscriptions lapsed/cancelled).

Label frequencies in the labelled data: none ~29%, every other family 9-12% each.
The metric is macro-F1, so every family matters as much as "none"; do not default
to "none" when there is real evidence of an active subscription.

## Transaction fields
date, direction (in = money received, out = money spent), type (card_payment,
topup, p2p_transfer, atm, fee, refund, ...), amount + currency (currencies are
mixed per client; the amount is what matters for matching charges), mcc
(merchant category code), description (free text).

## How recurring subscriptions look in this data
- A subscription is a series of outgoing charges at a near-constant amount
  (within ~3%), usually monthly (gaps of roughly 20-45 days).
- Descriptions are NOISY on purpose. Each family has a few canonical phrases,
  which get random prefixes/suffixes, abbreviations and truncations ("billing
  cover plan core", "prem plan", "safe", "urban"). The same subscription often
  rotates between its canonical phrases and GENERIC phrases ("digital plus",
  "premium plan", ...) from month to month, and its mcc can change too. Group
  charges by amount and cadence first, then decide the family by majority over
  the members that carry a family-specific phrase.
- Canonical phrases per family:
{_phrase_tables()}
- Music and streaming share mcc 5812 and the generic phrases; decide between
  them from the family-specific phrases in the same amount stream.
- Rows are pre-annotated by our rule-based matcher: "[family]" = family-specific
  phrase, "[family via mcc]" = generic phrase on a reliable mcc, "[media?]" =
  music or streaming, "[generic]" = subscription-like but no family. Rows with no
  annotation are ordinary spending or income (groceries, dining, ATM, p2p,
  salary, one-off purchases: {", ".join(sorted(NON_SUBSCRIPTION_PHRASES))}).
  The annotations are per-row hints and can be wrong; trust the stream as a whole.
- Refunds (type refund, same amount as a charge, 1-3 days later) reverse a
  charge; a subscription whose last event is a refund or that stopped charging
  months before the cutoff has probably been cancelled.

## What you get per client
1. A SUMMARY from our own subscription detector: every recurring stream it found
   (family, number of charges, amount, dates, cadence, when the next charge is
   due, LIVE / STOPPED / trial length), the family charges that are NOT part of
   any stream but happened in the last {NEW_CHARGE_WINDOW_DAYS} days (possible new subscriptions),
   and the answer of our best hand-written rule.
2. The full raw transaction history. Use it to check the summary: the detector
   misses streams whose amount changed, merges or splits streams, and cannot
   see a subscription that has only one charge so far.

## How to decide (measured on 2,000 labelled training clients)
The rule-based suggestion alone scores macro-F1 0.50, and it is hard to beat:
in a test, answers that overrode it were wrong twice as often as the rule.
KEEP THE SUGGESTION unless one of these specific cases applies:
  a) Suggestion is "none", but a family has exactly ONE charge 15-30 days
     before the cutoff -> answer that family.
  b) Suggestion is "none", but the raw history shows a clear monthly stream
     the detector missed (3+ charges at a similar amount, one of them in
     December 2025, next charge due in early January) -> answer its family.
  c) Suggestion is a family, but a single new family charge 15-30 days before
     the cutoff is due BEFORE the suggested stream's next charge -> answer the
     new family.
  d) The detector clearly got the family wrong (e.g. most charges of the
     stream say another family) -> answer the correct family.
Never override "none" because of an older, scattered or STOPPED charge, and
never change a family answer to "none" on a hunch. The facts behind these
cases:
- The answer is usually the family whose NEXT charge falls first after
  {CUTOFF}, counting both live streams and newly started subscriptions.
- Rule says a family (there is a live stream with 2 or 5+ charges): the answer
  is "none" 20% of the time, the soonest-due live stream 60%, the second-soonest
  16%, and a family with no live stream (usually a new subscription) the rest.
  A live stream with only 2 charges is "none" 35% of the time; with 7+ charges
  only 15%.
- No live stream at all: "none" 56% of the time. The other 44% are mostly NEW
  subscriptions:
    - a family with exactly ONE charge 15-30 days before the cutoff (so its
      next monthly charge is due in the first days of January) is the answer
      63% of the time; with one charge 0-15 or 30-60 days before the cutoff,
      about 30%;
    - a family whose only charges are older than 60 days is almost never the
      answer (1-5%): that subscription is gone;
    - 2-3 scattered family charges are rarely the answer (under 15%).
- Longest live stream has only 3-4 charges ("trial length"): "none" about 66%
  of the time.
- A NEW single recent charge (15-30 days before the cutoff) competes with live
  streams too: if its next charge (~30 days later) comes before every live
  stream's next charge, it is a strong candidate (about 50%).
- STOPPED streams (overdue by more than 5 days) almost never come back.
- Across all clients, about 29% of answers are "none" and 9-12% are each
  family. Do not overuse any one family; gym and insurance are no more common
  than the others.

## Output
Reply with a single JSON object and nothing else:
{{"reasoning": "<at most 3 sentences: which candidates, which one is due first, why>",
  "label": "<one of: {", ".join(LABELS)}>",
  "confidence": <probability from 0 to 1 that your label is correct; about 0.5 is
                 typical for this task, use under 0.4 when two answers are close>}}

## Labelled examples (real answers from the training set)

{example_blocks}
"""


def build_messages(system_prompt: str, client_text: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Label this client. Reply with the JSON object only.\n\n" + client_text},
    ]


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

class FatalAPIError(RuntimeError):
    """Out of credit, bad key, unknown model, ...: retrying won't help."""


def _error_code(detail: str) -> str | None:
    try:
        return json.loads(detail).get("error", {}).get("code")
    except (ValueError, AttributeError):
        return None


def call_openai(messages: list[dict], model: str, api_key: str, temperature: float | None) -> dict:
    body = {"model": model, "messages": messages, "response_format": {"type": "json_object"}}
    if temperature is not None:  # some models only accept their default temperature
        body["temperature"] = temperature
    data = json.dumps(body).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(API_URL, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:2000]
            code = _error_code(detail)
            if code in FATAL_ERROR_CODES or e.code in (401, 403):
                raise FatalAPIError(f"HTTP {e.code} {code}: {detail[:500]}") from e
            if e.code not in (408, 409, 429, 500, 502, 503, 504) or attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"HTTP {e.code}: {detail}") from e
            wait = max(float(e.headers.get("Retry-After") or 0), 2 ** attempt)
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError) as e:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"network error: {e}") from e
            wait = 2 ** attempt
        time.sleep(min(wait, 60))
    raise RuntimeError("unreachable")


def label_client(client_id: str, messages: list[dict], args, api_key: str,
                 stop: threading.Event) -> dict | None:
    if stop.is_set():  # the run is shutting down; leave this client for the next run
        return None
    resp = call_openai(messages, args.model, api_key, args.temperature)
    content = resp["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    label = str(parsed.get("label", "")).strip().lower()
    if label not in LABELS:
        raise ValueError(f"invalid label {label!r}")
    usage = resp.get("usage", {})
    return {
        "client_id": client_id,
        "cutoff_date": CUTOFF,
        TARGET_COL: label,
        "confidence": parsed.get("confidence"),
        "reasoning": parsed.get("reasoning"),
        "model": resp.get("model", args.model),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _done_ids(path: str, model: str | None = None) -> set[str]:
    """Clients already in path; with model, only those labelled by it.

    The API reports dated names ("gpt-4.1-mini-2025-04-14"), so the model
    matches by prefix. Old records of other models stay in the file, and
    export_labels_csv keeps the newest record per client.
    """
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    return {r["client_id"] for r in recs if model is None or r.get("model", "").startswith(model)}


def run(client_ids: list[str], clients: dict[str, str], system_prompt: str, out_path: str, args, api_key: str,
        checkpoint=None) -> bool:
    """Label client_ids, appending to out_path. Returns False if the run was cut short."""
    done = _done_ids(out_path, args.model if args.relabel_other_models else None)
    todo = [cid for cid in client_ids if cid not in done]
    print(f"{len(done)} already labelled in {out_path}, {len(todo)} to go", file=sys.stderr)
    n_ok = n_err = 0
    stop = threading.Event()
    completed = True
    pool = ThreadPoolExecutor(args.workers)
    try:
        with open(out_path, "a", encoding="utf-8") as out:

            def save_checkpoint() -> None:
                out.flush()
                os.fsync(out.fileno())
                if checkpoint:
                    checkpoint()
                print(f"  checkpoint: {len(done) + n_ok} labelled in total, saved to {out_path}", file=sys.stderr)

            futures = {pool.submit(label_client, cid, build_messages(system_prompt, clients[cid]),
                                   args, api_key, stop): cid for cid in todo}
            try:
                for fut in as_completed(futures):
                    cid = futures[fut]
                    if fut.cancelled():
                        continue
                    try:
                        rec = fut.result()
                    except FatalAPIError as e:
                        # don't break: keep draining so results that already came
                        # back are still written; queued requests are cancelled
                        if completed:
                            print(f"[fatal] {cid}: {e}\n  stopping, saving finished results ...", file=sys.stderr)
                            completed = False
                            stop.set()
                            for f in futures:
                                f.cancel()
                        continue
                    except Exception as e:  # logged and retried on the next (resumed) run
                        n_err += 1
                        print(f"[error] {cid}: {e}", file=sys.stderr)
                        continue
                    if rec is None:
                        continue
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    n_ok += 1
                    if n_ok % 50 == 0:
                        print(f"  {n_ok}/{len(todo)} labelled, {n_err} errors", file=sys.stderr)
                    if n_ok % CHECKPOINT_EVERY == 0:
                        save_checkpoint()
            except KeyboardInterrupt:
                print("\n[interrupted]", file=sys.stderr)
                completed = False
            finally:
                # stop queued requests; the few already in flight return None or finish
                stop.set()
                pool.shutdown(wait=False, cancel_futures=True)
                save_checkpoint()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    left = len(todo) - n_ok
    print(f"{'done' if completed else 'stopped early'}: {n_ok} labelled this run, {n_err} errors, "
          f"{left} still to label" + (" -> re-run the same command to resume" if left else ""), file=sys.stderr)
    return completed


def export_labels_csv(jsonl_path: str, csv_path: str) -> None:
    """Write the pseudo-labels in train_labels.csv schema (+ llm_confidence)."""
    if not os.path.exists(jsonl_path) or os.path.getsize(jsonl_path) == 0:
        return
    recs = pd.read_json(jsonl_path, lines=True, dtype={"client_id": str})
    recs = recs.drop_duplicates("client_id", keep="last").sort_values("client_id")
    recs = recs.rename(columns={"confidence": "llm_confidence"})
    recs[["client_id", "cutoff_date", TARGET_COL, "llm_confidence"]].to_csv(csv_path, index=False)
    print(f"wrote {len(recs)} labels to {csv_path}\n"
          f"{recs[TARGET_COL].value_counts(normalize=True).round(3).to_string()}", file=sys.stderr)


def report_eval(jsonl_path: str, client_ids: list[str], labels: pd.DataFrame, split: str) -> None:
    from sklearn.metrics import classification_report, f1_score

    recs = pd.read_json(jsonl_path, lines=True, dtype={"client_id": str})
    recs = recs[recs["client_id"].isin(client_ids)].drop_duplicates("client_id", keep="last")
    merged = recs.merge(labels, on="client_id", suffixes=("_llm", ""))
    y_true, y_pred = merged[TARGET_COL], merged[f"{TARGET_COL}_llm"]
    print(f"LLM on {len(merged)} held-out {split} clients: "
          f"macro-F1 {f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0):.4f}")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-key", default=None, help="defaults to the OPENAI_API_KEY env var")
    p.add_argument("--n-examples", type=int, default=N_EXAMPLES)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="pass a negative value to omit it (for models that reject it)")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--limit", type=int, default=None, help="label only the first N unlabeled clients")
    evals = p.add_mutually_exclusive_group()
    evals.add_argument("--eval-train", type=int, default=0, metavar="N",
                       help="instead of labelling, score the LLM on N train clients that are not "
                            "few-shot examples (use this to tune the prompt)")
    evals.add_argument("--eval-valid", type=int, default=0, metavar="N",
                       help="instead of labelling, score the LLM on N valid clients (final check only)")
    p.add_argument("--dry-run", action="store_true", help="print one prompt and a token estimate, no API calls")
    p.add_argument("--output", default=OUTPUT_PATH)
    p.add_argument("--labels-csv", default=LABELS_CSV_PATH)
    p.add_argument("--eval-output", default=None,
                   help=f"results file for --eval-train/--eval-valid (default {EVAL_TRAIN_OUTPUT_PATH} / "
                        f"{EVAL_VALID_OUTPUT_PATH}); use a new one per model/prompt")
    p.add_argument("--relabel-other-models", action="store_true",
                   help="also redo clients labelled by a different --model (old records are kept)")
    args = p.parse_args()
    if args.temperature is not None and args.temperature < 0:
        args.temperature = None

    # Few-shot examples come from train only; valid is read below only for --eval-valid.
    train_labels = pd.read_csv(TRAIN_LABELS_PATH, dtype={"client_id": str})[["client_id", TARGET_COL]]
    train_clients = load_clients(TRAIN_TXN_PATH)
    examples = select_examples(train_labels, args.n_examples, SEED)
    system_prompt = build_system_prompt(examples, train_clients)
    example_ids = {cid for cid, _ in examples}
    print(f"few-shot examples (train): {[lab for _, lab in examples]}", file=sys.stderr)

    eval_labels = eval_split = None
    if args.eval_train:
        eval_labels, eval_split = train_labels, "train"
        pool = sorted(set(train_labels["client_id"]) - example_ids)
        client_ids = sorted(random.Random(SEED).sample(pool, min(args.eval_train, len(pool))))
        clients, out_path = train_clients, args.eval_output or EVAL_TRAIN_OUTPUT_PATH
    elif args.eval_valid:
        eval_labels = pd.read_csv(VALID_LABELS_PATH, dtype={"client_id": str})[["client_id", TARGET_COL]]
        eval_split = "valid"
        pool = sorted(eval_labels["client_id"])
        client_ids = sorted(random.Random(SEED).sample(pool, min(args.eval_valid, len(pool))))
        clients, out_path = load_clients(VALID_TXN_PATH), args.eval_output or EVAL_VALID_OUTPUT_PATH
    else:
        print(f"loading {UNLABELED_PATH} ...", file=sys.stderr)
        clients = load_clients(UNLABELED_PATH)
        client_ids = sorted(clients)[: args.limit]
        out_path = args.output

    if args.dry_run:
        user_chars = [len(build_messages(system_prompt, clients[c])[1]["content"]) for c in client_ids]
        sys_tok = len(system_prompt) / CHARS_PER_TOKEN
        user_tok = sum(user_chars) / CHARS_PER_TOKEN
        print(build_messages(system_prompt, clients[client_ids[0]])[1]["content"][:3000])
        print("\n" + "=" * 80 + "\n" + system_prompt[:6000] + "\n...")
        print(f"\n~{sys_tok:,.0f} prompt-prefix tokens (cacheable) + ~{user_tok / len(client_ids):,.0f} "
              f"per client; {len(client_ids)} clients -> ~{(sys_tok * len(client_ids) + user_tok) / 1e6:.1f}M "
              f"input tokens in total", file=sys.stderr)
        return

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("No API key: set OPENAI_API_KEY (or pass --api-key).")

    if eval_split:
        run(client_ids, clients, system_prompt, out_path, args, api_key)
        report_eval(out_path, client_ids, eval_labels, eval_split)
    else:
        completed = run(client_ids, clients, system_prompt, out_path, args, api_key,
                        checkpoint=lambda: export_labels_csv(out_path, args.labels_csv))
        if not completed:
            sys.exit(1)


if __name__ == "__main__":
    main()
