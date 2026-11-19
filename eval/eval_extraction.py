"""Score activity type and head count extraction for each method, and the combination the message table uses.

Sets
  dev     data/extraction_dev.jsonl (made by tools/make_dev_sets.py): 72 Event messages with their activity category and
          head count. The draft ACTIVITY_CATEGORIES were written with these activities in view, so category scores
          here are optimistic.
  hidden  the Events of the TA's 10 hidden-test messages, scored against the TA's own activity names (the rules
          were written after seeing it)

Methods
  rules      activity: extract.find_topic;  count: extract.find_participants            (on the original text)
  qa         activity: the QA model's event_type, cleaned;  count: its participants     (on the redacted text)
  zero-shot  activity: the zero-shot model's ACTIVITY_CATEGORIES pick, whatever its score
  combined   what the message table does (extract.activity_type / extract.participant_count)

Activity is right when it is the true category (dev) or shares at least half its words with the true activity
(eval_hidden_test.same_activity). Count is right when it equals the true count, or is blank when the message gives none.

Run:  python eval/eval_extraction.py [--show_errors]
"""
import argparse
import json
import os
import sys

import pandas as pd
from datasets import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))  # pipeline modules
from eval_hidden_test import same_activity
from extract import (ACTIVITY_MIN_SCORE, OTHER_ACTIVITY, activity_type, classify_activities, clean_qa_label,
                     extract_dataset_info, find_participants, find_topic, participant_count)
from preprocess import preprocess_dataset
from redact import process_dataset_pii

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load_set(name: str) -> pd.DataFrame:
    """Columns: text, activity (the true category or activity name), participants (int or None, kept as objects:
    a float column would turn None into NaN, and NaN != None)."""
    if name == "dev":
        rows = [json.loads(line) for line in open(os.path.join(ROOT, "data", "extraction_dev.jsonl"), encoding="utf-8")]
        return pd.DataFrame({"text": [r["text"] for r in rows], "activity": [r["category"] for r in rows],
                             "participants": pd.Series([r["participants"] for r in rows], dtype=object)})
    if name == "hidden":
        msgs = pd.read_excel(os.path.join(ROOT, "data", "hidden_test_transcripts.xlsx"))
        key = pd.read_excel(os.path.join(ROOT, "data", "hidden_test_outputs.xlsx"))
        df = msgs.merge(key, on="message_id", suffixes=("", "_key"))
        df = df[df["relevance"] == "Event"]
        return pd.DataFrame({"text": list(df["transcript"]), "activity": list(df["activity_type"]),
                             "participants": pd.Series([None if pd.isna(p) else int(p) for p in df["participants"]],
                                                       dtype=object)})
    raise ValueError(f"unknown set {name!r}")


def predictions(texts: list[str]) -> tuple[dict[str, tuple[list, list]], list[float]]:
    """(activity, count) lists per method, and the zero-shot scores. The models see the cleaned, redacted text, as
    in the pipeline; every message is treated as an Event."""
    ds = process_dataset_pii(preprocess_dataset(Dataset.from_dict({"text": texts})))
    ds = classify_activities(ds.add_column("relevance", ["Event"] * len(texts)))
    rows = list(extract_dataset_info(ds))
    out = {method: ([], []) for method in ("rules", "qa", "zero-shot", "combined")}
    for text, row in zip(texts, rows):
        qa_count = str(row.get("participants") or "").strip()
        out["rules"][0].append(find_topic(text))
        out["rules"][1].append(find_participants(text))
        out["qa"][0].append(clean_qa_label(row["event_type"]))
        out["qa"][1].append(int(qa_count) if qa_count.isdigit() else None)
        out["zero-shot"][0].append(row["activity_category"])
        out["zero-shot"][1].append(None)
        out["combined"][0].append(activity_type(text, row["event_type"], row["activity_category"], row["activity_score"]))
        out["combined"][1].append(participant_count(text, row["participants"]))
    return out, [row["activity_score"] for row in rows]


def activity_right(pred: str, gold: str) -> bool:
    return pred == gold or same_activity(pred, gold)


def score(df: pd.DataFrame, activities: list[str], counts: list) -> str:
    act_found = sum(bool(a) and a != OTHER_ACTIVITY for a in activities)
    act_right = sum(activity_right(p, g) for p, g in zip(activities, df["activity"]))
    gold = list(df["participants"])
    right = sum(p == g for p, g in zip(counts, gold))
    wrong_number = sum(p is not None and p != g for p, g in zip(counts, gold))
    missed = sum(p is None and g is not None for p, g in zip(counts, gold))
    n = len(df)
    return (f"activity right {act_right}/{n} (labeled {act_found})   "
            f"count right {right}/{n} (wrong number {wrong_number}, missed {missed})")


def main():
    p = argparse.ArgumentParser(description="Score activity type and head count extraction")
    p.add_argument("--sets", nargs="+", default=["dev", "hidden"], choices=["dev", "hidden"])
    p.add_argument("--show_errors", action="store_true", help="List the combined method's mistakes")
    args = p.parse_args()

    for name in args.sets:
        df = load_set(name)
        preds, zs_scores = predictions(list(df["text"]))
        print(f"\n== {name}: {len(df)} Event messages ({df['participants'].notna().sum()} with a head count)")
        for method, (activities, counts) in preds.items():
            print(f"{method:<10} {score(df, activities, counts)}" + ("   (no counts)" if method == "zero-shot" else ""))
        right = [activity_right(a, g) for a, g in zip(preds["zero-shot"][0], df["activity"])]
        for t in (0.1, 0.2, 0.3, 0.4, 0.5):
            kept = [r for r, s in zip(right, zs_scores) if s >= t]
            print(f"   zero-shot score >= {t}: {len(kept)} kept, {sum(kept)} right, {len(kept) - sum(kept)} wrong"
                  + ("   <- ACTIVITY_MIN_SCORE" if t == ACTIVITY_MIN_SCORE else ""))
        if args.show_errors:
            for (_, row), act, cnt in zip(df.iterrows(), *preds["combined"]):
                if not activity_right(act, row["activity"]) or cnt != row["participants"]:
                    print(f"   true ({row['activity']}, {row['participants']}) got ({act}, {cnt}): {row['text'][:110]}")


if __name__ == "__main__":
    main()
