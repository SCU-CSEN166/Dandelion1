"""Score relevance (Event / Chit-chat): the zero-shot classifier and, for comparison, the in-sample rules.

Sets
  dev     data/relevance_dev.jsonl (made by tools/make_dev_sets.py): the set to choose the threshold on
  hidden  the TA's 10 hidden-test messages with their key (the rules were written after seeing it)
  farmer  data/farmerchat_kenya_sample.jsonl: 50 farmers' questions, all Chit-chat (none reports an activity)

Like the pipeline, the classifier sees the cleaned and redacted text; the rules see the original text.
On the dev set the script also lists Event precision / recall / F1 for a range of thresholds.

Run:  python eval/eval_relevance.py                 (all three sets, current threshold)
      python eval/eval_relevance.py --sets dev --show_errors
"""
import argparse
import json
import os
import sys
import re

import pandas as pd
from datasets import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))  # pipeline modules
from extract import find_participants, find_topic
from preprocess import preprocess_dataset
from redact import process_dataset_pii
from relevance import EVENT, EVENT_THRESHOLD, classify_dataset_relevance

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# The rule-based relevance that the classifier replaced: kept here only as a baseline
_ACTION = re.compile(r"\b(?:we held|we did|we went|we conducted|we organi[sz]ed|we carried out|we ran|"
                     r"we visited|we trained|held a|conducted a)\b", re.I)


def is_event(text: str) -> bool:
    """Event if the text has an attendance count, or a known topic plus an action like 'we held'."""
    return find_participants(text) is not None or bool(find_topic(text) and _ACTION.search(text or ""))


def load_set(name: str) -> tuple[list[str], list[str]]:
    """(texts, gold labels) of a named set."""
    if name == "dev":
        rows = [json.loads(line) for line in open(os.path.join(ROOT, "data", "relevance_dev.jsonl"), encoding="utf-8")]
        return [r["text"] for r in rows], [r["label"] for r in rows]
    if name == "hidden":
        msgs = pd.read_excel(os.path.join(ROOT, "data", "hidden_test_transcripts.xlsx"))
        key = pd.read_excel(os.path.join(ROOT, "data", "hidden_test_outputs.xlsx"))
        df = msgs.merge(key[["message_id", "relevance"]], on="message_id")
        return list(df["transcript"]), list(df["relevance"])
    if name == "farmer":
        rows = [json.loads(line) for line in open(os.path.join(ROOT, "data", "farmerchat_kenya_sample.jsonl"), encoding="utf-8")]
        return [r["query"] for r in rows], ["Chit-chat"] * len(rows)
    raise ValueError(f"unknown set {name!r}")


def event_scores(texts: list[str]) -> list[float]:
    """The classifier's Event score for each text, after the pipeline's cleaning and redaction."""
    ds = process_dataset_pii(preprocess_dataset(Dataset.from_dict({"text": texts})))
    return classify_dataset_relevance(ds, score_col="score")["score"]


def metrics(pred_event: list[bool], gold: list[str]) -> dict:
    gold_event = [g == EVENT for g in gold]
    tp = sum(p and g for p, g in zip(pred_event, gold_event))
    fp = sum(p and not g for p, g in zip(pred_event, gold_event))
    fn = sum(g and not p for p, g in zip(pred_event, gold_event))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall) if precision and recall
          else 0.0 if precision is not None and recall is not None else None)
    accuracy = sum(p == g for p, g in zip(pred_event, gold_event)) / len(gold)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "false_events": fp,
            "missed_events": fn}


def fmt(m: dict) -> str:
    f = lambda v: "  -  " if v is None else f"{v:.2f}"  # noqa: E731
    return (f"accuracy {f(m['accuracy'])}  precision {f(m['precision'])}  recall {f(m['recall'])}  F1 {f(m['f1'])}"
            f"  (false Events {m['false_events']}, missed Events {m['missed_events']})")


def main():
    p = argparse.ArgumentParser(description="Score Event / Chit-chat relevance")
    p.add_argument("--sets", nargs="+", default=["dev", "hidden", "farmer"], choices=["dev", "hidden", "farmer"])
    p.add_argument("--threshold", type=float, default=EVENT_THRESHOLD)
    p.add_argument("--show_errors", action="store_true", help="List the messages the classifier gets wrong")
    args = p.parse_args()

    for name in args.sets:
        texts, gold = load_set(name)
        scores = event_scores(texts)
        n_events = sum(g == EVENT for g in gold)
        print(f"\n== {name}: {len(texts)} messages ({n_events} Event, {len(texts) - n_events} Chit-chat)")
        print(f"classifier @ {args.threshold:.3f}: {fmt(metrics([s >= args.threshold for s in scores], gold))}")
        print(f"rules (in-sample):  {fmt(metrics([is_event(t) for t in texts], gold))}")
        if name == "dev":
            print("threshold sweep (classifier):")
            for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995):
                print(f"   {t:<6} {fmt(metrics([s >= t for s in scores], gold))}")
        if args.show_errors:
            for text, g, s in zip(texts, gold, scores):
                if (s >= args.threshold) != (g == EVENT):
                    print(f"   wrong ({g}, score {s:.3f}): {text[:150]}")


if __name__ == "__main__":
    main()
