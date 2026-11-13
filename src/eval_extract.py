"""Score src/extract.py output against the TA's answer key.

Run:  python src/eval_extract.py -p results/extract_out.csv -k hidden_test_outputs.csv
"""
import argparse
import re

import pandas as pd


def tokens(text):
    return set(re.findall(r"[a-z]+", str(text).lower()))


def same_activity(a, b):
    a, b = tokens(a), tokens(b)
    return bool(a and b) and len(a & b) / len(a | b) >= 0.5


def main():
    parser = argparse.ArgumentParser(description="Compare extraction output with the TA key")
    parser.add_argument("-p", "--pred", required=True)
    parser.add_argument("-k", "--key", required=True)
    args = parser.parse_args()

    pred = pd.read_csv(args.pred, dtype=str, keep_default_na=False).set_index("message_id")
    key = pd.read_csv(args.key, dtype=str, keep_default_na=False).set_index("message_id")
    ids = [i for i in key.index if i in pred.index]

    def report(name, hits, rows):
        print(f"{name}: {sum(hits)}/{len(hits)}")
        for ok, row in zip(hits, rows):
            if not ok:
                print("   miss:", row)

    rel = [pred.loc[i, "relevance"] == key.loc[i, "relevance"] for i in ids]
    report("relevance", rel, [f"msg {i}: got {pred.loc[i, 'relevance']}, key {key.loc[i, 'relevance']}" for i in ids])

    ev = [i for i in ids if key.loc[i, "relevance"] == "Event"]
    num = lambda s: str(int(float(s))) if s not in ("", "none") else ""
    part = [num(pred.loc[i, "participants"]) == num(key.loc[i, "participants"]) for i in ev]
    report("participants (events)", part, [f"msg {i}: got {pred.loc[i, 'participants']!r}, key {key.loc[i, 'participants']!r}" for i in ev])

    act = [same_activity(pred.loc[i, "activity_type"], key.loc[i, "activity_type"]) for i in ev]
    report("activity_type (events, word overlap >= 0.5)", act, [f"msg {i}: got {pred.loc[i, 'activity_type']!r}, key {key.loc[i, 'activity_type']!r}" for i in ev])

    dup = [pred.loc[i, "duplicate_of"] == key.loc[i, "duplicate_of"] for i in ids]
    report("duplicate_of", dup, [f"msg {i}: got {pred.loc[i, 'duplicate_of']}, key {key.loc[i, 'duplicate_of']}" for i in ids])

    exact = [pred.loc[i, "anonymized_text"] == key.loc[i, "expected_anonymized_text"] for i in ids]
    report("anonymized_text exact match", exact, [f"msg {i}" for i in ids])


if __name__ == "__main__":
    main()
