"""Compare a TA-format table (from main.py --ta_table_out) with the TA's expected outputs.

Run:  python src/eval_ta_table.py -p results/hidden_test_ta_table.csv -k results/hidden_test_outputs.xlsx

Scores: relevance, activity type (word overlap of 0.5 or more), participants, duplicate_of, PII recall (each item in
the key's pii_items must be gone from our anonymized text), pii_count, and exact match of the anonymized text.
"""
import argparse
import re

import pandas as pd


def read(path):
    df = pd.read_excel(path, dtype=str) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path, dtype=str)
    df = df.fillna("")
    df["message_id"] = df["message_id"].astype(float).astype(int)
    return df.set_index("message_id")


def key_items(cell):
    """'Lucy Kiprop (also 'Lucy'); Tumaini (women's group name)' -> ['Lucy Kiprop', 'Lucy', 'Tumaini']"""
    out = []
    for part in str(cell).split(";"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(.*?)\s*\((.*)\)$", part)
        if m:
            out.append(m.group(1).strip())
            extra = re.search(r"'([^']+)'", m.group(2))
            if "also" in m.group(2) and extra:
                out.append(extra.group(1))
        else:
            out.append(part)
    return out


def item_status(item, text):
    if item in text:
        return "missed"
    pieces = [p for p in re.split(r"\W+", item) if len(p) >= 3]
    return "partial" if any(p in text for p in pieces) else "caught"


def words(s):
    return set(re.findall(r"[a-z]+", str(s).lower())) - {"and", "of", "on", "the", "s"}


def same_activity(a, b):
    a, b = words(a), words(b)
    return bool(a and b) and len(a & b) / len(a | b) >= 0.5


def num(s):
    s = str(s).strip()
    return str(int(float(s))) if s not in ("", "none", "nan") else ""


def main():
    parser = argparse.ArgumentParser(description="Score a TA-format table against the TA key")
    parser.add_argument("-p", "--pred", required=True)
    parser.add_argument("-k", "--key", required=True)
    args = parser.parse_args()
    pred, key = read(args.pred), read(args.key)
    ids = [i for i in key.index if i in pred.index]
    events = [i for i in ids if key.loc[i, "relevance"] == "Event"]

    def show(name, oks, notes):
        print(f"{name}: {sum(oks)}/{len(oks)}")
        for ok, note in zip(oks, notes):
            if not ok:
                print("   miss:", note)

    show("relevance", [pred.loc[i, "relevance"] == key.loc[i, "relevance"] for i in ids],
         [f"msg {i}: ours {pred.loc[i, 'relevance']!r}, key {key.loc[i, 'relevance']!r}" for i in ids])
    show("activity_type (events)", [same_activity(pred.loc[i, "activity_type"], key.loc[i, "activity_type"]) for i in events],
         [f"msg {i}: ours {pred.loc[i, 'activity_type']!r}, key {key.loc[i, 'activity_type']!r}" for i in events])
    show("participants", [num(pred.loc[i, "participants"]) == num(key.loc[i, "participants"]) for i in ids],
         [f"msg {i}: ours {pred.loc[i, 'participants']!r}, key {key.loc[i, 'participants']!r}" for i in ids])
    show("duplicate_of", [pred.loc[i, "duplicate_of"] == key.loc[i, "duplicate_of"] for i in ids],
         [f"msg {i}: ours {pred.loc[i, 'duplicate_of']}, key {key.loc[i, 'duplicate_of']}" for i in ids])
    show("pii_count", [num(pred.loc[i, "pii_count"]) == num(key.loc[i, "pii_count"]) for i in ids],
         [f"msg {i}: ours {pred.loc[i, 'pii_count']}, key {key.loc[i, 'pii_count']}" for i in ids])

    counts = {"caught": 0, "partial": 0, "missed": 0}
    misses = []
    for i in ids:
        for item in key_items(key.loc[i, "pii_items"]):
            st = item_status(item, pred.loc[i, "expected_anonymized_text"])
            counts[st] += 1
            if st != "caught":
                misses.append(f"msg {i}: {st} {item!r}")
    total = sum(counts.values())
    print(f"PII recall: {counts['caught']}/{total} = {counts['caught'] / total:.3f} "
          f"(partial {counts['partial']}, missed {counts['missed']})")
    for m in misses:
        print("   ", m)
    show("anonymized text exact match", [pred.loc[i, "expected_anonymized_text"] == key.loc[i, "expected_anonymized_text"] for i in ids],
         [f"msg {i}" for i in ids])


if __name__ == "__main__":
    main()
