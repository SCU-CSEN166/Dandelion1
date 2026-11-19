"""Convert hand labels on the FarmerChat sample into the span format pii_eval.py expects.

You label data/farmerchat_kenya_sample.jsonl by editing two fields per row:
  pii_labels : list of {"text": "<exact words as they appear in the query>", "type": "<TYPE>"}
               TYPE is one of NAME, PHONE, EMAIL, LOCATION, OTHER
  key_info   : short note of what a good report should keep from the message

Example row after labeling:
  {"query": "Do pumpkins grow well in Burnt Forest, Uasin Gishu County?",
   "pii_labels": [{"text": "Burnt Forest", "type": "LOCATION"}, {"text": "Uasin Gishu County", "type": "LOCATION"}],
   "key_info": "asks whether pumpkins grow well in Burnt Forest"}

A row counts as reviewed once pii_labels or key_info is filled in. A query with no PII is
reviewed by writing key_info and leaving pii_labels as []. Rows with both empty are skipped
(use --include_unreviewed to keep them as "no PII").

Every occurrence of a label's text in the query is turned into a span with character offsets.
Output has the columns `text` and `spans`, so run the PII evaluation on it with:
  python eval/pii_eval.py -d data/farmerchat_kenya_labeled.jsonl --text_col text --spans_col spans -n 50 -o results/pii_eval_farmerchat.json
"""
import argparse
import json
import re
import sys
from collections import Counter

ALLOWED_TYPES = {"NAME", "PHONE", "EMAIL", "LOCATION", "OTHER"}


def find_spans(query: str, labels: list, row_index) -> list:
    spans = []
    for label in labels:
        text = (label.get("text") or "").strip()
        kind = (label.get("type") or "").strip().upper()
        if not text:
            raise ValueError(f"row {row_index}: a label has empty text: {label}")
        if kind not in ALLOWED_TYPES:
            raise ValueError(f"row {row_index}: type {kind!r} for {text!r} must be one of {sorted(ALLOWED_TYPES)}")
        hits = [m.start() for m in re.finditer(re.escape(text), query)]
        if not hits:
            raise ValueError(f"row {row_index}: label text {text!r} not found in the query. It must match exactly "
                             f"(same spelling and capitalization). Query: {query!r}")
        for start in hits:
            spans.append({"label": kind, "start": start, "end": start + len(text), "value": text})
    return sorted(spans, key=lambda s: (s["start"], s["end"]))


def main():
    parser = argparse.ArgumentParser(description="Turn hand labels into character-offset spans for pii_eval.py")
    parser.add_argument("-i", "--input", default="data/farmerchat_kenya_sample.jsonl")
    parser.add_argument("-o", "--output", default="data/farmerchat_kenya_labeled.jsonl")
    parser.add_argument("--include_unreviewed", action="store_true",
                        help="Keep rows with no labels and no key_info as having no PII")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    out_rows, errors, counts = [], [], Counter()
    skipped = 0
    for row in rows:
        labels = row.get("pii_labels") or []
        key_info = (row.get("key_info") or "").strip()
        if not labels and not key_info and not args.include_unreviewed:
            skipped += 1
            continue
        try:
            spans = find_spans(row["query"], labels, row.get("row_index"))
        except ValueError as e:
            errors.append(str(e))
            continue
        counts.update(s["label"] for s in spans)
        out_rows.append({"row_index": row.get("row_index"), "text": row["query"], "spans": spans, "key_info": key_info})

    if errors:
        print("Fix these labels, then run again:", file=sys.stderr)
        for e in errors:
            print(" -", e, file=sys.stderr)
        sys.exit(1)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Reviewed rows written: {len(out_rows)}  |  skipped (not reviewed yet): {skipped}")
    print(f"Spans by type: {dict(counts)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
