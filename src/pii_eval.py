"""PII redaction recall for redact.py on a labeled PII dataset, optionally through the summarizer too.

Default data: ai4privacy/pii-masking-400k (English rows). Each row has source_text plus privacy_mask, a list of
{"label", "start", "end", "value"} for every PII item. Any local .csv/.json/.jsonl file with a text column and a
spans column in that same format works too (e.g. our own labeled chats): -d my_chats.jsonl --text_col text --spans_col spans

Per PII item, after redaction:
  caught  = no piece of it survives        partial = some pieces survive (e.g. "john." of an email)
  missed  = it is still fully there
Recall = caught / total. "targeted" recall covers only what redact.py tries to catch (emails, phones, names);
other labels (addresses, IDs, dates...) are reported separately.
Over-redaction: share of ordinary (non-PII) words twaiut hat got destroyed.
With -s: also summarize the redacted text and count PII values that appear in the final summary (= leaked to the report).

Run on WAVE:
  python pii_eval.py -n 200
  python pii_eval.py -n 50 -s google/flan-t5-large
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd
from datasets import load_dataset

from redact import scan_batch

# ai4privacy label -> what redact.py targets; anything else counts as "other"
TARGETED = {"EMAIL": "email", "TELEPHONENUM": "phone", "PHONE": "phone", "GIVENNAME": "name",
            "SURNAME": "name", "FIRSTNAME": "name", "LASTNAME": "name", "NAME": "name", "PERSON": "name"}

p = argparse.ArgumentParser()
p.add_argument("-d", "--dataset", default="ai4privacy/pii-masking-400k", help="HF dataset or local .csv/.json/.jsonl")
p.add_argument("--split", default="validation")
p.add_argument("--language", default="en", help="Keep rows with this 'language' value (HF only); '' = all")
p.add_argument("--text_col", default="source_text")
p.add_argument("--spans_col", default="privacy_mask")
p.add_argument("-n", "--num_samples", type=int, default=200)
p.add_argument("-s", "--summarizer", help="Also summarize redacted text and check PII leaking into the summary")
p.add_argument("-l", "--length", type=int, default=10)
p.add_argument("-il", "--input_length", type=int, default=1024)
p.add_argument("-o", "--output", default="results/pii_eval.json")


def load_rows(a):
    if os.path.isfile(a.dataset):
        ext = os.path.splitext(a.dataset)[1].lower()
        df = pd.read_csv(a.dataset) if ext == ".csv" else pd.read_json(a.dataset, lines=ext == ".jsonl")
        rows = df.to_dict("records")[: a.num_samples or None]
    else:  # streaming: take n rows without downloading the whole split
        ds = load_dataset(a.dataset, split=a.split, streaming=True)
        if a.language:
            ds = ds.filter(lambda r: r.get("language", a.language) == a.language)  # datasets without the column pass
        rows = list(ds.take(a.num_samples))
    out = []
    for r in rows:
        text, spans = r[a.text_col], r[a.spans_col]
        spans = json.loads(spans) if isinstance(spans, str) else list(spans)  # CSV stores JSON text
        out.append((text, [{**s, "value": s.get("value") or text[s["start"]:s["end"]]} for s in spans]))  # some datasets only give offsets
    return out


def status(pii, out):
    if pii in out:
        return "missed"
    pieces = [x for x in re.split(r"\W+", pii) if len(x) >= 3]
    return "partial" if any(x in out for x in pieces) else "caught"


def plain_words(text, spans):
    """Ordinary words outside every PII span, for the over-redaction check."""
    keep = list(text)
    for s in spans:
        keep[s["start"]:s["end"]] = " " * (s["end"] - s["start"])
    return re.findall(r"[A-Za-z]{3,}", "".join(keep))


def main(a):
    rows = load_rows(a)
    texts = [t for t, _ in rows]
    redacted = scan_batch({"text": texts}, {"map": {}, "count": 0}, bare_senders=True)["text"]
    summaries = None
    if a.summarizer:
        from summarize import summarize_text  # only import torch/transformers when needed
        summaries = []
        for i, r in enumerate(redacted):
            summaries.append(summarize_text(r, a.summarizer, a.length, a.input_length))
            print(f"[{i + 1}/{len(redacted)}] summarized")

    by_label, examples = defaultdict(Counter), defaultdict(list)
    leaks, words_total, words_kept = Counter(), 0, 0
    for i, ((text, spans), red) in enumerate(zip(rows, redacted)):
        for s in spans:
            st = status(s["value"], red)
            by_label[s["label"]][st] += 1
            if st != "caught" and len(examples[s["label"]]) < 2:
                examples[s["label"]].append({"value": s["value"], "status": st, "redacted_text": red[:300]})
            if summaries and s["value"] in summaries[i]:
                leaks[TARGETED.get(s["label"], "other")] += 1
        words = plain_words(text, spans)
        words_total += len(words)
        words_kept += sum(w in red for w in words)

    def recall(labels):
        c = sum((by_label[l] for l in labels), Counter())
        return round(c["caught"] / sum(c.values()), 3) if c else None, dict(c)

    groups = defaultdict(list)
    for l in by_label:
        groups[TARGETED.get(l, "other")].append(l)
    targeted = [l for l in by_label if l in TARGETED]
    res = {
        "dataset": a.dataset, "rows": len(rows),
        "recall_targeted": recall(targeted)[0],
        "recall_by_group": {g: recall(ls)[0] for g, ls in groups.items()},
        "counts_by_label": {l: dict(c) for l, c in sorted(by_label.items())},
        "plain_words_kept": round(words_kept / max(words_total, 1), 3),
        "miss_examples": examples,
    }
    if summaries:
        n_targeted = sum(sum(by_label[l].values()) for l in targeted)
        res["summary_leaks"] = dict(leaks)
        res["summary_leak_rate_targeted"] = round(sum(v for k, v in leaks.items() if k != "other") / max(n_targeted, 1), 3)
        res["examples_with_summary"] = [{"redacted": r[:300], "summary": s} for r, s in list(zip(redacted, summaries))[:5]]

    for g, ls in groups.items():
        r, c = recall(ls)
        print(f"{g:6} recall {r:.2f}  {c}")
    print(f"\nTARGETED recall (email/phone/name): {res['recall_targeted']}  (target 0.90)")
    print(f"plain words kept (1.0 = no over-redaction): {res['plain_words_kept']}")
    if summaries:
        print(f"targeted PII leaked into summaries: {res['summary_leak_rate_targeted']}  {dict(leaks)}")
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"Saved {a.output}")


if __name__ == "__main__":
    # self-check of the scoring rules, so a bug here can't fake a good recall
    assert status("a.b@x.com", "mail a.b@x.com") == "missed"
    assert status("john.doe@gmail.com", "john.*************") == "partial"
    assert status("Wanjiru", "Person0: hi") == "caught"
    assert plain_words("Hi Wanjiru here", [{"start": 3, "end": 10}]) == ["here"]
    main(p.parse_args())
