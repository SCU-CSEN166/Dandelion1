"""Score main.py's report.md: faithfulness, completeness and conciseness (after FineSurE, Song et al., ACL 2024).

  faithfulness  share of report sentences the messages support (precision: does the report invent anything?)
  completeness  share of key facts the report states (recall: does it drop any Event?)
  conciseness   share of report sentences that state some key fact (low = filler or chit-chat)

A judge model answers one question per sentence ("supported by the messages?") and one per key fact ("which
sentences state it?"). The source is messages.csv's anonymized text, the same Person0 / Place0 aliases the report
uses. Key facts: by default one per Event in the TA's hidden-test key (activity type and head count, duplicates
counted once); for other data, pass --keyfacts, a text file with one fact per line, written or checked by a person.

The judge is a model too: read the unsupported sentences and missed facts it lists before trusting the scores.

Run:  python eval/eval_summary.py -r outputs/hidden_test/report.md -m outputs/hidden_test/messages.csv
"""
import argparse
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))  # pipeline modules
from tables import split_sentences  # noqa: E402

KEY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "hidden_test_outputs.xlsx")
DEFAULT_JUDGE = "Qwen/Qwen2.5-7B-Instruct"

# Section titles of summarize.DEFAULT_SYSTEM_PROMPT's report template: headings, not claims
_HEADING = re.compile(r"^(?:field outcomes report|practical actions completed|blockers (?:&|and) technical issues|"
                      r"outcomes (?:&|and) resolutions|pending follow-ups)\s*:?\s*", re.I)

FAITHFUL_PROMPT = (
    "Messages from a field team's group chat:\n\n{source}\n\n"
    "Statement from a report written from these messages:\n{sentence}\n\n"
    "Is every detail of the statement supported by the messages (names, places, numbers, what happened)? "
    "Answer with one word: Yes or No."
)
ALIGN_PROMPT = (
    "Numbered sentences of a report:\n\n{sentences}\n\n"
    "Fact: {fact}\n\n"
    "Which sentences state this fact (the same activity, and the same number if the fact has one)? "
    "Answer with the sentence numbers separated by commas, or None."
)


def report_sentences(report: str) -> list[str]:
    """The claims of a report: sentences without markdown, bullets, numbering or template headings."""
    out = []
    for line in report.splitlines():
        line = re.sub(r"[*_`]+", "", line).strip()
        line = re.sub(r"^(?:#+|[-•]|\d+[.)])\s*", "", line)
        line = _HEADING.sub("", line)
        if not line or line.endswith(":"):
            continue
        out += [s for s in split_sentences(line) if len(s.split()) >= 3]  # drops "None.", "N/A"
    return out


def key_facts_from_key(key_path: str) -> list[str]:
    """One fact per Event in the TA key, duplicates (duplicate_of set) counted once."""
    key = pd.read_excel(key_path)
    facts = []
    for _, row in key[key["relevance"] == "Event"].iterrows():
        if str(row["duplicate_of"]).strip().lower() not in ("none", "", "nan"):
            continue
        count = "" if pd.isna(row["participants"]) else f" with {int(row['participants'])} participants"
        facts.append(f"{row['activity_type']}{count}")
    return facts


def parse_yes(answer: str) -> bool:
    return bool(re.match(r"\W*yes\b", answer, re.I))


def parse_numbers(answer: str, n: int) -> list[int]:
    """Sentence numbers (1..n) in the judge's answer; [] for 'None' or anything out of range."""
    return sorted({int(x) for x in re.findall(r"\d+", answer) if 1 <= int(x) <= n})


def scores(supported: list[bool], alignments: list[list[int]], n_sentences: int) -> dict:
    aligned = {i for nums in alignments for i in nums}
    ratio = lambda a, b: round(a / b, 3) if b else None  # noqa: E731
    return {"faithfulness": ratio(sum(supported), len(supported)),
            "completeness": ratio(sum(bool(a) for a in alignments), len(alignments)),
            "conciseness": ratio(len(aligned), n_sentences)}


class Judge:
    def __init__(self, model_name: str):
        from summarize import load_model  # only load torch/transformers when judging
        self.tokenizer, self.model = load_model(model_name)

    def ask(self, prompt: str, max_new_tokens: int = 16) -> str:
        from summarize import _generate_chunk
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True)
        # The whole chat goes in every prompt; a long chat would need splitting into chunks
        ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        return _generate_chunk(self.model, self.tokenizer, ids, 0, max_new_tokens, self.model.device)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-r", "--report", required=True, help="report.md from main.py -s")
    p.add_argument("-m", "--messages", required=True, help="messages.csv from the same run (the source)")
    p.add_argument("--keyfacts", help="Text file, one key fact per line (default: the Events of the TA key)")
    p.add_argument("--key", default=KEY, help="TA key to take key facts from when --keyfacts is not given")
    p.add_argument("-j", "--judge", default=DEFAULT_JUDGE, help="Instruction model that answers the questions")
    p.add_argument("-o", "--output", help="JSON with every judgment (default: summary_eval.json next to the report)")
    args = p.parse_args()

    report = open(args.report, encoding="utf-8").read()
    sentences = report_sentences(report)
    if args.keyfacts:
        facts = [line.strip() for line in open(args.keyfacts, encoding="utf-8") if line.strip()]
    else:
        facts = key_facts_from_key(args.key)
    msgs = pd.read_csv(args.messages, dtype=str).fillna("")
    source = "\n".join(f"[{i}] {t}" for i, t in zip(msgs["message_id"], msgs["text"]) if t.strip())
    print(f"{len(sentences)} report sentences, {len(facts)} key facts, {len(msgs)} messages")

    judge = Judge(args.judge)
    supported = [parse_yes(judge.ask(FAITHFUL_PROMPT.format(source=source, sentence=s))) for s in sentences]
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
    alignments = [parse_numbers(judge.ask(ALIGN_PROMPT.format(sentences=numbered, fact=f)), len(sentences))
                  for f in facts]
    result = scores(supported, alignments, len(sentences))

    print(json.dumps(result, indent=2))
    for s, ok in zip(sentences, supported):
        if not ok:
            print("   unsupported:", s)
    for f, nums in zip(facts, alignments):
        if not nums:
            print("   missed fact:", f)

    output = args.output or os.path.join(os.path.dirname(args.report) or ".", "summary_eval.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump({**result, "judge": args.judge, "report": args.report,
                   "sentences": [{"sentence": s, "supported": ok} for s, ok in zip(sentences, supported)],
                   "key_facts": [{"fact": f, "sentences": nums} for f, nums in zip(facts, alignments)]},
                  f, indent=2, ensure_ascii=False)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
