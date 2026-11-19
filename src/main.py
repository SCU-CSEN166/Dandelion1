"""Turn exported WhatsApp messages / voice-note transcripts into anonymized tables and a field report.

Writes to the --out folder:
  messages.csv    one row per message: anonymized text, relevance, activity type, head count, duplicates,
                  sensitive topic, and confidence with the reasons to review it (see tables.py)
  follow_ups.csv  one row per sentence that asks someone to act
  report.md       with -s: the summary report, written from the anonymized text with sentences about one person
                  in sensitive messages withheld

Run:  python src/main.py -d data/hidden_test_transcripts.xlsx -o outputs/hidden_test --extract_info -s <model>
"""
import argparse
import os

import torch

from load_dataset import prepare_dataset
from relevance import is_chitchat
from summarize import summarize_text
from tables import FOLLOWUP_COLUMNS, MESSAGE_COLUMNS, build_followups, build_message_table, write_table


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-d", "--dataset", required=True, help="Local file (.csv, .tsv, .json, .jsonl, .xlsx, .parquet, .txt) or Hugging Face dataset")
    parser.add_argument("-o", "--out", required=True, help="Folder for messages.csv, follow_ups.csv and report.md")
    parser.add_argument("-s", "--summarizer", help="Model that writes report.md (e.g. meta-llama/Llama-3.1-8B-Instruct); no report without it")
    parser.add_argument("-l", "--length", type=int, default=100, help="Minimum length of the report in tokens")
    parser.add_argument("-m", "--max_new_tokens", type=int, default=1024, help="Maximum length of the report in tokens")
    parser.add_argument("--filter_relevant", action="store_true", help="Write the report from Event messages only (the tables keep every message)")
    parser.add_argument("--classifier_model", default="facebook/bart-large-mnli", help="Zero-shot model for Event / Chit-chat")
    parser.add_argument("--extract_info", action="store_true", help="Also ask a QA model for each Event's activity and head count")
    parser.add_argument("--qa_model", default="deepset/roberta-base-squad2", help="QA model for --extract_info")
    return parser.parse_args(argv)


def report_facts(rows: list[dict]) -> list[str]:
    """The QA model's answers, one line per message, given to the summarizer as extra context."""
    facts = []
    for row in rows:
        event = (row.get("event_type") or "").strip()
        count = (row.get("participants") or "").strip()
        if event or count:
            facts.append(f"- Event: {event or 'Field activity'}" + (f" | Attendees/Participants: {count}" if count else ""))
    return facts


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available")
    os.makedirs(args.out, exist_ok=True)

    dataset = prepare_dataset(args.dataset, classify_relevance=True, classifier_model=args.classifier_model,
                              extract_info=args.extract_info, qa_model=args.qa_model)
    rows = list(dataset)
    messages = build_message_table(rows)
    for name, table, columns in (("messages.csv", messages, MESSAGE_COLUMNS),
                                 ("follow_ups.csv", build_followups(messages), FOLLOWUP_COLUMNS)):
        write_table(table, os.path.join(args.out, name), columns)
        print(f"Saved {os.path.join(args.out, name)}")

    if args.summarizer:
        texts = messages["report_text"]
        if args.filter_relevant:  # only the report is filtered; the tables keep every message
            texts = texts[[not is_chitchat(label) for label in messages["relevance"]]]
        facts = report_facts(rows) if args.extract_info else []
        summary = summarize_text("\n".join(t for t in texts if t and t.strip()), args.summarizer, args.length,
                                 max_new_tokens=args.max_new_tokens, extracted_info=facts or None)
        path = os.path.join(args.out, "report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Saved {path}" if summary else f"Warning: the summarizer generated nothing; wrote an empty {path}")


if __name__ == "__main__":
    main()
