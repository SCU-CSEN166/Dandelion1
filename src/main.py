import os
import argparse
import torch
from datasets import Dataset
from load_dataset import prepare_dataset
from relevance import is_chitchat
from summarize import summarize_text

# The only columns --table_out writes: the ones the pipeline makes from redacted text. Every other column of the
# input file (a sender or phone column, the raw text, the list of redacted items) could hold personal data.
TABLE_OUT_COLUMNS = ["message_id", "text", "pii_count", "pii_notes", "relevance", "relevance_score", "event_type",
                     "participants", "summary"]


def _make_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def table_out_view(ds: Dataset) -> Dataset:
    """The dataset restricted to TABLE_OUT_COLUMNS, in that order."""
    return ds.select_columns([c for c in TABLE_OUT_COLUMNS if c in ds.column_names])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a WhatsApp dataset and summarize its content.")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Name of the dataset to process, either locally stored or from Huggingface")
    parser.add_argument("-s", "--summarizer", type=str, help="Name of the summarization model to use, either locally or from Huggingface (such as 'facebook/bart-large-cnn')")
    parser.add_argument("-l", "--length", type=int, default=100, help="Minimum length of the summary")
    parser.add_argument("-m", "--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate in the summary")
    parser.add_argument("-o", "--output", type=str, help="File to write the summary to")
    parser.add_argument("--input_out", type=str, help="File to write the redacted input text to")
    parser.add_argument("--table_out", type=str, help="File (.csv, .json, .jsonl, .parquet) for the processed rows: message_id, redacted text, pii_count, pii_notes, and relevance / extracted info / row summaries when computed (other input columns are left out)")
    parser.add_argument("--classify_relevance", action="store_true", help="Label each message Event or Chit-chat with the zero-shot classifier (always on when an output needs relevance: --ta_table_out, --review_out, --followups_out, --protect_sensitive, --filter_relevant)")
    parser.add_argument("--filter_relevant", action="store_true", help="Leave chit-chat out of the summarizer's input (implies --classify_relevance; the tables keep every message)")
    parser.add_argument("--classifier_model", type=str, default="facebook/bart-large-mnli", help="Model for zero-shot relevance classification")
    parser.add_argument("--extract_info", action="store_true", help="Extract event type and participant counts using QA model")
    parser.add_argument("--qa_model", type=str, default="deepset/roberta-base-squad2", help="Model for question answering extraction")
    parser.add_argument("--row_summaries", action="store_true", help="Generate row-level summary column in tabular output")
    parser.add_argument("--row_summarizer", type=str, default="facebook/bart-large-cnn", help="Model for per-row summaries (default: facebook/bart-large-cnn)")
    parser.add_argument("--review_out", type=str, help="File (.csv or .xlsx) for the human review queue: confidence, reasons to check, sensitive topic, and report text with sentences about individuals withheld")
    parser.add_argument("--followups_out", type=str, help="File (.csv or .xlsx) for the follow-up list: referrals, people to call back, return visits, reports to send, supplies")
    parser.add_argument("--protect_sensitive", action="store_true", help="Give the summarizer the protected report text (sentences about individuals in sensitive messages withheld)")
    parser.add_argument("--ta_table_out", type=str, help="File (.csv or .xlsx) to write one row per message in the TA's hidden-test format: relevance, activity_type, participants, duplicate_of, pii_count, expected_anonymized_text")

    args = parser.parse_args(argv)
    if bool(args.summarizer) != bool(args.output):
        parser.error("--summarizer and --output must be given together (the summary is only written to --output)")
    # Relevance comes from the classifier for every output that uses it
    args.classify_relevance = bool(args.classify_relevance or args.filter_relevant or args.ta_table_out
                                   or args.review_out or args.followups_out or args.protect_sensitive)
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available")

    dataset = prepare_dataset(
        args.dataset,
        classify_relevance=args.classify_relevance,
        classifier_model=args.classifier_model,
        extract_info=args.extract_info,
        qa_model=args.qa_model,
    )

    # Generate row-level summaries if requested
    if args.row_summaries:
        print(f"Generating row-level summaries for tabular dataset using {args.row_summarizer}...")
        row_sums = [
            summarize_text(t, args.row_summarizer, min_length=15, max_new_tokens=64, prefix="Summarize: ")
            for t in dataset["text"]
        ]
        dataset = dataset.add_column("summary", row_sums)

    # Export processed tabular data if requested (only the pipeline's own columns)
    if args.table_out:
        _make_parent_dir(args.table_out)
        table = table_out_view(dataset)
        ext = os.path.splitext(args.table_out)[1].lower()
        if ext in (".json", ".jsonl"):
            table.to_json(args.table_out, lines=(ext == ".jsonl"))
        elif ext == ".parquet":
            table.to_parquet(args.table_out)
        else:
            table.to_csv(args.table_out)
        print(f"Saved processed tabular dataset to {args.table_out}")

    # Export the table in the TA's hidden-test format, the review queue and the follow-up list if requested
    protected_texts = None
    if args.ta_table_out or args.review_out or args.followups_out or args.protect_sensitive:
        from ta_table import build_ta_table, write_table
        from insights import build_followups, build_review_table
        rows = list(dataset)
        ta_table = build_ta_table(rows)
        review = build_review_table(ta_table, rows) if args.review_out or args.protect_sensitive else None
        for path, make_table, label in (
            (args.ta_table_out, lambda: ta_table, "TA-format table"),
            (args.review_out, lambda: review, "review queue"),
            (args.followups_out, lambda: build_followups(ta_table), "follow-up list"),
        ):
            if path:
                _make_parent_dir(path)
                write_table(make_table(), path)
                print(f"Saved {label} to {path}")
        if args.protect_sensitive:
            protected_texts = list(review["report_text"])

    # Collect structured facts from extraction stage to enrich the summarizer prompt
    extracted_facts = []
    if args.extract_info and "event_type" in dataset.column_names:
        for row in dataset:
            ev = (row.get("event_type") or "").strip()
            part = (row.get("participants") or "").strip()
            if ev or part:
                fact = f"- Event: {ev or 'Field activity'}"
                if part:
                    fact += f" | Attendees/Participants: {part}"
                extracted_facts.append(fact)

    texts = protected_texts if protected_texts is not None else dataset["text"]
    if args.filter_relevant:  # only the summarizer's input is filtered; the tables above keep every message
        texts = [t for t, label in zip(texts, dataset["relevance"]) if not is_chitchat(label)]
    text_input = "\n".join(t for t in texts if t and t.strip())
    if args.input_out:
        _make_parent_dir(args.input_out)
        with open(args.input_out, "w", encoding="utf-8") as f:
            f.write(text_input)

    if args.summarizer:
        _make_parent_dir(args.output)
        summary = summarize_text(
            text_input,
            args.summarizer,
            args.length,
            max_new_tokens=args.max_new_tokens,
            extracted_info=extracted_facts if extracted_facts else None,
        )
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(summary)
        if summary:
            print(f"Saved summary to {args.output}")
        else:
            print(f"Warning: the summarizer generated nothing; wrote an empty {args.output}")

if __name__ == "__main__":
    main()
