import sys
import os
import argparse
import torch
from datasets import DatasetDict
from load_dataset import prepare_dataset
from summarize import summarize_text

def main():
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available")

    parser = argparse.ArgumentParser(description="Process a WhatsApp dataset and summarize its content.")
    parser.add_argument("-d", "--dataset", type=str, help="Name of the dataset to process, either locally stored or from Huggingface")
    parser.add_argument("-s", "--summarizer", type=str, help="Name of the summarization model to use, either locally or from Huggingface (such as 'facebook/bart-large-cnn')")
    parser.add_argument("-l", "--length", type=int, default=100, help="Minimum length of the summary")
    parser.add_argument("-m", "--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate in the summary")
    parser.add_argument("-b", "--use_blocks", action="store_true", help="Split long inputs into blocks for hierarchical summarization")
    parser.add_argument("-o", "--output", type=str, help="File to write the summary to")
    parser.add_argument("--input_out", type=str, help="File to write the redacted input text to")
    parser.add_argument("--table_out", type=str, help="File to write the processed tabular dataset with relevance/extracted info to (.csv, .json, .parquet)")
    parser.add_argument("--classify_relevance", action="store_true", help="Classify line relevance with zero-shot classifier")
    parser.add_argument("--filter_relevant", action="store_true", help="Filter out irrelevant lines before summarization")
    parser.add_argument("--classifier_model", type=str, default="facebook/bart-large-mnli", help="Model for zero-shot relevance classification")
    parser.add_argument("--extract_info", action="store_true", help="Extract event type and participant counts using QA model")
    parser.add_argument("--qa_model", type=str, default="deepset/roberta-base-squad2", help="Model for question answering extraction")
    parser.add_argument("--row_summaries", action="store_true", help="Generate row-level summary column in tabular output")
    parser.add_argument("--row_summarizer", type=str, default="facebook/bart-large-cnn", help="Model for per-row summaries (default: facebook/bart-large-cnn)")
    parser.add_argument("--review_out", type=str, help="File (.csv or .xlsx) for the human review queue: confidence, reasons to check, sensitive topic, and report text with sentences about individuals withheld")
    parser.add_argument("--followups_out", type=str, help="File (.csv or .xlsx) for the follow-up list: referrals, people to call back, return visits, reports to send, supplies")
    parser.add_argument("--protect_sensitive", action="store_true", help="Give the summarizer the protected report text (sentences about individuals in sensitive messages withheld)")
    parser.add_argument("--ta_table_out", type=str, help="File (.csv or .xlsx) to write one row per message in the TA's hidden-test format: relevance, activity_type, participants, duplicate_of, pii_count, pii_items, expected_anonymized_text")

    args = parser.parse_args()

    if not args.dataset:
        print("Dataset name is required")
        sys.exit(1)

    dataset_name = args.dataset
    dataset = prepare_dataset(
        dataset_name,
        classify_relevance=args.classify_relevance,
        filter_relevant=args.filter_relevant,
        classifier_model=args.classifier_model,
        extract_info=args.extract_info,
        qa_model=args.qa_model,
    )

    # Resolve active split gracefully (supports Dataset and DatasetDict)
    if isinstance(dataset, DatasetDict):
        split_name = "train" if "train" in dataset else next(iter(dataset.keys()))
        active_split = dataset[split_name]
    else:
        active_split = dataset

    # Generate row-level summaries if requested
    if args.row_summaries:
        row_model = args.row_summarizer or args.summarizer or "facebook/bart-large-cnn"
        print(f"Generating row-level summaries for tabular dataset using {row_model}...")
        row_sums = [
            summarize_text(t, row_model, min_length=15, max_new_tokens=64, prefix="Summarize: ")
            for t in active_split["text"]
        ]
        active_split = active_split.add_column("summary", row_sums)

    # Export processed tabular data if requested
    if args.table_out:
        os.makedirs(os.path.dirname(args.table_out) or ".", exist_ok=True)
        ext = os.path.splitext(args.table_out)[1].lower()
        if ext == ".csv":
            active_split.to_csv(args.table_out)
        elif ext in (".json", ".jsonl"):
            active_split.to_json(args.table_out, lines=(ext == ".jsonl"))
        elif ext == ".parquet":
            active_split.to_parquet(args.table_out)
        else:
            active_split.to_csv(args.table_out)
        print(f"Saved processed tabular dataset to {args.table_out}")

    # Export the table in the TA's hidden-test format, the review queue and the follow-up list if requested
    protected_texts = None
    if args.ta_table_out or args.review_out or args.followups_out or args.protect_sensitive:
        from ta_table import build_ta_table, write_ta_table
        from insights import build_followups, build_review_table, write_table
        rows = list(active_split)
        ta_table = build_ta_table(rows)
        review = build_review_table(ta_table, rows)
        for path, table, label, writer in (
            (args.ta_table_out, ta_table, "TA-format table", write_ta_table),
            (args.review_out, review, "review queue", write_table),
            (args.followups_out, build_followups(ta_table), "follow-up list", write_table),
        ):
            if path:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                writer(table, path)
                print(f"Saved {label} to {path}")
        if args.protect_sensitive:
            protected_texts = list(review["report_text"])

    # Collect structured facts from extraction stage to enrich the summarizer prompt
    extracted_facts = []
    if args.extract_info and "event_type" in active_split.column_names:
        for row in active_split:
            ev = (row.get("event_type") or "").strip()
            part = (row.get("participants") or "").strip()
            if ev or part:
                fact = f"- Event: {ev or 'Field activity'}"
                if part:
                    fact += f" | Attendees/Participants: {part}"
                extracted_facts.append(fact)

    text_input = "\n".join(protected_texts if protected_texts is not None else active_split["text"])
    if args.input_out:
        os.makedirs(os.path.dirname(args.input_out) or ".", exist_ok=True)
        with open(args.input_out, "w", encoding="utf-8") as f:
            f.write(text_input)

    if args.summarizer and args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        summary = summarize_text(
            text_input,
            args.summarizer,
            args.length,
            args.use_blocks,
            max_new_tokens=args.max_new_tokens,
            extracted_info=extracted_facts if extracted_facts else None,
        )
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Saved summary to {args.output}")

if __name__ == "__main__":
    main()
