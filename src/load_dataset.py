"""Dataset loading and preparation pipeline for Dandelion Africa field communications.

Loads raw data (.csv, .tsv, .json, .jsonl, .xlsx, .parquet, .txt, or Hugging Face dataset),
preprocesses boilerplate/system notices, redacts PII, and optionally classifies relevance
or extracts structured event information.

This is the only place that looks for the text column: every later stage reads 'text'.
"""
import os
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from preprocess import preprocess_dataset
from redact import process_dataset_pii
from relevance import classify_dataset_relevance, load_zero_shot_classifier
from extract import classify_activities, extract_dataset_info

# Columns that hold the message text in common datasets, used when there is no 'text' column
TEXT_COLUMNS = ["transcript", "dialogue", "message", "query", "content", "source_text", "sentence", "post", "line"]


def load_raw_dataset(dataset_name: str, split: str | None = None) -> Dataset:
    """Load one split of a local tabular/text file or a Hugging Face dataset.

    For a Hugging Face dataset with several splits, the split is `split` if given, else "train", else the first one.
    """
    if not os.path.isfile(dataset_name):
        ds = load_dataset(dataset_name, split=split) if split else load_dataset(dataset_name)
        if isinstance(ds, DatasetDict):
            ds = ds["train"] if "train" in ds else ds[next(iter(ds))]
        return ds

    ext = os.path.splitext(dataset_name)[1].lower()
    if ext == ".csv":
        return load_dataset("csv", data_files=dataset_name, split="train")
    if ext in (".json", ".jsonl"):
        return load_dataset("json", data_files=dataset_name, split="train")
    if ext == ".parquet":
        return load_dataset("parquet", data_files=dataset_name, split="train")
    if ext == ".tsv":
        return Dataset.from_pandas(pd.read_csv(dataset_name, sep="\t", encoding_errors="replace"))
    if ext in (".xlsx", ".xls"):
        return Dataset.from_pandas(pd.read_excel(dataset_name))
    if ext == ".txt":
        with open(dataset_name, "r", encoding="utf-8", errors="replace") as f:
            lines = [line.strip() for line in f if line.strip()]
        return Dataset.from_dict({"text": lines})
    raise ValueError(f"Unsupported file format '{ext}' for file: {dataset_name}")


def _add_standard_columns(ds: Dataset) -> Dataset:
    """Give the dataset a 'text' column, an 'original_text' copy of it, and a 'message_id' (1, 2, ... if missing).

    The text column is renamed rather than copied, so 'original_text' is the only column with the unredacted text.
    message_id is set here, before anything can drop rows, so it always points at the same message.
    """
    if "text" not in ds.column_names:
        col = next((c for c in TEXT_COLUMNS if c in ds.column_names), None)
        if col is not None:
            ds = ds.rename_column(col, "text")
    if "text" in ds.column_names and "original_text" not in ds.column_names:
        ds = ds.map(lambda row: {"original_text": row["text"]})
    if "message_id" not in ds.column_names:
        ds = ds.add_column("message_id", list(range(1, len(ds) + 1)))
    return ds


def prepare_dataset(
    dataset_name: str,
    classify_relevance: bool = False,
    classifier_model: str = "facebook/bart-large-mnli",
    extract_info: bool = False,
    qa_model: str = "deepset/roberta-base-squad2",
    clean_text: bool = True,
    split: str | None = None,
) -> Dataset:
    """Load one split, clean boilerplate, redact PII, and optionally classify relevance or extract event info."""
    raw = _add_standard_columns(load_raw_dataset(dataset_name, split=split))

    # 1. Clean WhatsApp system banners, attachments, and placeholders first
    if clean_text:
        raw = preprocess_dataset(raw)

    # 2. Redact PII (phone numbers, emails, names and places)
    processed = process_dataset_pii(raw)

    # 3. Classify relevance and, for Events, the activity category, with the same zero-shot model
    #    (rows are never dropped here: main.py filters only the summarizer's input)
    if classify_relevance:
        classifier = load_zero_shot_classifier(classifier_model)
        processed = classify_dataset_relevance(processed, score_col="relevance_score", classifier=classifier)
        processed = classify_activities(processed, classifier=classifier)

    # 4. Ask the QA model for each message's event type and head count
    if extract_info:
        processed = extract_dataset_info(processed, model_name=qa_model)

    return processed
