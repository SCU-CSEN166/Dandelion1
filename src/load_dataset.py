"""Dataset loading and preparation pipeline for Dandelion Africa field communications.

Loads raw data (.csv, .tsv, .json, .jsonl, .xlsx, .parquet, .txt, or Hugging Face dataset),
preprocesses boilerplate/system notices, redacts PII, and optionally classifies relevance
or extracts structured event information.
"""
import os
from datasets import Dataset, DatasetDict, load_dataset
from preprocess import preprocess_dataset
from redact import process_dataset_pii
from relevance import classify_dataset_relevance, filter_relevant_lines
from extract import extract_dataset_info

try:
    import pandas as pd
except ImportError:
    pd = None


def load_raw_dataset(dataset_name: str, split: str | None = None) -> Dataset | DatasetDict:
    """Load a dataset from a local tabular/text file or the Hugging Face Hub."""
    if os.path.isfile(dataset_name):
        ext = os.path.splitext(dataset_name)[1].lower()
        if ext == ".csv":
            return load_dataset("csv", data_files=dataset_name)
        elif ext == ".tsv":
            if pd is None:
                raise ImportError("pandas is required to load .tsv files")
            df = pd.read_csv(dataset_name, sep="\t", encoding_errors="replace")
            return DatasetDict({"train": Dataset.from_pandas(df)})
        elif ext in (".json", ".jsonl"):
            return load_dataset("json", data_files=dataset_name)
        elif ext in (".xlsx", ".xls"):
            if pd is None:
                raise ImportError("pandas and openpyxl are required to load Excel files")
            df = pd.read_excel(dataset_name)
            if "text" not in df.columns:
                for col in ["transcript", "dialogue", "message", "query", "content", "source_text", "sentence", "post", "line"]:
                    if col in df.columns:
                        df["text"] = df[col]
                        break
            return DatasetDict({"train": Dataset.from_pandas(df)})
        elif ext == ".parquet":
            return load_dataset("parquet", data_files=dataset_name)
        elif ext == ".txt":
            with open(dataset_name, "r", encoding="utf-8", errors="replace") as f:
                lines = [line.strip() for line in f if line.strip()]
            return DatasetDict({"train": Dataset.from_dict({"text": lines})})
        else:
            raise ValueError(f"Unsupported file format '{ext}' for file: {dataset_name}")
    else:
        if split:
            return load_dataset(dataset_name, split=split)
        return load_dataset(dataset_name)


def prepare_dataset(
    dataset_name: str,
    classify_relevance: bool = False,
    filter_relevant: bool = False,
    classifier_model: str = "facebook/bart-large-mnli",
    candidate_labels: list[str] | None = None,
    extract_info: bool = False,
    qa_model: str = "deepset/roberta-base-squad2",
    clean_text: bool = True,
    split: str | None = None,
) -> Dataset | DatasetDict:
    """Load a dataset, clean boilerplate, redact PII, and optionally classify relevance or extract event info."""
    raw = load_raw_dataset(dataset_name, split=split)

    # Ensure dataset splits have a 'text' column if another candidate column exists
    def _ensure_text_col(ds):
        if "text" not in ds.column_names:
            for col in ["transcript", "dialogue", "message", "query", "content", "source_text", "sentence", "post", "line"]:
                if col in ds.column_names:
                    return ds.map(lambda row: {"text": row[col]})
        return ds

    if isinstance(raw, DatasetDict):
        raw = DatasetDict({k: _ensure_text_col(v) for k, v in raw.items()})
    elif isinstance(raw, Dataset):
        raw = _ensure_text_col(raw)

    # Keep the untouched text so outputs can show the original next to the redacted version
    def _keep_original(ds):
        if "text" not in ds.column_names or "original_text" in ds.column_names:
            return ds
        return ds.map(lambda row: {"original_text": row["text"]})

    if isinstance(raw, DatasetDict):
        raw = DatasetDict({k: _keep_original(v) for k, v in raw.items()})
    elif isinstance(raw, Dataset):
        raw = _keep_original(raw)

    # 1. Clean WhatsApp system banners, attachments, and placeholders first
    if clean_text:
        raw = preprocess_dataset(raw)

    # 2. Redact PII (phone numbers, emails, and names)
    processed = process_dataset_pii(raw)

    # 3. Classify relevance and optionally filter out casual chit-chat
    if classify_relevance or filter_relevant:
        processed = classify_dataset_relevance(
            processed,
            model_name=classifier_model,
            candidate_labels=candidate_labels,
        )
        if filter_relevant:
            processed = filter_relevant_lines(processed)

    # 4. Extract structured event type and participant counts
    if extract_info:
        processed = extract_dataset_info(
            processed,
            model_name=qa_model,
        )

    return processed
