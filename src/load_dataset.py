import json
import os

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from redact import process_dataset_pii

# Local files we can read (anything else is treated as a Hugging Face dataset name)
LOCAL_EXTENSIONS = {".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".xls"}
TEXT_COLUMN_CANDIDATES = ["text", "transcript", "message", "content", "body", "dialogue", "utterance", "caption"]


def _read_json(path: str) -> pd.DataFrame:
    """Read a .json (list of records, or an object holding a list of records) or .jsonl file."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]  # JSON Lines
    if isinstance(data, dict):  # e.g. {"messages": [ {...}, {...} ]}
        lists = [v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
        if not lists:
            raise ValueError(f"{path}: expected a list of records or an object that contains one")
        data = lists[0]
    return pd.DataFrame(data)


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, encoding_errors="replace")
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t", encoding_errors="replace")
    if ext in (".json", ".jsonl"):
        return _read_json(path)
    return pd.read_excel(path)  # .xlsx / .xls (needs openpyxl)


def _pick_text_column(df: pd.DataFrame, text_col: str | None, path: str) -> str:
    if text_col:
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in {path}. Available: {list(df.columns)}")
        return text_col
    lower = {str(c).lower(): c for c in df.columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lower:
            return lower[candidate]
    text_like = [c for c in df.columns if df[c].dtype == object]
    if len(text_like) == 1:
        return text_like[0]
    raise ValueError(f"Could not tell which column of {path} holds the text. Pass --text_col. Available: {list(df.columns)}")


def load_local_file(path: str, text_col: str | None = None) -> DatasetDict:
    """Load a local .csv/.tsv/.json/.jsonl/.xlsx file as a DatasetDict with one 'train' split and a 'text' column."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")
    df = _read_table(path)
    column = _pick_text_column(df, text_col, path)
    texts = [str(t).strip() for t in df[column].dropna()]
    texts = [t for t in texts if t]
    if not texts:
        raise ValueError(f"{path}: column '{column}' has no text")
    return DatasetDict({"train": Dataset.from_dict({"text": texts})})


def load_raw_dataset(dataset_name: str, text_col: str | None = None) -> DatasetDict:
    """A local file (csv/tsv/json/jsonl/xlsx) or a Hugging Face dataset name -> DatasetDict with a 'text' column."""
    if os.path.splitext(dataset_name)[1].lower() in LOCAL_EXTENSIONS:
        return load_local_file(dataset_name, text_col)
    raw = load_dataset(dataset_name)
    if text_col and text_col != "text":
        raw = raw.rename_column(text_col, "text")
    return raw


def prepare_dataset(dataset_name: str | None = None, text_col: str | None = None, extra_texts: list[str] | None = None):
    """Load, optionally add extra texts (e.g. audio transcripts), then redact everything in one pass.

    Redacting in one pass keeps names consistent (the same person gets the same Person0, Person1, ...)."""
    if dataset_name:
        raw = load_raw_dataset(dataset_name, text_col)
        if extra_texts:
            train = raw["train"].select_columns(["text"])
            raw["train"] = Dataset.from_dict({"text": list(train["text"]) + list(extra_texts)})
    elif extra_texts:
        raw = DatasetDict({"train": Dataset.from_dict({"text": list(extra_texts)})})
    else:
        raise ValueError("Give a dataset name or file, or audio files")
    return process_dataset_pii(raw)
