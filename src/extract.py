"""Extraction stage using a question answering model to extract structured event information.

Extracts:
1. The type of event or activity being discussed (event_type).
2. The number of people participating or attended (participants).

Returns the dataset with additional columns for this extra information.
Designed for Dandelion Africa field communications (WhatsApp and voice-note transcripts).
"""
from functools import lru_cache
import argparse
import os
import re
import sys
import torch
from transformers import pipeline, Pipeline
from datasets import Dataset, DatasetDict

try:
    import pandas as pd
except ImportError:
    pd = None

# Default QA model: SQuAD 2.0-trained model handles unanswerable questions accurately
DEFAULT_MODEL = "deepset/roberta-base-squad2"

DEFAULT_EVENT_QUESTION = "What event, activity, or session is being held or discussed?"
DEFAULT_PARTICIPANTS_QUESTION = "How many people or participants attended, came, or were served?"

# Numbers dictionary for word-to-integer conversion
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

CANDIDATE_TEXT_COLUMNS = [
    "text",
    "transcript",
    "dialogue",
    "message",
    "query",
    "content",
    "source_text",
    "sentence",
    "post",
    "line",
]


@lru_cache(maxsize=2)
def load_qa_pipeline(
    model_name: str = DEFAULT_MODEL,
    device: str | int | None = None,
) -> Pipeline:
    """Load and cache a question-answering pipeline.

    Args:
        model_name: Hugging Face QA model identifier (default: 'deepset/roberta-base-squad2').
        device: Device ID (0 for CUDA, -1 for CPU, or 'cuda'/'cpu').

    Returns:
        A transformers Pipeline for question-answering.
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "question-answering",
        model=model_name,
        device=device,
    )


def parse_participant_count(text: str | None) -> str:
    """Extract and normalize participant count from a text span into an integer string.

    Handles digits ('42 women', 'around 35') and number words ('forty-two', 'twenty five').
    If no numeric count is found, returns the raw text stripped, or empty string.

    Args:
        text: Extracted answer string from the QA model.

    Returns:
        String representing the participant count (e.g. '42', '25'), or empty string.
    """
    if not text or not str(text).strip():
        return ""

    cleaned = str(text).strip().lower()

    # Match numeric digits (e.g. 42, 1,000, 25)
    digit_match = re.search(r"\b\d[\d,]*\b", cleaned)
    if digit_match:
        try:
            return str(int(digit_match.group(0).replace(",", "")))
        except ValueError:
            pass

    # Match word numbers (e.g. 'forty two', 'twenty-five', 'sixty')
    words = re.findall(r"[a-z]+", cleaned)
    total, current = 0, 0
    found_number = False

    for w in words:
        if w in _UNITS:
            current += _UNITS[w]
            found_number = True
        elif w in _TENS:
            current += _TENS[w]
            found_number = True
        elif w == "hundred":
            current = max(current, 1) * 100
            found_number = True
        elif w == "thousand":
            current = max(current, 1) * 1000
            total += current
            current = 0
            found_number = True
        elif found_number:
            break

    total += current
    if found_number and total > 0:
        return str(total)

    # If it's a descriptive phrase (e.g. 'a few elders', 'unknown'), return stripped text
    return str(text).strip()


def clean_event_label(text: str | None) -> str:
    """Clean up extracted event string by stripping leading articles/fillers and punctuation."""
    if not text or not str(text).strip():
        return ""

    cleaned = str(text).strip()
    # Strip trailing punctuation
    cleaned = re.sub(r"[.,;:!]+$", "", cleaned).strip()

    # Strip common leading demonstratives, possessives, and articles
    cleaned = re.sub(
        r"^(?:today's|yesterday's|our monthly|the monthly|this month's|the|a|an|our|this)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    # Capitalize first letter cleanly if lowercased
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned


def _resolve_text_column(dataset: Dataset | DatasetDict, preferred_col: str) -> str:
    """Find the text column in a Dataset or DatasetDict."""
    if isinstance(dataset, DatasetDict):
        sample_split = next(iter(dataset.keys()))
        columns = dataset[sample_split].column_names
    elif isinstance(dataset, Dataset):
        columns = dataset.column_names
    elif pd is not None and isinstance(dataset, pd.DataFrame):
        columns = list(dataset.columns)
    else:
        columns = getattr(dataset, "column_names", [])

    if preferred_col in columns:
        return preferred_col

    for candidate in CANDIDATE_TEXT_COLUMNS:
        if candidate in columns:
            return candidate

    raise ValueError(
        f"Text column '{preferred_col}' not found. Available columns: {columns}. "
        f"Specify the column name via text_col."
    )


def extract_line_info(
    text: str,
    qa_pipeline: Pipeline | None = None,
    model_name: str = DEFAULT_MODEL,
    event_question: str = DEFAULT_EVENT_QUESTION,
    participants_question: str = DEFAULT_PARTICIPANTS_QUESTION,
    min_score: float = 0.05,
    normalize_participants: bool = True,
    clean_event: bool = True,
    device: str | int | None = None,
) -> dict[str, str]:
    """Extract event type and participants from a single line of text.

    Args:
        text: String of text to analyze.
        qa_pipeline: Optional pre-loaded QA pipeline.
        model_name: HF model name if pipeline needs to be loaded.
        event_question: Question string for event type extraction.
        participants_question: Question string for participant count extraction.
        min_score: Minimum confidence score to accept an extracted answer.
        normalize_participants: Whether to parse participant text to an integer string.
        clean_event: Whether to strip filler words and punctuation from event type.
        device: Device ID (0 for CUDA, -1 for CPU).

    Returns:
        Dict with keys 'event_type' and 'participants'.
    """
    if not text or not str(text).strip():
        return {"event_type": "", "participants": ""}

    cleaned_text = str(text).strip()
    if qa_pipeline is None:
        qa_pipeline = load_qa_pipeline(model_name=model_name, device=device)

    # Query QA model for event type
    event_res = qa_pipeline(question=event_question, context=cleaned_text)
    event_type = ""
    if event_res and event_res.get("score", 0.0) >= min_score:
        raw_event = event_res.get("answer", "")
        event_type = clean_event_label(raw_event) if clean_event else raw_event.strip()

    # Query QA model for participant count
    part_res = qa_pipeline(question=participants_question, context=cleaned_text)
    participants = ""
    if part_res and part_res.get("score", 0.0) >= min_score:
        raw_part = part_res.get("answer", "")
        participants = parse_participant_count(raw_part) if normalize_participants else raw_part.strip()

    return {"event_type": event_type, "participants": participants}


def extract_dataset_info(
    dataset: Dataset | DatasetDict,
    model_name: str = DEFAULT_MODEL,
    text_col: str = "text",
    event_type_col: str = "event_type",
    participants_col: str = "participants",
    event_question: str = DEFAULT_EVENT_QUESTION,
    participants_question: str = DEFAULT_PARTICIPANTS_QUESTION,
    min_score: float = 0.05,
    normalize_participants: bool = True,
    clean_event_type: bool = True,
    use_relevance: bool = True,
    relevance_col: str = "relevance",
    batch_size: int = 16,
    qa_pipeline: Pipeline | None = None,
    device: str | int | None = None,
) -> Dataset | DatasetDict:
    """Extract event type and participant counts from dataset text lines using a QA model.

    Its input is a dataset, and it outputs the dataset with additional columns for
    event type and participant count.

    Args:
        dataset: A datasets.Dataset, datasets.DatasetDict, or pandas DataFrame.
        model_name: Hugging Face QA model identifier (default: 'deepset/roberta-base-squad2').
        text_col: Name of column containing the text lines.
        event_type_col: Additional column name for extracted event type (default: 'event_type').
        participants_col: Additional column name for extracted participant count (default: 'participants').
        event_question: Question to ask for extracting the event type.
        participants_question: Question to ask for extracting the participant count.
        min_score: Minimum confidence threshold to accept an extracted answer.
        normalize_participants: If True, normalizes numbers (e.g. '42 women' -> '42').
        clean_event_type: If True, strips leading articles and punctuation from event types.
        use_relevance: If True and relevance column is present, skips chit-chat / irrelevant lines.
        relevance_col: Name of the relevance column to check if present.
        batch_size: Number of QA queries to process per inference batch.
        qa_pipeline: Optional pre-loaded pipeline to reuse.
        device: Device ID (0 for CUDA, -1 for CPU, or 'cuda'/'cpu').

    Returns:
        The dataset with additional columns containing the line's event type and participants.
    """
    # Gracefully support pandas DataFrame
    is_df = False
    if pd is not None and isinstance(dataset, pd.DataFrame):
        is_df = True
        df_copy = dataset.copy()
        dataset = Dataset.from_pandas(df_copy)

    resolved_text_col = _resolve_text_column(dataset, text_col)

    # Check if relevance column exists
    sample_cols = []
    if isinstance(dataset, DatasetDict):
        first_split = next(iter(dataset.keys()))
        sample_cols = dataset[first_split].column_names
    elif isinstance(dataset, Dataset):
        sample_cols = dataset.column_names
    has_relevance = use_relevance and (relevance_col in sample_cols)

    # Load QA pipeline if not passed
    if qa_pipeline is None:
        qa_pipeline = load_qa_pipeline(model_name=model_name, device=device)

    irrelevant_keywords = {"chit-chat", "chitchat", "banter", "small talk", "irrelevant", "unimportant"}

    def _is_irrelevant_line(val: str) -> bool:
        low = val.lower().strip()
        if low in ("chit-chat", "chitchat", "irrelevant"):
            return True
        return any(k in low for k in irrelevant_keywords)

    def _batch_extract(batch):
        raw_texts = batch[resolved_text_col]
        texts = ["" if t is None else str(t) for t in raw_texts]
        n = len(texts)

        out_event_types = [""] * n
        out_participants = [""] * n

        # Check relevance tags if present
        relevance_tags = batch.get(relevance_col, [None] * n) if has_relevance else [None] * n

        # Find rows eligible for QA extraction
        active_indices = []
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            if has_relevance and relevance_tags[i] is not None:
                if _is_irrelevant_line(str(relevance_tags[i])):
                    continue
            active_indices.append(i)

        if not active_indices:
            res = {event_type_col: out_event_types, participants_col: out_participants}
            return res

        # Prepare paired QA inputs: for each active row, ask event question and participants question
        qa_inputs = []
        for idx in active_indices:
            ctx = texts[idx].strip()
            qa_inputs.append({"question": event_question, "context": ctx})
            qa_inputs.append({"question": participants_question, "context": ctx})

        # Run QA inference
        qa_results = qa_pipeline(qa_inputs, batch_size=min(len(qa_inputs), batch_size))
        if isinstance(qa_results, dict):
            qa_results = [qa_results]

        # Parse responses
        for k, idx in enumerate(active_indices):
            event_res = qa_results[2 * k]
            part_res = qa_results[2 * k + 1]

            if event_res and event_res.get("score", 0.0) >= min_score:
                ans = event_res.get("answer", "")
                out_event_types[idx] = clean_event_label(ans) if clean_event_type else ans.strip()

            if part_res and part_res.get("score", 0.0) >= min_score:
                ans = part_res.get("answer", "")
                out_participants[idx] = parse_participant_count(ans) if normalize_participants else ans.strip()

        return {
            event_type_col: out_event_types,
            participants_col: out_participants,
        }

    processed = dataset.map(
        _batch_extract,
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=False,
    )

    if is_df:
        return processed.to_pandas()
    return processed


# Alias matching project naming convention
process_dataset_extraction = extract_dataset_info
extract_event_info = extract_dataset_info


def _load_data_file(path: str) -> DatasetDict:
    """Load a local tabular file (.csv, .tsv, .json, .jsonl, .xlsx, .parquet) as a DatasetDict."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return DatasetDict({"train": Dataset.from_pandas(pd.read_csv(path, encoding_errors="replace"))})
    elif ext == ".tsv":
        return DatasetDict({"train": Dataset.from_pandas(pd.read_csv(path, sep="\t", encoding_errors="replace"))})
    elif ext in (".json", ".jsonl"):
        return DatasetDict({"train": Dataset.from_pandas(pd.read_json(path, lines=(ext == ".jsonl")))})
    elif ext in (".xlsx", ".xls"):
        return DatasetDict({"train": Dataset.from_pandas(pd.read_excel(path))})
    elif ext == ".parquet":
        return DatasetDict({"train": Dataset.from_pandas(pd.read_parquet(path))})
    else:
        raise ValueError(f"Unsupported file format '{ext}' for file: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract event type and participant counts from text lines using a question answering model."
    )
    parser.add_argument("-d", "--dataset", required=True, help="Hugging Face dataset name or local file (.csv, .json, .jsonl, .xlsx, .parquet)")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"QA model (default: {DEFAULT_MODEL})")
    parser.add_argument("--text_col", default="text", help="Column containing text lines (default: 'text')")
    parser.add_argument("--event_type_col", default="event_type", help="Output column name for event type (default: 'event_type')")
    parser.add_argument("--participants_col", default="participants", help="Output column name for participants (default: 'participants')")
    parser.add_argument("--event_question", default=DEFAULT_EVENT_QUESTION, help="Custom QA question for event type")
    parser.add_argument("--participants_question", default=DEFAULT_PARTICIPANTS_QUESTION, help="Custom QA question for participants")
    parser.add_argument("--min_score", type=float, default=0.05, help="Minimum confidence threshold for QA answers (default: 0.05)")
    parser.add_argument("--no_normalize_participants", action="store_true", help="Do not convert participant phrases to numeric counts")
    parser.add_argument("--relevance_col", default="relevance", help="Relevance column name if dataset is already classified")
    parser.add_argument("--ignore_relevance", action="store_true", help="Run extraction on all lines even if relevance column is present")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Inference batch size (default: 16)")
    parser.add_argument("-n", "--num_samples", type=int, default=0, help="Process first N samples (0 = all)")
    parser.add_argument("-o", "--output", help="Path to save output (.csv, .json, .jsonl, .parquet)")

    args = parser.parse_args()

    # Load dataset
    if os.path.isfile(args.dataset):
        dataset = _load_data_file(args.dataset)
    else:
        from datasets import load_dataset
        dataset = load_dataset(args.dataset)

    # Slice if num_samples specified
    if args.num_samples > 0:
        if isinstance(dataset, DatasetDict):
            for k in dataset.keys():
                dataset[k] = dataset[k].select(range(min(args.num_samples, len(dataset[k]))))
        else:
            dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    print(f"Extracting structured info with QA model: {args.model}")
    print(f"Event question: {args.event_question}")
    print(f"Participants question: {args.participants_question}")

    extracted = extract_dataset_info(
        dataset,
        model_name=args.model,
        text_col=args.text_col,
        event_type_col=args.event_type_col,
        participants_col=args.participants_col,
        event_question=args.event_question,
        participants_question=args.participants_question,
        min_score=args.min_score,
        normalize_participants=not args.no_normalize_participants,
        use_relevance=not args.ignore_relevance,
        relevance_col=args.relevance_col,
        batch_size=args.batch_size,
    )

    # Show preview
    preview_split = next(iter(extracted.keys())) if isinstance(extracted, DatasetDict) else None
    sample_view = extracted[preview_split] if preview_split else extracted
    print(f"\nCompleted extraction. Sample output ({min(5, len(sample_view))} rows):")
    for i in range(min(5, len(sample_view))):
        row = sample_view[i]
        line_snippet = str(row.get(args.text_col, ""))[:50]
        event = row.get(args.event_type_col, "")
        part = row.get(args.participants_col, "")
        print(f"  [Event: {event!r:30} | Participants: {part!r:6}] {line_snippet}...")

    # Save output if path provided
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        ext = os.path.splitext(args.output)[1].lower()
        export_ds = sample_view
        if ext == ".csv":
            export_ds.to_csv(args.output, index=False)
        elif ext in (".json", ".jsonl"):
            export_ds.to_json(args.output, orient="records", lines=(ext == ".jsonl"))
        elif ext == ".parquet":
            export_ds.to_parquet(args.output)
        else:
            export_ds.to_csv(args.output, index=False)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
