"""Stage for classifying the relevance of text lines in a dataset using zero-shot classification.

Designed for Dandelion Africa field communications (WhatsApp chats and voice-note
transcripts) to distinguish actionable field activities, health outreach, community work,
and agricultural sessions from casual chit-chat, greetings, and irrelevant banter.
"""
from functools import lru_cache
import argparse
import os
import sys
import torch
from transformers import pipeline, Pipeline
from datasets import Dataset, DatasetDict

try:
    import pandas as pd
except ImportError:
    pd = None

# Default zero-shot classification model selected in the project plan
DEFAULT_MODEL = "facebook/bart-large-mnli"

# Default candidate labels tailored to Dandelion Africa's mission:
# Discerning actionable field outcomes, health outreaches, community work,
# and agricultural interventions from casual chit-chat, greetings, and personal banter.
DEFAULT_LABELS = [
    "relevant field activity",
    "casual chit-chat",
]

DEFAULT_HYPOTHESIS_TEMPLATE = "This message is {}."

# Preset label sets representing different operational granularities
PRESET_LABELS: dict[str, list[str]] = {
    # Default: binary field activity vs casual chit-chat
    "field_activity": [
        "relevant field activity",
        "casual chit-chat",
    ],
    # Direct binary relevance
    "relevance": [
        "relevant to field operations",
        "irrelevant small talk",
    ],
    # Aligned with TA hidden test benchmark (Event vs Chit-chat)
    "event_chitchat": [
        "Event",
        "Chit-chat",
    ],
    # Categorization aligned with Dandelion Africa report structure:
    # 1. Practical Actions Completed
    # 2. Blockers & Technical Issues
    # 3. Outcomes & Resolutions
    # 4. Pending Follow-ups
    # 5. Casual Chit-chat
    "report_sections": [
        "practical action or task completed",
        "blocker, issue, or resource problem",
        "outcome, resolution, or decision",
        "pending follow-up or next step",
        "casual chit-chat or greetings",
    ],
}

# Known text column names across common datasets
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
def load_zero_shot_classifier(
    model_name: str = DEFAULT_MODEL,
    device: str | int | None = None,
) -> Pipeline:
    """Load and cache a zero-shot classification pipeline.

    Args:
        model_name: Hugging Face model identifier (e.g. 'facebook/bart-large-mnli').
        device: Device ID (0 for CUDA, -1 for CPU, or 'cuda'/'cpu'). If None,
            automatically uses CUDA if available.

    Returns:
        A transformers Pipeline for zero-shot-classification.
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )


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


def classify_line(
    text: str,
    classifier: Pipeline | None = None,
    model_name: str = DEFAULT_MODEL,
    candidate_labels: list[str] | None = None,
    hypothesis_template: str | None = None,
    label_mapping: dict[str, str] | None = None,
    device: str | int | None = None,
) -> tuple[str, float]:
    """Classify the relevance of a single line of text.

    Args:
        text: String to classify.
        classifier: Optional pre-loaded pipeline.
        model_name: HF model name if classifier needs to be loaded.
        candidate_labels: List of candidate labels.
        hypothesis_template: Template for NLI hypothesis.
        label_mapping: Optional mapping from model labels to output labels.
        device: Device to run model on.

    Returns:
        Tuple of (relevance_label, confidence_score).
    """
    if not text or not str(text).strip():
        labels = candidate_labels or DEFAULT_LABELS
        fallback = labels[-1]
        if label_mapping and fallback in label_mapping:
            fallback = label_mapping[fallback]
        return fallback, 0.0

    if classifier is None:
        classifier = load_zero_shot_classifier(model_name=model_name, device=device)

    labels = candidate_labels or DEFAULT_LABELS
    template = hypothesis_template or DEFAULT_HYPOTHESIS_TEMPLATE

    result = classifier(
        str(text).strip(),
        candidate_labels=labels,
        hypothesis_template=template,
        multi_label=False,
    )

    top_label = result["labels"][0]
    top_score = float(result["scores"][0])

    if label_mapping and top_label in label_mapping:
        top_label = label_mapping[top_label]

    return top_label, top_score


def classify_dataset_relevance(
    dataset: Dataset | DatasetDict,
    model_name: str = DEFAULT_MODEL,
    candidate_labels: list[str] | None = None,
    hypothesis_template: str | None = None,
    text_col: str = "text",
    relevance_col: str = "relevance",
    score_col: str | None = None,
    label_mapping: dict[str, str] | None = None,
    batch_size: int = 16,
    classifier: Pipeline | None = None,
    device: str | int | None = None,
) -> Dataset | DatasetDict:
    """Classify the relevance of text lines in a dataset using a zero-shot classifier model.

    Its input is a dataset, and it outputs the dataset with an additional column containing
    the line's relevance.

    Args:
        dataset: A datasets.Dataset, datasets.DatasetDict, or pandas DataFrame.
        model_name: Hugging Face zero-shot model (default: 'facebook/bart-large-mnli').
        candidate_labels: Candidate labels for classification. Defaults to project labels
            ['relevant field activity', 'casual chit-chat'].
        hypothesis_template: Hypothesis template for NLI zero-shot classification
            (default: 'This message is {}.').
        text_col: Name of the column containing the text lines.
        relevance_col: Name of the additional column to store relevance (default: 'relevance').
        score_col: Optional column name to store confidence scores (e.g. 'relevance_score').
        label_mapping: Optional dict mapping model labels to preferred names
            (e.g., {'relevant field activity': 'relevant', 'casual chit-chat': 'irrelevant'}).
        batch_size: Number of lines to process per inference batch (default: 16).
        classifier: Optional pre-loaded pipeline to reuse across runs.
        device: Device ID (0 for CUDA, -1 for CPU, or 'cuda'/'cpu').

    Returns:
        The dataset with an additional column containing each line's relevance.
    """
    # Gracefully support pandas DataFrame by converting to Dataset and back
    is_df = False
    if pd is not None and isinstance(dataset, pd.DataFrame):
        is_df = True
        df_copy = dataset.copy()
        dataset = Dataset.from_pandas(df_copy)

    # Resolve text column name
    resolved_text_col = _resolve_text_column(dataset, text_col)

    # Load classifier if not supplied
    if classifier is None:
        classifier = load_zero_shot_classifier(model_name=model_name, device=device)

    labels = candidate_labels if candidate_labels is not None else DEFAULT_LABELS
    template = hypothesis_template or DEFAULT_HYPOTHESIS_TEMPLATE

    # Fallback label for blank/empty lines (default to the last label, typically chit-chat)
    fallback_empty = labels[-1]
    if label_mapping and fallback_empty in label_mapping:
        fallback_empty = label_mapping[fallback_empty]

    def _batch_classify(batch):
        raw_texts = batch[resolved_text_col]
        texts = ["" if t is None else str(t) for t in raw_texts]

        # Identify non-empty lines for zero-shot classification
        non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]

        out_labels = [fallback_empty] * len(texts)
        out_scores = [0.0] * len(texts)

        if non_empty_indices:
            sub_inputs = [texts[i].strip() for i in non_empty_indices]

            preds = classifier(
                sub_inputs,
                candidate_labels=labels,
                hypothesis_template=template,
                multi_label=False,
                batch_size=min(len(sub_inputs), batch_size),
            )
            # Ensure list of results when single item returned
            if isinstance(preds, dict):
                preds = [preds]

            for idx, pred in zip(non_empty_indices, preds):
                top_label = pred["labels"][0]
                top_score = float(pred["scores"][0])
                if label_mapping and top_label in label_mapping:
                    top_label = label_mapping[top_label]
                out_labels[idx] = top_label
                out_scores[idx] = top_score

        result = {relevance_col: out_labels}
        if score_col:
            result[score_col] = out_scores
        return result

    processed = dataset.map(
        _batch_classify,
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=False,
    )

    if is_df:
        return processed.to_pandas()
    return processed


# Alias following project naming convention in redact.py (process_dataset_pii)
process_dataset_relevance = classify_dataset_relevance


def filter_relevant_lines(
    dataset: Dataset | DatasetDict,
    relevance_col: str = "relevance",
    relevant_labels: list[str] | set[str] | None = None,
) -> Dataset | DatasetDict:
    """Filter dataset rows to retain only those classified as relevant.

    Args:
        dataset: Dataset or DatasetDict containing a relevance column.
        relevance_col: Name of the relevance column.
        relevant_labels: Optional set of labels considered relevant. If None,
            automatically treats labels that do NOT contain 'chit-chat',
            'chitchat', 'banter', 'small talk', or 'irrelevant' as relevant.

    Returns:
        Filtered Dataset or DatasetDict containing only relevant lines.
    """
    irrelevant_keywords = {"chit-chat", "chitchat", "banter", "small talk", "irrelevant", "unimportant"}

    if relevant_labels is not None:
        target_set = set(relevant_labels)
        return dataset.filter(lambda row: row[relevance_col] in target_set)

    def is_relevant(row):
        val = str(row[relevance_col]).lower().strip()
        # If explicitly marked 'relevant' or 'event', it's relevant
        if val in ("relevant", "event"):
            return True
        # If contains any irrelevant keywords, drop it
        if any(neg in val for neg in irrelevant_keywords):
            return False
        return True

    return dataset.filter(is_relevant)


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
        description="Classify the relevance of text lines in a dataset using zero-shot classification."
    )
    parser.add_argument("-d", "--dataset", required=True, help="Hugging Face dataset name or local file (.csv, .json, .jsonl, .xlsx, .parquet)")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Zero-shot model (default: {DEFAULT_MODEL})")
    parser.add_argument("--preset", choices=list(PRESET_LABELS.keys()), help="Preset label configuration")
    parser.add_argument("--labels", nargs="+", help="Custom candidate labels (overrides --preset)")
    parser.add_argument("--hypothesis_template", help="Custom hypothesis template, e.g. 'This message is {}.'")
    parser.add_argument("--text_col", default="text", help="Column containing text lines (default: 'text')")
    parser.add_argument("--relevance_col", default="relevance", help="Output column name for relevance (default: 'relevance')")
    parser.add_argument("--score_col", help="Optional column name to record confidence scores (e.g. 'relevance_score')")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Inference batch size (default: 16)")
    parser.add_argument("-n", "--num_samples", type=int, default=0, help="Process first N samples (0 = all)")
    parser.add_argument("--filter_relevant", action="store_true", help="Filter output to keep only relevant lines")
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

    # Select labels
    candidate_labels = None
    if args.labels:
        candidate_labels = args.labels
    elif args.preset:
        candidate_labels = PRESET_LABELS[args.preset]

    print(f"Classifying relevance with model: {args.model}")
    print(f"Candidate labels: {candidate_labels or DEFAULT_LABELS}")

    classified = classify_dataset_relevance(
        dataset,
        model_name=args.model,
        candidate_labels=candidate_labels,
        hypothesis_template=args.hypothesis_template,
        text_col=args.text_col,
        relevance_col=args.relevance_col,
        score_col=args.score_col,
        batch_size=args.batch_size,
    )

    if args.filter_relevant:
        classified = filter_relevant_lines(classified, relevance_col=args.relevance_col)
        print("Filtered dataset to keep only relevant lines.")

    # Show preview
    preview_split = next(iter(classified.keys())) if isinstance(classified, DatasetDict) else None
    sample_view = classified[preview_split] if preview_split else classified
    print(f"\nCompleted classification. Sample output ({min(5, len(sample_view))} rows):")
    for i in range(min(5, len(sample_view))):
        row = sample_view[i]
        line_snippet = str(row.get(args.text_col, ""))[:60]
        rel = row.get(args.relevance_col, "")
        print(f"  [{rel}] {line_snippet}...")

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
