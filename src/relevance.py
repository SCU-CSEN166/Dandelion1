"""Stage for classifying the relevance of text lines in a dataset using zero-shot classification.

Designed for Dandelion Africa field communications (WhatsApp chats and voice-note
transcripts) to distinguish actionable field activities, health outreach, community work,
and agricultural sessions from casual chit-chat, greetings, and irrelevant banter.
"""
from functools import lru_cache
import torch
from transformers import pipeline, Pipeline
from datasets import Dataset, DatasetDict

# Default zero-shot classification model selected in the project plan
DEFAULT_MODEL = "facebook/bart-large-mnli"

# Binary labels: actionable field outcomes vs casual chit-chat, greetings, and personal banter
DEFAULT_LABELS = [
    "relevant field activity",
    "casual chit-chat",
]

DEFAULT_HYPOTHESIS_TEMPLATE = "This message is {}."

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
        columns = dataset[next(iter(dataset.keys()))].column_names
    else:
        columns = dataset.column_names

    if preferred_col in columns:
        return preferred_col

    for candidate in CANDIDATE_TEXT_COLUMNS:
        if candidate in columns:
            return candidate

    raise ValueError(
        f"Text column '{preferred_col}' not found. Available columns: {columns}. "
        f"Specify the column name via text_col."
    )


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
        dataset: A datasets.Dataset or datasets.DatasetDict.
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
    resolved_text_col = _resolve_text_column(dataset, text_col)

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

    return dataset.map(
        _batch_classify,
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=False,
    )


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
