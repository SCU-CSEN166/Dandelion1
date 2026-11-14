from datasets import load_dataset
from redact import process_dataset_pii
from relevance import classify_dataset_relevance, filter_relevant_lines

def prepare_dataset(
    dataset_name: str,
    classify_relevance: bool = False,
    filter_relevant: bool = False,
    classifier_model: str = "facebook/bart-large-mnli",
    candidate_labels: list[str] | None = None,
):
    """Load a dataset, redact PII, and optionally classify or filter by relevance."""
    if ".csv" in dataset_name:
        raw = load_dataset("csv", data_files=dataset_name)
    else:
        raw = load_dataset(dataset_name)
    redacted = process_dataset_pii(raw)

    if classify_relevance or filter_relevant:
        redacted = classify_dataset_relevance(
            redacted,
            model_name=classifier_model,
            candidate_labels=candidate_labels,
        )
        if filter_relevant:
            redacted = filter_relevant_lines(redacted)

    return redacted
