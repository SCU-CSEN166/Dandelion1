from datasets import load_dataset
from redact import process_dataset_pii
from relevance import classify_dataset_relevance, filter_relevant_lines
from extract import extract_dataset_info

def prepare_dataset(
    dataset_name: str,
    classify_relevance: bool = False,
    filter_relevant: bool = False,
    classifier_model: str = "facebook/bart-large-mnli",
    candidate_labels: list[str] | None = None,
    extract_info: bool = False,
    qa_model: str = "deepset/roberta-base-squad2",
):
    """Load a dataset, redact PII, and optionally classify relevance or extract event info."""
    if ".csv" in dataset_name:
        raw = load_dataset("csv", data_files=dataset_name)
    else:
        raw = load_dataset(dataset_name)
    processed = process_dataset_pii(raw)

    if classify_relevance or filter_relevant:
        processed = classify_dataset_relevance(
            processed,
            model_name=classifier_model,
            candidate_labels=candidate_labels,
        )
        if filter_relevant:
            processed = filter_relevant_lines(processed)

    if extract_info:
        processed = extract_dataset_info(
            processed,
            model_name=qa_model,
        )

    return processed
