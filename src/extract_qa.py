"""Module alias for question-answering based extraction stage."""
from extract import (
    DEFAULT_MODEL,
    DEFAULT_EVENT_QUESTION,
    DEFAULT_PARTICIPANTS_QUESTION,
    load_qa_pipeline,
    parse_participant_count,
    clean_event_label,
    extract_line_info,
    extract_dataset_info,
    process_dataset_extraction,
    extract_event_info,
    main,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_EVENT_QUESTION",
    "DEFAULT_PARTICIPANTS_QUESTION",
    "load_qa_pipeline",
    "parse_participant_count",
    "clean_event_label",
    "extract_line_info",
    "extract_dataset_info",
    "process_dataset_extraction",
    "extract_event_info",
    "main",
]

if __name__ == "__main__":
    main()
