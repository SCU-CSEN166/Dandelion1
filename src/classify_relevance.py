"""Module alias providing relevance classification for datasets."""
from relevance import (
    DEFAULT_LABELS,
    DEFAULT_HYPOTHESIS_TEMPLATE,
    DEFAULT_MODEL,
    PRESET_LABELS,
    load_zero_shot_classifier,
    classify_dataset_relevance,
    process_dataset_relevance,
    filter_relevant_lines,
    classify_line,
    main,
)

__all__ = [
    "DEFAULT_LABELS",
    "DEFAULT_HYPOTHESIS_TEMPLATE",
    "DEFAULT_MODEL",
    "PRESET_LABELS",
    "load_zero_shot_classifier",
    "classify_dataset_relevance",
    "process_dataset_relevance",
    "filter_relevant_lines",
    "classify_line",
    "main",
]

if __name__ == "__main__":
    main()
