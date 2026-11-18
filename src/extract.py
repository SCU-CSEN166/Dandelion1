"""Extraction stage using a question answering model to extract structured event information.

Extracts:
1. The type of event or activity being discussed (event_type).
2. The number of people participating or attended (participants).

Returns the dataset with additional columns for this extra information.
Designed for Dandelion Africa field communications (WhatsApp and voice-note transcripts).
"""
from functools import lru_cache
import re
import torch
from transformers import pipeline, Pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from datasets import Dataset, DatasetDict

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


class QAWrapperPipeline:
    """Question Answering pipeline wrapper compatible with transformers >= 5."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | int | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
            self.device = torch.device(f"cuda:{device}" if device >= 0 and torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.model.to(self.device)
        self.model.eval()

    def _predict_one(self, question: str, context: str) -> dict:
        if not context or not str(context).strip():
            return {"answer": "", "score": 0.0, "start": 0, "end": 0}

        cleaned_context = str(context).strip()
        inputs = self.tokenizer(
            question,
            cleaned_context,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        offsets = inputs.pop("offset_mapping")[0]
        seq_ids = inputs.sequence_ids(0)
        model_inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**model_inputs)

        start_logits = outputs.start_logits[0].cpu()
        end_logits = outputs.end_logits[0].cpu()

        start_probs = torch.softmax(start_logits, dim=-1)
        end_probs = torch.softmax(end_logits, dim=-1)

        ctx_indices = [i for i, s in enumerate(seq_ids) if s == 1]
        if not ctx_indices:
            return {"answer": "", "score": 0.0, "start": 0, "end": 0}

        best_score = 0.0
        best_s, best_e = 0, 0
        for s in ctx_indices:
            max_e = min(s + 30, ctx_indices[-1] + 1)
            for e in range(s, max_e):
                if seq_ids[e] != 1:
                    break
                prob = (start_probs[s] * end_probs[e]).item()
                if prob > best_score:
                    best_score = prob
                    best_s, best_e = s, e

        if best_score > 0 and best_s <= best_e:
            start_char = int(offsets[best_s][0])
            end_char = int(offsets[best_e][1])
            answer = cleaned_context[start_char:end_char].strip()
            return {"answer": answer, "score": float(best_score), "start": start_char, "end": end_char}

        return {"answer": "", "score": 0.0, "start": 0, "end": 0}

    def __call__(self, inputs, batch_size=None, **kwargs):
        single = isinstance(inputs, dict)
        items = [inputs] if single else inputs
        results = [self._predict_one(item.get("question", ""), item.get("context", "")) for item in items]
        return results[0] if single else results


@lru_cache(maxsize=2)
def load_qa_pipeline(
    model_name: str = DEFAULT_MODEL,
    device: str | int | None = None,
):
    """Load and cache a question-answering pipeline.

    Args:
        model_name: Hugging Face QA model identifier (default: 'deepset/roberta-base-squad2').
        device: Device ID (0 for CUDA, -1 for CPU, or 'cuda'/'cpu').

    Returns:
        A transformers Pipeline or QAWrapperPipeline for question-answering.
    """
    try:
        dev = (0 if torch.cuda.is_available() else -1) if device is None else device
        return pipeline(
            "question-answering",
            model=model_name,
            device=dev,
        )
    except (KeyError, Exception):
        return QAWrapperPipeline(model_name=model_name, device=device)


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


def extract_dataset_info(
    dataset: Dataset | DatasetDict,
    model_name: str = DEFAULT_MODEL,
    text_col: str = "text",
    event_type_col: str = "event_type",
    participants_col: str = "participants",
    event_question: str = DEFAULT_EVENT_QUESTION,
    participants_question: str = DEFAULT_PARTICIPANTS_QUESTION,
    min_score: float = 0.005,
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
        dataset: A datasets.Dataset or datasets.DatasetDict.
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
    resolved_text_col = _resolve_text_column(dataset, text_col)

    # Check if relevance column exists
    if isinstance(dataset, DatasetDict):
        sample_cols = dataset[next(iter(dataset.keys()))].column_names
    else:
        sample_cols = dataset.column_names
    has_relevance = use_relevance and (relevance_col in sample_cols)

    # Load QA pipeline if not passed
    if qa_pipeline is None:
        qa_pipeline = load_qa_pipeline(model_name=model_name, device=device)

    irrelevant_keywords = {"chit-chat", "chitchat", "banter", "small talk", "irrelevant", "unimportant"}

    def _is_irrelevant_line(val: str) -> bool:
        low = val.lower().strip()
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

    return dataset.map(
        _batch_extract,
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=False,
    )

