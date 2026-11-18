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
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from datasets import Dataset

from relevance import is_chitchat
from ta_table import first_number

# Default QA model: SQuAD 2.0-trained model handles unanswerable questions accurately
DEFAULT_MODEL = "deepset/roberta-base-squad2"

DEFAULT_EVENT_QUESTION = "What event, activity, or session is being held or discussed?"
DEFAULT_PARTICIPANTS_QUESTION = "How many people or participants attended, came, or were served?"

_NO_ANSWER = {"answer": "", "score": 0.0, "start": 0, "end": 0}

# Numbers in a QA answer that are not head counts: years, clock times, money, and "one of ..." (one person of a group)
_NOT_A_COUNT_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b\d{1,2}[:.]\d{2}\b|\b\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.|o'?clock)(?![a-z])"
    r"|\b(?:ksh|kes|shillings?|bob|usd|dollars?)\b|\$|\bone of\b",
    re.IGNORECASE,
)


class QAPipeline:
    """Extractive question answering (transformers 5 no longer has a "question-answering" pipeline).

    Called like the old pipeline: with one {"question", "context"} dict, or a list of them.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | int | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
            self.device = torch.device(f"cuda:{device}" if device >= 0 and torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def _predict_one(self, question: str, context: str) -> dict:
        context = str(context or "").strip()
        if not context:
            return dict(_NO_ANSWER)

        inputs = self.tokenizer(
            question,
            context,
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

        ctx_indices = [i for i, s in enumerate(seq_ids) if s == 1]
        if not ctx_indices:
            return dict(_NO_ANSWER)

        # Only context tokens can be an answer, plus the first token (CLS), where SQuAD 2.0 models point both ends
        # to say "no answer" (as transformers' own pipeline did with handle_impossible_answer)
        not_answer = torch.tensor([s != 1 for s in seq_ids])
        not_answer[0] = False
        start_probs = torch.softmax(outputs.start_logits[0].cpu().masked_fill(not_answer, -1e4), dim=-1)
        end_probs = torch.softmax(outputs.end_logits[0].cpu().masked_fill(not_answer, -1e4), dim=-1)
        null_score = (start_probs[0] * end_probs[0]).item()

        # Best answer span of at most 30 tokens inside the context
        best_score, best_s, best_e = 0.0, 0, 0
        for s in ctx_indices:
            for e in range(s, min(s + 30, ctx_indices[-1] + 1)):
                if seq_ids[e] != 1:
                    break
                prob = (start_probs[s] * end_probs[e]).item()
                if prob > best_score:
                    best_score, best_s, best_e = prob, s, e

        if best_score <= null_score:  # the model's answer is "not in the text"
            return dict(_NO_ANSWER)
        start_char = int(offsets[best_s][0])
        end_char = int(offsets[best_e][1])
        return {"answer": context[start_char:end_char].strip(), "score": float(best_score),
                "start": start_char, "end": end_char}

    def __call__(self, inputs, batch_size=None, **kwargs):
        single = isinstance(inputs, dict)
        items = [inputs] if single else inputs
        results = [self._predict_one(item.get("question", ""), item.get("context", "")) for item in items]
        return results[0] if single else results


@lru_cache(maxsize=2)
def load_qa_pipeline(model_name: str = DEFAULT_MODEL, device: str | int | None = None) -> QAPipeline:
    """Load and cache a question-answering model (device: 0 for CUDA, -1 for CPU, or 'cuda'/'cpu')."""
    return QAPipeline(model_name=model_name, device=device)


def parse_participant_count(text: str | None) -> str:
    """Normalize a QA answer to an integer string ('42 women' -> '42', 'forty-two' -> '42').

    If no number is found, returns the stripped text (e.g. 'a few elders'), or '' for empty input.
    Returns '' when the answer holds a number that is not a head count (a year, a time, money, 'one of ...').
    """
    if not text or not str(text).strip() or _NOT_A_COUNT_RE.search(str(text)):
        return ""
    number = first_number(str(text))
    return str(number) if number is not None else str(text).strip()


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


def extract_dataset_info(
    dataset: Dataset,
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
    qa_pipeline: QAPipeline | None = None,
    device: str | int | None = None,
) -> Dataset:
    """Extract event type and participant counts from dataset text lines using a QA model.

    Its input is a dataset, and it outputs the dataset with additional columns for
    event type and participant count.

    Args:
        dataset: A datasets.Dataset.
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
        qa_pipeline: Optional pre-loaded QA model to reuse.
        device: Device ID (0 for CUDA, -1 for CPU, or 'cuda'/'cpu').

    Returns:
        The dataset with additional columns containing the line's event type and participants.
    """
    has_relevance = use_relevance and relevance_col in dataset.column_names

    if qa_pipeline is None:
        qa_pipeline = load_qa_pipeline(model_name=model_name, device=device)

    def _batch_extract(batch):
        texts = ["" if t is None else str(t) for t in batch[text_col]]
        n = len(texts)
        out_event_types = [""] * n
        out_participants = [""] * n
        relevance_tags = batch[relevance_col] if has_relevance else [None] * n

        # Rows eligible for QA extraction: non-empty and not labeled chit-chat
        active_indices = [
            i for i, text in enumerate(texts)
            if text.strip() and not (relevance_tags[i] is not None and is_chitchat(relevance_tags[i]))
        ]
        if not active_indices:
            return {event_type_col: out_event_types, participants_col: out_participants}

        # For each active row, ask the event question and the participants question
        qa_inputs = []
        for idx in active_indices:
            ctx = texts[idx].strip()
            qa_inputs.append({"question": event_question, "context": ctx})
            qa_inputs.append({"question": participants_question, "context": ctx})
        qa_results = qa_pipeline(qa_inputs, batch_size=min(len(qa_inputs), batch_size))

        for k, idx in enumerate(active_indices):
            event_res, part_res = qa_results[2 * k], qa_results[2 * k + 1]

            if event_res and event_res.get("score", 0.0) >= min_score:
                ans = event_res.get("answer", "")
                out_event_types[idx] = clean_event_label(ans) if clean_event_type else ans.strip()

            if part_res and part_res.get("score", 0.0) >= min_score:
                ans = part_res.get("answer", "")
                out_participants[idx] = parse_participant_count(ans) if normalize_participants else ans.strip()

        return {event_type_col: out_event_types, participants_col: out_participants}

    return dataset.map(
        _batch_extract,
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=False,
    )
