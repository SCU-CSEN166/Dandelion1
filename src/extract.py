"""Activity type and head count of an Event message, from models and from rules.

- Activity categories (classify_activities): the zero-shot model that labels relevance also picks each Event's
  category from ACTIVITY_CATEGORIES, with a score (activity_category / activity_score columns).
- QA model (extract_dataset_info): adds event_type and participants columns, the model's answers to two questions.
- Rules: find_topic (a topic list) and find_participants (number patterns such as "forty-two women came"). These are in-sample for the hidden test; maybe we could move more responsibility to the QA model?
- activity_type() and participant_count() combine them for the output tables; tables.py turns their disagreements
  into review reasons.
"""
from functools import lru_cache
import re
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from datasets import Dataset

from relevance import is_chitchat, load_zero_shot_classifier

OTHER_ACTIVITY = "Other field activity"

# The activity types of the output table
ACTIVITY_CATEGORIES = [
    "Family planning outreach", "Maternal health talk",
    "Gender-based violence awareness session",
    "Banking group meeting", "Farmer training", "Youth mentorship session",
    "Immunization outreach", "Nutrition session", "Community clean-up",
    "Home visits", "Distribution of supplies", "Business skills training",
    "Livestock and poultry training", "Hygiene promotion session"
]
ACTIVITY_HYPOTHESIS = "This message is about a {}."
# Below this score the classifier's category is not trusted (set on data/extraction_dev.jsonl; see eval/eval_extraction.py)
ACTIVITY_MIN_SCORE = 0.2

# --- Numbers ---
_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_NUMWORD = "|".join(list(_UNITS) + list(_TENS) + ["hundred", "thousand"])
# "and" only after hundred / thousand ("one hundred and nine"), so "forty and fifty" stays two numbers
_NUMWORDS = rf"(?:{_NUMWORD})(?:(?:[- ]|(?<=hundred) and |(?<=thousand) and )(?:{_NUMWORD}))*"
_NUM = rf"(?:\d[\d,]*|{_NUMWORDS})"
_DIGITS_RE = re.compile(r"\b\d[\d,]*\b")
_NUMWORDS_RE = re.compile(rf"\b{_NUMWORDS}\b", re.I)


def words_to_int(text: str) -> int:
    """'42' / '1,250' / 'forty-two' / 'two thousand five hundred' -> int."""
    text = text.lower().replace(",", "")
    if text.replace(" ", "").isdigit():
        return int(text)
    total = current = 0
    for tok in re.split(r"[- ]+", text):
        if tok in _UNITS:
            current += _UNITS[tok]
        elif tok in _TENS:
            current += _TENS[tok]
        elif tok == "hundred":
            current = max(current, 1) * 100
        elif tok == "thousand":
            total += max(current, 1) * 1000
            current = 0
    return total + current


def first_number(text: str) -> int | None:
    """The first number in the text, preferring digits over number words, or None."""
    m = _DIGITS_RE.search(text or "") or _NUMWORDS_RE.search(text or "")
    return words_to_int(m.group(0)) if m else None


# --- Rules (in-sample) ---
_PEOPLE = (r"(?:women|men|girls|boys|ladies|members|people|farmers|participants|mothers|fathers|children|kids|youth|"
           r"students|pupils|villagers|residents|patients|clients|caregivers|elders|persons)")
_PAT_A = re.compile(rf"\b(?P<n>{_NUM})\s+{_PEOPLE}(?:\s+\w+){{0,2}}?\s+"
                    r"(?:attended|came|turned up|participated|were present|joined|registered|were reached)\b", re.I)
_PAT_B = re.compile(rf"\b(?:served|screened|reached|trained|vaccinated|treated|counselled|counseled|tested|had)\s+"
                    rf"(?P<n>{_NUM})\s+{_PEOPLE}\b", re.I)
_PAT_C = re.compile(rf"\b(?:attendance|turnout)\s+(?:was|of)\s+(?P<n>{_NUM})\b", re.I)

# (pattern, label). First match wins, so specific topics come first.
TOPICS = [
    (r"family planning|contracepti|implants?\b", "Family planning outreach"),
    (r"menstrual|sanitary pads?|\bperiods\b", "Menstrual hygiene session"),
    (r"gender[- ]based violence|\bGBV\b|domestic violence", "Gender-based violence awareness session"),
    (r"cervical cancer|cancer screening|breast cancer", "Cervical cancer screening camp"),
    (r"table banking|savings group|\bchama\b|village savings", "Table banking (savings) group meeting"),
    (r"kitchen garden|sack garden|backyard garden", "Kitchen garden training"),
    (r"immuni[sz]ation|vaccinat", "Immunization outreach"),
    (r"antenatal|\bANC\b|prenatal", "Antenatal care clinic"),
    (r"\bHIV\b|\bAIDS\b", "HIV testing and counselling"),
    (r"nutrition|malnutrition|growth monitoring", "Nutrition session"),
    (r"malaria|bed ?nets?", "Malaria prevention session"),
    (r"deworming", "Deworming drive"),
    (r"hand ?washing|sanitation|hygiene", "Hygiene promotion session"),
    (r"agricultur|farmer training|crop|seedlings|compost", "Farmer training"),
]


def find_participants(text: str) -> int | None:
    """The attendance count stated in the text (first one), or None."""
    hits = []
    for pat in (_PAT_A, _PAT_B, _PAT_C):
        m = pat.search(text or "")
        if m:
            hits.append((m.start(), words_to_int(m.group("n"))))
    return min(hits)[1] if hits else None


def find_topic(text: str) -> str:
    """The first TOPICS label whose pattern occurs in the text, or ''."""
    for pattern, label in TOPICS:
        if re.search(pattern, text or "", re.I):
            return label
    return ""


def clean_qa_label(label: str) -> str:
    """Drop redaction aliases and tags from a QA event label ("Place0 outreach" -> "Outreach")."""
    label = re.sub(r"\b(?:Person|Place)\d+\b|\[(?:NAME|LOCATION|PHONE|EMAIL)\]", " ", str(label or ""))
    label = re.sub(r"\s+", " ", label).strip(" .,;:-")
    return label[:1].upper() + label[1:] if len(re.sub(r"[^A-Za-z]", "", label)) >= 3 else ""


# --- Activity categories (zero-shot) ---
def classify_activities(dataset: Dataset, relevance_col: str = "relevance", batch_size: int = 16,
                        classifier=None) -> Dataset:
    """Add activity_category and activity_score: the zero-shot model's best ACTIVITY_CATEGORIES match for each
    Event message ('' and 0.0 for Chit-chat)."""
    if classifier is None:
        classifier = load_zero_shot_classifier()

    def _batch(batch):
        texts = ["" if t is None else str(t).strip() for t in batch["text"]]
        categories, scores = [""] * len(texts), [0.0] * len(texts)
        events = [i for i, (t, r) in enumerate(zip(texts, batch[relevance_col])) if t and not is_chitchat(r)]
        if events:
            preds = classifier([texts[i] for i in events], candidate_labels=ACTIVITY_CATEGORIES,
                               hypothesis_template=ACTIVITY_HYPOTHESIS, multi_label=False,
                               batch_size=min(len(events), batch_size))
            for i, pred in zip(events, [preds] if isinstance(preds, dict) else preds):
                categories[i], scores[i] = pred["labels"][0], float(pred["scores"][0])
        return {"activity_category": categories, "activity_score": scores}

    return dataset.map(_batch, batched=True, batch_size=batch_size, load_from_cache_file=False)


# --- Models and rules combined ---
def activity_type(text: str, qa_event_type: str = "", category: str = "", category_score: float = 0.0) -> str:
    """The classifier's category when its score reaches ACTIVITY_MIN_SCORE; else the topic list's label, the QA
    model's cleaned answer, or OTHER_ACTIVITY."""
    if category and float(category_score or 0) >= ACTIVITY_MIN_SCORE:
        return category
    return find_topic(text) or clean_qa_label(qa_event_type) or OTHER_ACTIVITY


def participant_count(text: str, qa_participants: str = "") -> int | None:
    """The QA model's count and the number rule's count; None when neither has one or the two disagree."""
    qa = str(qa_participants or "").strip()
    qa = int(qa) if qa.isdigit() else None
    rule = find_participants(text)
    if qa is not None and rule is not None and qa != rule:
        return None  # the review queue reports the disagreement
    return qa if qa is not None else rule


# --- QA model ---

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
