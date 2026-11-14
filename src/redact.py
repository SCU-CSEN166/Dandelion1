"""PII Redaction module for chat logs and transcripts.

Scans for and redacts:
1. Email addresses (e.g. user@domain.com) -> masked with asterisks
2. Phone numbers (standard international, US, and Kenyan formats) -> masked with asterisks
3. Sender names (e.g. 'John Doe: ', 'Grace Wanjiru: ') -> replaced with Person0, Person1, ...
4. Dialogue body mentions of known sender names -> replaced with their assigned Person alias
"""
import re
from datasets import Dataset, DatasetDict

try:
    import hyperscan
    _HS_AVAILABLE = True
except (ImportError, Exception):
    hyperscan = None
    _HS_AVAILABLE = False


# Email regex pattern
EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

# Phone regex patterns:
# - Kenyan formats: +254 7XX XXX XXX, 07XX XXX XXX, 01XX XXX XXX, 07560080089, with spaces/hyphens/compact
# - International / North American formats: +1 (123) 456-7890, (123) 456-7890, 123-456-7890
KENYAN_PHONE_PATTERN = r"(?:\+?254|0)[\s-]?[17](?:[\s-]?\d){7,9}\b"
STANDARD_PHONE_PATTERN = r"(?:\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}\b"
PHONE_PATTERN = f"(?:{KENYAN_PHONE_PATTERN}|{STANDARD_PHONE_PATTERN})"

# Sender prefix in chat transcripts (e.g. "Alice: ", "[12:00] Bob: ", "Grace Wanjiru: ")
SENDER_PATTERN = r"(?:^|(?<=\n)|(?<=\]\s))([A-Za-z0-9_.\s]+?):\s"

# Non-name headers and labels that should never be redacted as person names
EXCLUDED_HEADERS = {
    "SUBREDDIT", "TITLE", "POST", "TL;DR", "TLDR", "NOTE", "WARNING",
    "REPORT", "EVENT", "CHIT-CHAT", "HTTP", "HTTPS", "TIME", "DATE",
    "MESSAGE", "TRANSCRIPT", "DIALOGUE", "QUERY", "CONTENT", "INPUT", "REFERENCE"
}

EMAIL_RE = re.compile(EMAIL_PATTERN, re.IGNORECASE)
PHONE_RE = re.compile(PHONE_PATTERN)
SENDER_RE = re.compile(SENDER_PATTERN, re.MULTILINE)

# Optional Hyperscan database compilation for environments where Hyperscan is installed
db = None
if _HS_AVAILABLE:
    try:
        db = hyperscan.Database()
        db.compile(
            expressions=[
                EMAIL_PATTERN,
                PHONE_PATTERN,
                r"\b\S+:\s",
            ],
            flags=[hyperscan.HS_FLAG_SOM_LEFTMOST] * 3,
        )
    except Exception:
        db = None


def redact_line(text: str, namemap: dict) -> str:
    """Redact PII from a single text string."""
    if not text:
        return ""

    # 1. Mask email addresses with asterisks
    text = EMAIL_RE.sub(lambda m: "*" * len(m.group(0)), text)

    # 2. Mask phone numbers with asterisks
    text = PHONE_RE.sub(lambda m: "*" * len(m.group(0)), text)

    # 3. Anonymize sender names (e.g. 'Alice: ' -> 'Person0: ')
    def _replace_sender(match):
        raw_name = match.group(1).strip()
        # Do not redact known headers, pure digits, or already anonymized aliases
        if raw_name.upper() in EXCLUDED_HEADERS or re.match(r"^Person\d+$", raw_name):
            return match.group(0)
        if raw_name.isdigit():
            return match.group(0)

        if raw_name not in namemap["map"]:
            namemap["map"][raw_name] = f"Person{namemap['count']}"
            namemap["count"] += 1

        alias = namemap["map"][raw_name]
        return f"{alias}: "

    text = SENDER_RE.sub(_replace_sender, text)

    # 4. Redact known names occurring in the body of messages
    for name, alias in sorted(namemap["map"].items(), key=lambda x: len(x[0]), reverse=True):
        if len(name) >= 3 and name.upper() not in EXCLUDED_HEADERS:
            text = re.sub(r"\b" + re.escape(name) + r"\b", alias, text)

    return text


def redact_text(text: str, namemap: dict | None = None) -> str:
    """Redact PII from a single text string with optional caller-provided namemap."""
    if namemap is None:
        namemap = {"map": {}, "count": 0}
    return redact_line(text, namemap)


def scan_batch(batch: dict, namemap: dict | None = None) -> dict:
    """Redact PII across a batch of text lines.

    Accepts an optional namemap to support both scan_batch(batch) and
    scan_batch(batch, namemap) callers seamlessly.
    """
    if namemap is None:
        namemap = {"map": {}, "count": 0}

    # Resolve text column if 'text' is missing
    col = "text"
    if col not in batch:
        for c in ["transcript", "dialogue", "message", "query", "content", "source_text", "sentence", "post", "line"]:
            if c in batch:
                col = c
                break

    redacted_texts: list[str] = []
    for text in batch.get(col, []):
        redacted_texts.append(redact_line(text, namemap))

    res = {"text": redacted_texts, "redacted": redacted_texts}
    return res


def process_dataset_pii(dataset: Dataset | DatasetDict, batch_size: int = 1000) -> Dataset | DatasetDict:
    """Anonymize and redact PII across all lines in a dataset."""
    namemap = {"map": {}, "count": 0}
    processed_dataset = dataset.map(
        lambda batch: scan_batch(batch, namemap),
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=False,
    )
    return processed_dataset
