"""PII Redaction module for chat logs and transcripts.

Scans for and redacts:
1. Email addresses (e.g. user@domain.com) -> masked with asterisks
2. Phone numbers (standard international, US, and Kenyan formats) -> masked with asterisks
3. Sender names (e.g. 'John Doe: ', 'Grace Wanjiru: ') -> replaced with Person0, Person1, ...
4. Dialogue body mentions of known sender names -> replaced with their assigned Person alias
5. Names found from context (titles like "Sister Mary Chebet", introductions like "this is Grace Wanjiru",
   "Rose Njeri here") -> Person aliases, and later bare mentions of those names (e.g. "Grace")
6. Places: facility names ("Rongai Health Centre", "Sarambei Primary School") and Kenyan place names from
   data/kenya_places.txt (built by build_place_list.py) -> Place0, Place1, ...

Every redacted item is also recorded, so scan_batch can report pii_items and pii_count per line.
"""
import os
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

# --- Names and places found from context (voice-note transcripts have no "Name:" sender labels) ---
_NAME = r"[A-Z][a-z]+(?:['’][a-z]+)?"
_TITLES = (r"(?:Sister|Sr|Brother|Mrs|Mr|Ms|Miss|Mama|Mzee|Dr|Prof|Rev|Pastor|Chief|Corporal|Sergeant|"
           r"Officer|Nurse|Teacher|Madam)")

# Capitalized words that are not names (sentence starters, days, months, greetings, titles...)
_STOP = {w.lower() for w in """
Hello Hi Hey Good Morning Evening Afternoon Night Okay Ok Thanks Thank Sorry Please Anyway Yes No Today Yesterday
Tomorrow Tonight Team Everyone Someone Anyone Nobody Ladies Gentlemen Next Last This That These Those The A An We
They He She It One Some Most Many All Each Our My Your His Her I Also Just Now Then So But And Or If When While
After Before During Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June
July August September October November December Kenya God Jesus Habari Asante Karibu Eeh Sister Brother Mrs Mr Ms
Miss Mama Mzee Dr Prof Rev Pastor Chief Corporal Sergeant Officer Nurse Teacher Madam Person Place Together
""".split()}

_TITLE_RE = re.compile(rf"\b{_TITLES}\.?\s+(?P<name>{_NAME}(?:\s+{_NAME}){{0,2}})")
_INTRO_RE = re.compile(rf"(?i:\b(?:this is|it['’]s|it is|i am|i['’]m|my name is))\s+(?P<name>{_NAME}(?:\s+{_NAME})?)")
_HERE_RE = re.compile(rf"\b(?P<name>{_NAME}(?:\s+{_NAME})?)\s+here\b")
_GROUP_RE = re.compile(rf"\b(?P<name>{_NAME})\s+(?i:women['’]s group|youth group|self[- ]help group|chama)\b")
_FACILITY_RE = re.compile(
    rf"\b(?P<stem>{_NAME}(?:\s+(?:ya|wa|{_NAME})){{0,4}})\s+"
    r"(?i:primary school|secondary school|health cent(?:re|er)|dispensary|hospital|market)\b")

_PLACES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "kenya_places.txt")


def _load_place_regex():
    """Kenyan place names, one per line (optional: skipped if the file is missing)."""
    if not os.path.isfile(_PLACES_FILE):
        return None
    with open(_PLACES_FILE, encoding="utf-8") as f:
        places = sorted({line.strip() for line in f if line.strip()}, key=len, reverse=True)
    if not places:
        return None
    return re.compile(r"(?<![A-Za-z])(?:" + "|".join(re.escape(p) for p in places) + r")(?![A-Za-z])")


_PLACE_RE = _load_place_regex()

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


def _split_trim(text, lead=False, trail=False):
    """Split text into (prefix, core, suffix), dropping stop words from the chosen ends. None if nothing is left."""
    parts = re.split(r"(\s+)", text)  # words at even positions, separators at odd positions
    words = parts[0::2]
    i, j = 0, len(words)
    while lead and i < j and words[i].lower() in _STOP:
        i += 1
    while trail and j > i and words[j - 1].lower() in _STOP:
        j -= 1
    if i >= j or any(w.lower() in _STOP for w in words[i:j]):
        return None
    return "".join(parts[: 2 * i]), "".join(parts[2 * i: 2 * (j - 1) + 1]), "".join(parts[2 * (j - 1) + 1:])


def _person_alias(namemap, name):
    if name not in namemap["map"]:
        namemap["map"][name] = f"Person{namemap['count']}"
        namemap["count"] += 1
    alias = namemap["map"][name]
    tokens = namemap.setdefault("tokens", {})
    for token in name.split():
        if len(token) >= 3 and token.lower() not in _STOP:
            tokens[token] = alias
    return alias


def _place_alias(namemap, name):
    places = namemap.setdefault("places", {})
    if name not in places:
        places[name] = f"Place{len(places)}"
    return places[name]


def redact_line_with_items(text: str, namemap: dict) -> tuple[str, list[str]]:
    """Redact PII from one text string. Returns (redacted_text, list of the original items that were redacted)."""
    if not text:
        return "", []
    found: list[tuple[str, str]] = []  # (original item, key used to merge repeats of the same item)

    def _mask(match):
        found.append((match.group(0), match.group(0)))
        return "*" * len(match.group(0))

    # 1. Mask email addresses with asterisks
    text = EMAIL_RE.sub(_mask, text)

    # 2. Mask phone numbers with asterisks
    text = PHONE_RE.sub(_mask, text)

    # 3. Anonymize sender names (e.g. 'Alice: ' -> 'Person0: ')
    def _replace_sender(match):
        raw_name = match.group(1).strip()
        # Do not redact known headers, pure digits, or already anonymized aliases
        if raw_name.upper() in EXCLUDED_HEADERS or re.match(r"^Person\d+$", raw_name):
            return match.group(0)
        if raw_name.isdigit():
            return match.group(0)
        if len(raw_name.split()) > 4:  # a sentence ending in a colon ("Just adding to her report:"), not a sender name
            return match.group(0)
        alias = _person_alias(namemap, raw_name)
        found.append((raw_name, alias))
        return f"{alias}: "

    text = SENDER_RE.sub(_replace_sender, text)

    # 4. Redact known names occurring in the body of messages
    for name, alias in sorted(namemap["map"].items(), key=lambda x: len(x[0]), reverse=True):
        if len(name) >= 3 and name.upper() not in EXCLUDED_HEADERS:
            text, n = re.subn(r"\b" + re.escape(name) + r"\b", alias, text)
            if n:
                found.append((name, alias))

    # 5. Names found from context: titles, introductions, "<name> here", group names
    def _name_sub(regex, line):
        def repl(m):
            trimmed = _split_trim(m.group("name"), trail=True)
            if not trimmed:
                return m.group(0)
            prefix, core, suffix = trimmed
            alias = _person_alias(namemap, core)
            found.append((core, alias))
            whole = m.group(0)
            return whole[: m.start("name") - m.start()] + prefix + alias + suffix + whole[m.end("name") - m.start():]
        return regex.sub(repl, line)

    for regex in (_TITLE_RE, _INTRO_RE, _HERE_RE, _GROUP_RE):
        text = _name_sub(regex, text)

    # 6. Places: facility names
    def _facility(m):
        trimmed = _split_trim(m.group("stem"), lead=True)
        if not trimmed:
            return m.group(0)
        prefix, core, _ = trimmed
        whole = m.group(0)
        tail = whole[m.end("stem") - m.start():]  # " Health Centre", " dispensary", ...
        if tail.strip()[:1].islower():  # generic noun ("Kampi ya Moto dispensary"): hide the name, keep the noun
            alias = _place_alias(namemap, core)
            found.append((core, alias))
            return prefix + alias + tail
        full = whole[len(prefix):]  # the noun is part of the proper name ("Rongai Health Centre")
        alias = _place_alias(namemap, full)
        if " " in core:  # remember multi-word stems ("Nakuru Level Five") for later shorter mentions
            namemap.setdefault("places", {}).setdefault(core, alias)
        found.append((full, alias))
        return prefix + alias

    text = _FACILITY_RE.sub(_facility, text)

    # 7. Later bare mentions of names and places learned so far (e.g. "Grace" after "Grace Wanjiru")
    known = dict(namemap.get("tokens", {}))
    known.update(namemap.get("places", {}))
    for token in sorted(known, key=len, reverse=True):
        text, n = re.subn(rf"(?<![A-Za-z]){re.escape(token)}(?![A-Za-z])", known[token], text)
        if n:
            found.append((token, known[token]))

    # 8. Kenyan place list (after step 7, so a known longer place name wins over a shorter listed one)
    if _PLACE_RE is not None:
        def _place(m):
            alias = _place_alias(namemap, m.group(0))
            found.append((m.group(0), alias))
            return alias
        text = _PLACE_RE.sub(_place, text)

    # One entry per distinct item: repeats of the same person or place count once
    items, seen_keys = [], set()
    for original, key in found:
        if key not in seen_keys:
            seen_keys.add(key)
            items.append(original)
    return text, items


def redact_line(text: str, namemap: dict) -> str:
    """Redact PII from a single text string."""
    return redact_line_with_items(text, namemap)[0]


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
    pii_items: list[str] = []
    pii_counts: list[int] = []
    for text in batch.get(col, []):
        redacted, items = redact_line_with_items(text, namemap)
        redacted_texts.append(redacted)
        pii_items.append("; ".join(items))
        pii_counts.append(len(items))

    res = {"text": redacted_texts, "redacted": redacted_texts, "pii_items": pii_items, "pii_count": pii_counts}
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
