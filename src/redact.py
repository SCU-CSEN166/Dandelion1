import os
import re

import hyperscan
from datasets import DatasetDict

PATTERNS = [
    r"\b\w+@\w+\.\w+\b",  # 0: Email addresses
    r"(?:\+?\d{1,3})? \(?\d{3}\)? \d{3}[-\s]\d{4}",  # 1: Phone numbers (US-style)
    r"\b\S+:\s",  # 2: Names (speaker labels)
    r"(?:\+254|\b254|\b0)[ -]?[17]\d{2}[ -]?\d{3}[ -]?\d{3}\b",  # 3: Kenyan phone numbers, e.g. 0711 000 482
]

db = hyperscan.Database()
db.compile(
    expressions=PATTERNS,
    flags=[hyperscan.HS_FLAG_SOM_LEFTMOST] * len(PATTERNS),
)

# ---------------------------------------------------------------------------
# Context-based names and places (voice-note transcripts have no "Name:" labels)
# ---------------------------------------------------------------------------
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
_FACILITY_RE = re.compile(
    rf"\b(?P<stem>{_NAME}(?:\s+(?:ya|wa|{_NAME})){{0,4}})\s+"
    r"(?i:primary school|secondary school|health cent(?:re|er)|dispensary|hospital|market)\b")

_PLACES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "kenya_places.txt")


def _load_place_regex():
    """Kenyan place names (built by build_place_list.py). Optional: skipped if the file is missing."""
    if not os.path.isfile(_PLACES_FILE):
        return None
    with open(_PLACES_FILE, encoding="utf-8") as f:
        places = sorted({line.strip() for line in f if line.strip()}, key=len, reverse=True)
    if not places:
        return None
    return re.compile(r"(?<![A-Za-z])(?:" + "|".join(re.escape(p) for p in places) + r")(?![A-Za-z])")


_PLACE_RE = _load_place_regex()


def _split_trim(text, lead=False, trail=False):
    """Split text into (prefix, core, suffix), dropping stop words from the chosen ends. None if nothing is left."""
    parts = re.split(r"(\s+)", text)  # words at even positions, separators at odd positions
    words = parts[0::2]
    i, j = 0, len(words)
    while lead and i < j and words[i].lower() in _STOP:
        i += 1
    while trail and j > i and words[j - 1].lower() in _STOP:
        j -= 1
    if i >= j:
        return None
    core_words = words[i:j]
    if any(w.lower() in _STOP for w in core_words):
        return None
    prefix = "".join(parts[: 2 * i])
    core = "".join(parts[2 * i: 2 * (j - 1) + 1])
    suffix = "".join(parts[2 * (j - 1) + 1:])
    return prefix, core, suffix


def _person(namemap, name):
    if name not in namemap["map"]:
        namemap["map"][name] = f"Person{namemap['count']}"
        namemap["count"] += 1
    pseudonym = namemap["map"][name]
    tokens = namemap.setdefault("name_tokens", {})
    for token in name.split():
        if len(token) >= 3 and token.lower() not in _STOP:
            tokens[token] = pseudonym
    return pseudonym


def _place(namemap, name):
    places = namemap.setdefault("places", {})
    if name not in places:
        places[name] = f"Place{len(places)}"
    return places[name]


def _sub_name(regex, text, namemap):
    def repl(m):
        trimmed = _split_trim(m.group("name"), trail=True)
        if not trimmed:
            return m.group(0)
        prefix, core, suffix = trimmed
        return m.group(0)[: m.start("name") - m.start()] + prefix + _person(namemap, core) + suffix
    return regex.sub(repl, text)


def _sub_facility(text, namemap):
    def repl(m):
        trimmed = _split_trim(m.group("stem"), lead=True)
        if not trimmed:
            return m.group(0)
        prefix, core, _ = trimmed
        pseudonym = _place(namemap, core)
        return prefix + pseudonym

    return _FACILITY_RE.sub(repl, text)


def _detect_context(text, namemap):
    """Redact names (titles, introductions) and places (facility names, Kenyan place list) found in context."""
    text = _sub_name(_TITLE_RE, text, namemap)
    text = _sub_name(_INTRO_RE, text, namemap)
    text = _sub_name(_HERE_RE, text, namemap)
    text = _sub_facility(text, namemap)
    if _PLACE_RE is not None:
        text = _PLACE_RE.sub(lambda m: _place(namemap, m.group(0)), text)
    return text


def _propagate(text, namemap):
    """Replace later mentions of names and places learned elsewhere in the same batch (e.g. a bare first name)."""
    known = dict(namemap.get("name_tokens", {}))
    known.update(namemap.get("places", {}))
    for token in sorted(known, key=len, reverse=True):
        text = re.sub(rf"(?<![A-Za-z]){re.escape(token)}(?![A-Za-z])", known[token], text)
    return text


def replace_subs(source: bytes, sub: str, start: int, end: int):
    subenc: bytes = sub.encode()
    return source[:start] + subenc + source[end:], len(subenc) - (end - start)

def scan_batch(batch, namemap):
    redacted_texts: list[str] = []
    for text in batch["text"]:
        line_matches = []
        bytes_text = text.encode()
        db.scan(bytes_text, match_event_handler=lambda id, start, end, *_: line_matches.append((id, start, end)))

        # Ensure matches are processed in left-to-right order
        line_matches.sort(key=lambda m: (m[1], m[2]))

        offset = 0
        prev_end = 0
        for id, start, end in line_matches:
            if start < prev_end:
                continue
            prev_end = end

            adj_start = start + offset
            adj_end = end + offset
            match id:
                case 2:
                    name = bytes_text[adj_start:adj_end].decode(errors="replace")
                    if name not in namemap['map']:
                        namemap['map'][name] = f"Person{namemap['count']}"
                        namemap['count'] += 1
                    bytes_text, delta = replace_subs(bytes_text, namemap['map'][name] + ': ', adj_start, adj_end)
                    offset += delta
                case _:
                    bytes_text, delta = replace_subs(bytes_text, '*' * (end - start), adj_start, adj_end)
                    offset += delta

        redacted_texts.append(_detect_context(bytes_text.decode(errors="replace"), namemap))

    # Second pass over the whole batch, so a name or place learned in one message is also hidden in the others
    redacted_texts = [_propagate(t, namemap) for t in redacted_texts]
    return {"text": redacted_texts}

def process_dataset_pii(dataset: DatasetDict, batch_size: int = 1000) -> DatasetDict:
    namemap = {"map": dict(), "count": 0}
    processed_dataset = dataset.map(lambda batch: scan_batch(batch, namemap), batched=True, batch_size=batch_size, load_from_cache_file=False)
    return processed_dataset
