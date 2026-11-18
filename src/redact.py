"""PII Redaction module for chat logs and transcripts.

Scans for and redacts:
1. Email addresses (e.g. user@domain.com) -> [EMAIL]
2. Phone numbers (standard international, US, and Kenyan formats) -> [PHONE]
3. Sender names after a WhatsApp export timestamp ('[12/03/24, 10:15] Grace Wanjiru: ', '12/03/2024, 10:15 - John: ')
   -> replaced with Person0, Person1, ... (bare 'Name: ' labels only with bare_senders=True, for dialogue datasets)
4. Dialogue body mentions of known sender names -> replaced with their assigned Person alias
5. Names found from context (titles like "Sister Mary Chebet", introductions like "this is Grace Wanjiru",
   "Rose Njeri here") -> Person aliases, and later bare mentions of those names (e.g. "Grace")
6. Places: facility names ("Rongai Health Centre", "Sarambei Primary School") and Kenyan place names from
   data/kenya_places.txt (built by build_place_list.py) -> Place0, Place1, ...

7. Places and people found from sentence context ("the matatu from Solai", "her home in Kabarak",
   "send it to Ruth"). This also covers names that speech recognition misspelled, which no list contains.
8. Speech-recognition misspellings of known places ("Girissa" for Garissa, "Kampiamoto" for Kampi ya Moto),
   matched approximately against the place list and the places already seen. Each such match is noted for review.

Every redacted item is also recorded, so scan_batch can report pii_items, pii_count and pii_notes per line.
scan_batch reads the batch twice: once to learn the names and places, once to redact, so a name introduced in a
later message is also hidden in the earlier ones.
"""
import os
import re
from datasets import Dataset

# Columns that hold unredacted text: used inside the pipeline, never written to output files
RAW_COLUMNS = ("original_text", "transcript", "pii_items")


# Email regex pattern
EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

# Phone regex patterns:
# - Kenyan formats: +254 7XX XXX XXX, 07XX XXX XXX, 01XX XXX XXX, 07560080089, with spaces/hyphens/compact
# - International / North American formats: +1 (123) 456-7890, (123) 456-7890, 123-456-7890
KENYAN_PHONE_PATTERN = r"(?:\+?254|0)[\s-]?[17](?:[\s-]?\d){7,9}\b"
STANDARD_PHONE_PATTERN = r"(?:\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}\b"
PHONE_PATTERN = f"(?:{KENYAN_PHONE_PATTERN}|{STANDARD_PHONE_PATTERN})"

# WhatsApp export timestamp in front of a sender: "[12/03/24, 10:15:22] " (iOS) or "12/03/2024, 10:15 - " (Android)
_TIMESTAMP = (r"(?:\[[^\]\n]*\d{1,2}[:.]\d{2}[^\]\n]*\]\s*"
              r"|\d{1,4}[./-]\d{1,2}[./-]\d{1,4},?\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap]\.?[Mm]\.?)?\s+-\s+)")
# Sender name before a colon (no line breaks inside, so a line without a colon is never merged into the next line)
_SENDER_NAME = r"(?P<name>[A-Za-z0-9_. \t]+?):\s"

# Non-name headers and labels that should never be redacted as person names
EXCLUDED_HEADERS = {
    "SUBREDDIT", "TITLE", "POST", "TL;DR", "TLDR", "NOTE", "WARNING",
    "REPORT", "EVENT", "CHIT-CHAT", "HTTP", "HTTPS", "TIME", "DATE",
    "MESSAGE", "TRANSCRIPT", "DIALOGUE", "QUERY", "CONTENT", "INPUT", "REFERENCE"
}

# Words that mark a form field label ("Contact Number:", "Date of Birth:"), never a sender name
FORM_WORDS = {
    "NAME", "ADDRESS", "CONTACT", "NUMBER", "EMAIL", "PHONE", "TELEPHONE", "MOBILE", "GENDER", "SEX", "AGE", "DATE",
    "BIRTH", "DOB", "EMERGENCY", "PATIENT", "PARTICIPANT", "DOCUMENT", "DETAILS", "LOCATION", "COMPLAINT", "CITY",
    "COUNTRY", "STATE", "ID", "LICENSE", "LICENCE", "SOCIAL", "DRIVER", "SUBJECT", "TOTAL", "SUMMARY", "STATUS",
    "TYPE", "INFORMATION", "INFO", "NOTES", "COMMENTS", "DESCRIPTION", "REASON", "SIGNATURE", "PASSPORT", "ACCOUNT",
}

EMAIL_RE = re.compile(EMAIL_PATTERN, re.IGNORECASE)
PHONE_RE = re.compile(PHONE_PATTERN)
# Only a timestamped label counts as a sender: a bare "Update:" or "Agenda:" heading must not become a person.
SENDER_RE = re.compile(rf"^(?P<ts>{_TIMESTAMP}){_SENDER_NAME}", re.MULTILINE)
# Dialogue datasets without timestamps ("Amanda: hi"), used by the evaluation scripts with bare_senders=True
BARE_SENDER_RE = re.compile(rf"^(?P<ts>(?:{_TIMESTAMP})?){_SENDER_NAME}", re.MULTILINE)

# --- Names and places found from context (voice-note transcripts have no "Name:" sender labels) ---
_NAME = r"[A-Z][a-z]+(?:['’][a-z]+)?"
_TITLES = (r"(?:Sister|Sr|Brother|Mrs|Mr|Ms|Miss|Mama|Mzee|Dr|Prof|Rev|Pastor|Chief|Corporal|Sergeant|"
           r"Officer|Nurse|Teacher|Madam)")

# Capitalized words that are not names (sentence starters, days, months, greetings, titles...)
STOP_WORDS = {w.lower() for w in """
Hello Hi Hey Good Morning Evening Afternoon Night Okay Ok Thanks Thank Sorry Please Anyway Yes No Today Yesterday
Tomorrow Tonight Team Everyone Someone Anyone Nobody Ladies Gentlemen Next Last This That These Those The A An We
They He She It One Some Most Many All Each Our My Your His Her I Also Just Now Then So But And Or If When While
After Before During Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June
July August September October November December Kenya God Jesus Habari Asante Karibu Eeh Sister Brother Mrs Mr Ms
Miss Mama Mzee Dr Prof Rev Pastor Chief Corporal Sergeant Officer Nurse Teacher Madam Person Place Together
""".split()}

_FULLNAME = rf"{_NAME}(?:\s+[A-Z]\.)?(?:\s+{_NAME})?"  # "Grace", "Grace Wanjiru", "Faith A. King"
_TITLE_RE = re.compile(rf"\b(?i:{_TITLES})\.?\s+(?P<name>{_NAME}(?:\s+[A-Z]\.)?(?:\s+{_NAME}){{0,2}})")
_INTRO_RE = re.compile(rf"(?i:\b(?:this is|it['’]s|it is|i am|i['’]m|my name is))\s+(?P<name>{_FULLNAME})")
_HERE_RE = re.compile(rf"\b(?P<name>{_FULLNAME})\s+here\b")

# People from sentence context: "send your contribution to Ruth", "I was with Grace", "call Amina"
_GIVE_TO_RE = re.compile(
    rf"\b(?i:send|sent|give|gave|pay|paid|talk|talked|speak|spoke|contribution|contributions|money)\b[^.!?\n]{{0,40}}?"
    rf"\bto\s+(?P<name>{_FULLNAME})\b")
_WITH_RE = re.compile(rf"\b(?i:with|call|ask|tell|thank|remind|helping)\s+(?P<name>{_FULLNAME})\b")

# Places from sentence context: "from Solai", "in Kabarak", "at the Kampia Moto Outreach", "referred to Nakuru"
_PLACE_STEM = rf"{_NAME}(?:\s+(?:ya|wa|{_NAME})){{0,3}}"
_PREP_PLACE_RE = re.compile(rf"\b(?i:in|from|at|near|around|outside)\s+(?:the\s+)?(?P<stem>{_PLACE_STEM})\b")
_GO_TO_RE = re.compile(
    rf"\b(?i:went|go|going|gone|came|come|coming|travel\w*|refer\w*|moved|return\w*|trip|visit)\s+to\s+(?:the\s+)?"
    rf"(?P<stem>{_PLACE_STEM})\b")

# Generic words that can end a place phrase; they stay visible ("Kampia Moto Outreach" -> "Place0 Outreach")
GENERIC_PLACE_WORDS = {w.lower() for w in """
Outreach Camp Clinic Meeting Training Session Dispensary Market Hospital School Centre Center Church Office Hall
Group Level Road Town Village County Ward Area Stage Primary Secondary Health Women Youth Men Girls Boys
""".split()}
STOP_WORDS |= {w.lower() for w in """
English Swahili Kiswahili Christmas Easter Whatsapp WhatsApp Facebook Google Zoom Tecno Safaricom Pesa Mpesa Africa
Lord Total
""".split()}
# Capitalized words that can follow "in/at/from" without being a place name
NOT_PLACE_WORDS = {w.lower() for w in "Level Class Form Home School Church Town Work Hospital Case Order Time".split()}
_GROUP_RE = re.compile(rf"\b(?P<name>{_NAME})\s+(?i:women['’]s group|youth group|self[- ]help group|chama)\b")
_FACILITY_RE = re.compile(
    rf"\b(?P<stem>{_NAME}(?:\s+(?:ya|wa|\d+|{_NAME})){{0,4}})\s+"
    r"(?i:primary school|secondary school|health cent(?:re|er)|dispensary|hospital|market)\b")

_PLACES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "kenya_places.txt")


def _load_place_names():
    """Kenyan place names, one per line (optional: empty if the file is missing)."""
    if not os.path.isfile(_PLACES_FILE):
        return []
    with open(_PLACES_FILE, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _norm(name):
    return re.sub(r"[^a-z]", "", name.lower())


_PLACE_NAMES = _load_place_names()
_PLACE_RE = re.compile(
    r"(?<![A-Za-z])(?:" + "|".join(re.escape(p) for p in sorted(set(_PLACE_NAMES), key=len, reverse=True))
    + r")(?![A-Za-z])") if _PLACE_NAMES else None
_LISTED_PLACES = {_norm(p): p for p in _PLACE_NAMES if len(_norm(p)) >= 5}


def _edit_distance(a, b, limit):
    """Levenshtein distance, or limit + 1 as soon as it is certain to be larger."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def _fuzzy_place(token, namemap):
    """A known place that this capitalized word is probably a misspelling of (speech-recognition error), or None."""
    norm = _norm(token)
    if len(norm) < 5:
        return None
    limit = 1 if len(norm) < 8 else 2
    seen = {_norm(p): p for p in namemap.get("places", {}) if len(_norm(p)) >= 5}
    best = None
    for candidate, original in {**_LISTED_PLACES, **seen}.items():
        if candidate[0] != norm[0] or candidate == norm:
            continue
        d = _edit_distance(norm, candidate, limit)
        if d <= limit and (best is None or d < best[0]):
            best = (d, original)
    return best[1] if best else None


def _split_trim(text, lead=False, trail=False):
    """Split text into (prefix, core, suffix), dropping stop words from the chosen ends. None if nothing is left."""
    parts = re.split(r"(\s+)", text)  # words at even positions, separators at odd positions
    words = parts[0::2]
    i, j = 0, len(words)
    while lead and i < j and words[i].lower() in STOP_WORDS:
        i += 1
    while trail and j > i and words[j - 1].lower() in STOP_WORDS:
        j -= 1
    if i >= j or any(w.lower() in STOP_WORDS for w in words[i:j]):
        return None
    return "".join(parts[: 2 * i]), "".join(parts[2 * i: 2 * (j - 1) + 1]), "".join(parts[2 * (j - 1) + 1:])


def _person_alias(namemap, name):
    if name not in namemap["map"]:
        namemap["map"][name] = f"Person{namemap['count']}"
        namemap["count"] += 1
    alias = namemap["map"][name]
    tokens = namemap.setdefault("tokens", {})
    for token in name.split():
        if len(token) >= 3 and token.lower() not in STOP_WORDS:
            tokens[token] = alias
    return alias


def _place_alias(namemap, name):
    places = namemap.setdefault("places", {})
    if name not in places:
        places[name] = f"Place{len(places)}"
    return places[name]


def redact_line(text: str, namemap: dict, bare_senders: bool = False) -> tuple[str, list[str], list[str]]:
    """Redact PII from one text string.

    bare_senders: also treat a "Name: " label without a WhatsApp timestamp as a sender (for dialogue datasets).
    Returns (redacted_text, the original items that were redacted, notes on low-certainty decisions).
    """
    if not text:
        return "", [], []
    found: list[tuple[str, str]] = []  # (original item, key used to merge repeats of the same item)
    notes: list[str] = []  # low-certainty decisions, for the human review queue
    origin = namemap.setdefault("origin", {})  # place -> how it was found, when that was a guess (alias only, no PII)

    def _tagger(tag):
        def _tag(match):
            found.append((match.group(0), match.group(0)))
            return tag
        return _tag

    # 1-2. Email addresses and phone numbers
    text = EMAIL_RE.sub(_tagger("[EMAIL]"), text)
    text = PHONE_RE.sub(_tagger("[PHONE]"), text)

    # 3. Anonymize sender names (e.g. '[12/03/24, 10:15] Alice: ' -> '[12/03/24, 10:15] Person0: ')
    def _replace_sender(match):
        raw_name = match.group("name").strip()
        # Do not redact known headers, pure digits, or already anonymized aliases
        if raw_name.upper() in EXCLUDED_HEADERS or re.match(r"^Person\d+$", raw_name):
            return match.group(0)
        if raw_name.isdigit():
            return match.group(0)
        words = raw_name.split()
        if len(words) > 4:  # a sentence ending in a colon ("Just adding to her report:"), not a sender name
            return match.group(0)
        if any(w.upper().strip(".") in FORM_WORDS for w in words):  # a form field ("Contact Number:", "Date of Birth:")
            return match.group(0)
        if len(words) > 1 and not all(w[0].isupper() or w[0].isdigit() for w in words):
            return match.group(0)  # a lowercase phrase ("details for treatment:"), not a name
        alias = _person_alias(namemap, raw_name)
        found.append((raw_name, alias))
        return f"{match.group('ts')}{alias}: "

    text = (BARE_SENDER_RE if bare_senders else SENDER_RE).sub(_replace_sender, text)

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
            if not trimmed or any(w.upper().strip(".") in FORM_WORDS for w in trimmed[1].split()):
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
            if token in origin:
                notes.append(origin[token])

    # 8. Kenyan place list (after step 7, so a known longer place name wins over a shorter listed one)
    if _PLACE_RE is not None:
        def _place(m):
            alias = _place_alias(namemap, m.group(0))
            found.append((m.group(0), alias))
            return alias
        text = _PLACE_RE.sub(_place, text)

    # 9. People and places from sentence context (also catches misspelled names that no list contains)
    for regex in (_GIVE_TO_RE, _WITH_RE):
        text = _name_sub(regex, text)

    def _context_place(m):
        stem = m.group("stem")
        parts = re.split(r"(\s+)", stem)
        words = parts[0::2]
        j = len(words)
        while j > 0 and words[j - 1].lower() in GENERIC_PLACE_WORDS:  # keep "Outreach", "Camp", ... visible
            j -= 1
        if j == 0:
            return m.group(0)
        core = "".join(parts[: 2 * (j - 1) + 1])
        trimmed = _split_trim(core, lead=True, trail=True)
        if not trimmed or trimmed[2] or any(w.lower() in NOT_PLACE_WORDS for w in trimmed[1].split()):
            return m.group(0)
        prefix, name, _ = trimmed
        match = _fuzzy_place(name, namemap)
        if match:  # a misspelling of a place we already know: reuse its alias
            alias = _place_alias(namemap, match)
            namemap.setdefault("places", {}).setdefault(name, alias)
            origin[name] = f"possible misspelling of a known place: {alias}"
        else:
            alias = _place_alias(namemap, name)
            origin.setdefault(name, f"place guessed from context: {alias}")
        found.append((name, alias))
        notes.append(origin[name])
        start = m.start("stem") - m.start()
        return m.group(0)[:start] + prefix + alias + stem[len(core):]

    for regex in (_PREP_PLACE_RE, _GO_TO_RE):
        text = regex.sub(_context_place, text)

    # 10. Misspelled places: capitalized words inside a sentence that are close to a known place name
    def _fuzzy(m):
        token = m.group(0)
        before = text[: m.start()].rstrip()
        if not before or before[-1] in ".!?\n" or token.lower() in STOP_WORDS or token.lower() in GENERIC_PLACE_WORDS:
            return token  # sentence-initial words are capitalized anyway, so they are skipped
        match = _fuzzy_place(token, namemap)
        if not match:
            return token
        alias = _place_alias(namemap, match)
        namemap.setdefault("places", {}).setdefault(token, alias)
        origin[token] = f"possible misspelling of a known place: {alias}"
        found.append((token, alias))
        notes.append(origin[token])
        return alias

    text = re.sub(r"(?<![A-Za-z0-9])[A-Z][a-z]{4,}(?![A-Za-z0-9])", _fuzzy, text)

    # One entry per distinct item: repeats of the same person or place count once
    items, seen_keys = [], set()
    for original, key in found:
        if key not in seen_keys:
            seen_keys.add(key)
            items.append(original)
    return text, items, list(dict.fromkeys(notes))


def scan_batch(batch: dict, namemap: dict | None = None, bare_senders: bool = False) -> dict:
    """Redact PII across a batch of 'text' lines, sharing one namemap so aliases stay consistent.

    bare_senders: also treat "Name: " labels without a WhatsApp timestamp as senders (see redact_line).
    """
    if namemap is None:
        namemap = {"map": {}, "count": 0}

    texts = list(batch["text"])
    for text in texts:  # first pass only learns the names and places
        redact_line(text, namemap, bare_senders)

    res = {"text": [], "pii_items": [], "pii_count": [], "pii_notes": []}
    for text in texts:
        redacted, items, notes = redact_line(text, namemap, bare_senders)
        res["text"].append(redacted)
        res["pii_items"].append("; ".join(items))
        res["pii_count"].append(len(items))
        res["pii_notes"].append("; ".join(notes))
    return res


def process_dataset_pii(dataset: Dataset) -> Dataset:
    """Anonymize and redact PII across all lines in a dataset.

    The whole dataset is one batch, so scan_batch's first pass learns every name and place before any line is
    redacted: a name introduced in the last message is also hidden in the first.
    """
    return dataset.map(
        lambda batch: scan_batch(batch, {"map": {}, "count": 0}),
        batched=True,
        batch_size=None,
        load_from_cache_file=False,
    )
