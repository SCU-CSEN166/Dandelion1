"""Build the output table in the same format as the TA's hidden-test key.

One row per message with the TA's columns:
  message_id, relevance, activity_type, participants, duplicate_of, pii_count, pii_items,
  expected_anonymized_text, transcript

- relevance: "Event" if the message reports a field activity (an attendance count, or a known activity topic plus
  an action like "we held"), otherwise "Chit-chat". Rule-based, so it does not need a model.
- activity_type: a label from the topic list below; otherwise the QA model's event_type with names and places removed.
- participants: the QA model's count when it has one, otherwise a number pattern ("forty-two women came").
- duplicate_of: message_id of an earlier event with the same activity type and participant count, else "none".
- pii_count / pii_items / expected_anonymized_text: from redact.py, with Person/Place aliases and masks turned into
  the TA's [NAME], [LOCATION], [PHONE] and [EMAIL] tags. (The column is named like the TA's so the two tables line up;
  here it holds our predicted anonymized text.)
"""
import re

import pandas as pd

TA_COLUMNS = ["message_id", "relevance", "activity_type", "participants", "duplicate_of", "pii_count", "pii_items",
              "expected_anonymized_text", "transcript"]

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_NUMWORD = "|".join(list(_UNITS) + list(_TENS) + ["hundred"])
_NUM = rf"(?:\d[\d,]*|(?:{_NUMWORD})(?:[- ](?:{_NUMWORD}))*)"
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
_ACTION = re.compile(r"\b(?:we held|we did|we went|we conducted|we organi[sz]ed|we carried out|we ran|"
                     r"we visited|we trained|held a|conducted a)\b", re.I)


def words_to_int(text: str) -> int:
    text = text.lower().replace(",", "")
    if text.replace(" ", "").isdigit():
        return int(text)
    current = 0
    for tok in re.split(r"[- ]+", text):
        if tok in _UNITS:
            current += _UNITS[tok]
        elif tok in _TENS:
            current += _TENS[tok]
        elif tok == "hundred":
            current = max(current, 1) * 100
    return current


def find_participants(text: str) -> int | None:
    """The attendance count stated in the text (first one), or None."""
    hits = []
    for pat in (_PAT_A, _PAT_B, _PAT_C):
        m = pat.search(text or "")
        if m:
            hits.append((m.start(), words_to_int(m.group("n"))))
    return min(hits)[1] if hits else None


def find_topic(text: str) -> str:
    for pattern, label in TOPICS:
        if re.search(pattern, text or "", re.I):
            return label
    return ""


def is_event(text: str) -> bool:
    return find_participants(text) is not None or bool(find_topic(text) and _ACTION.search(text or ""))


def clean_qa_label(label: str) -> str:
    """Drop redaction aliases and tags from a QA event label ("Place0 outreach" -> "Outreach")."""
    label = re.sub(r"\b(?:Person|Place)\d+\b|\[(?:NAME|LOCATION|PHONE|EMAIL)\]|\*{4,}", " ", str(label or ""))
    label = re.sub(r"\s+", " ", label).strip(" .,;:-")
    return label[:1].upper() + label[1:] if len(re.sub(r"[^A-Za-z]", "", label)) >= 3 else ""


def to_ta_tags(text: str, pii_items: str = "") -> str:
    """Turn Person/Place aliases and asterisk masks into the TA's [NAME] / [LOCATION] / [PHONE] / [EMAIL] tags."""
    email_lengths = {len(i.strip()) for i in str(pii_items).split(";") if "@" in i}
    text = re.sub(r"\bPerson\d+\b", "[NAME]", str(text or ""))
    text = re.sub(r"\bPlace\d+\b", "[LOCATION]", text)
    return re.sub(r"\*{4,}", lambda m: "[EMAIL]" if len(m.group(0)) in email_lengths else "[PHONE]", text)


def _participants_from(row: dict, text: str):
    qa = str(row.get("participants") or "").strip()
    if qa.isdigit():
        return int(qa)
    return find_participants(text)


def build_ta_table(rows: list[dict], text_key: str = "text") -> pd.DataFrame:
    """rows: dicts from the processed dataset (redacted 'text', 'pii_items', 'pii_count', optional QA columns)."""
    out, earlier_events = [], []  # earlier_events: (message_id, activity, participants)
    for i, row in enumerate(rows):
        message_id = row.get("message_id")
        message_id = i + 1 if message_id in (None, "") else message_id
        redacted = str(row.get(text_key) or "")
        transcript = str(row.get("transcript") or row.get("original_text") or "")
        source = transcript or redacted  # topic and number rules see the same words either way

        relevance, activity, participants, duplicate_of = "Chit-chat", "", None, "none"
        if is_event(source):
            relevance = "Event"
            activity = find_topic(source) or clean_qa_label(row.get("event_type", "")) or "Other field activity"
            participants = _participants_from(row, source)
            if participants is not None:
                for earlier_id, earlier_activity, earlier_participants in earlier_events:
                    if earlier_activity == activity and earlier_participants == participants:
                        duplicate_of = str(earlier_id)
                        break
                else:
                    earlier_events.append((message_id, activity, participants))

        items = str(row.get("pii_items") or "")
        out.append({
            "message_id": message_id,
            "relevance": relevance,
            "activity_type": activity,
            "participants": "" if participants is None else participants,
            "duplicate_of": duplicate_of,
            "pii_count": int(row.get("pii_count") or 0),
            "pii_items": items,
            "expected_anonymized_text": to_ta_tags(redacted, items),
            "transcript": transcript,
        })
    return pd.DataFrame(out, columns=TA_COLUMNS)


def write_ta_table(df: pd.DataFrame, path: str) -> None:
    if path.lower().endswith((".xlsx", ".xls")):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
