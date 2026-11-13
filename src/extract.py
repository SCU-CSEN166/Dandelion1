"""Turn voice-note transcripts into one structured record per message.

Fields (same names as the TA's answer key):
  relevance      "Event" if the message reports a field activity, else "Chit-chat"
  activity_type  short label such as "Family planning outreach" (empty for chit-chat)
  participants   how many people attended / were served (empty if not stated)
  duplicate_of   message_id of the earlier message reporting the same event, or "none"
  anonymized_text  the transcript after redaction, with [NAME] / [LOCATION] / [PHONE] tags

This is a rule-based baseline: keyword and number patterns, no model. It is deterministic and runs anywhere.
Run:  python src/extract.py -i data/hidden_test_transcripts.xlsx -o results/extract_out.csv
"""
import argparse
import os
import re

import pandas as pd

from redact import scan_batch

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_NUMWORD = "|".join(list(_UNITS) + list(_TENS) + ["hundred"])
_NUM = rf"(?:\d[\d,]*|(?:{_NUMWORD})(?:[- ](?:{_NUMWORD}))*)"
_PEOPLE = (r"(?:women|men|girls|boys|ladies|members|people|farmers|participants|mothers|fathers|children|kids|youth|"
           r"students|pupils|villagers|residents|patients|clients|caregivers|elders|persons)")

# number + people + (attended|came|...)   e.g. "forty-two women came", "Sixty girls attended"
_PAT_A = re.compile(rf"\b(?P<n>{_NUM})\s+{_PEOPLE}(?:\s+\w+){{0,2}}?\s+"
                    r"(?:attended|came|turned up|participated|were present|joined|registered|were reached)\b", re.I)
# (served|screened|...) + number + people   e.g. "we served forty-two women", "we had twenty-eight women screened"
_PAT_B = re.compile(rf"\b(?:served|screened|reached|trained|vaccinated|treated|counselled|counseled|tested|had)\s+"
                    rf"(?P<n>{_NUM})\s+{_PEOPLE}\b", re.I)
_PAT_C = re.compile(rf"\b(?:attendance|turnout)\s+(?:was|of)\s+(?P<n>{_NUM})\b", re.I)

# (pattern, label). First match wins, so list specific topics first.
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
    (r"agricultur|farmer training|crop|seedlings", "Farmer training"),
]
_ACTION = re.compile(r"\b(?:we held|we did|we went|we conducted|we organi[sz]ed|we carried out|we ran|"
                     r"we visited|we trained|held a|conducted a)\b", re.I)


def words_to_int(text):
    text = text.lower().replace(",", "")
    if text.replace(" ", "").isdigit():
        return int(text)
    total, current = 0, 0
    for tok in re.split(r"[- ]+", text):
        if tok in _UNITS:
            current += _UNITS[tok]
        elif tok in _TENS:
            current += _TENS[tok]
        elif tok == "hundred":
            current = max(current, 1) * 100
    return total + current


def find_participants(text):
    hits = []
    for pat in (_PAT_A, _PAT_B, _PAT_C):
        m = pat.search(text)
        if m:
            hits.append((m.start(), words_to_int(m.group("n"))))
    return min(hits)[1] if hits else None


def find_activity(text):
    for pattern, label in TOPICS:
        if re.search(pattern, text, re.I):
            return label
    return ""


def classify(text):
    participants = find_participants(text)
    activity = find_activity(text)
    is_event = participants is not None or (activity and _ACTION.search(text) is not None)
    return ("Event", activity or "Other field activity", participants) if is_event else ("Chit-chat", "", None)


def tag(text):
    text = re.sub(r"Person\d+", "[NAME]", text)
    text = re.sub(r"Place\d+", "[LOCATION]", text)
    return re.sub(r"\*{4,}", "[PHONE]", text)  # all masked numbers; email masks are also tagged [PHONE]


def read_input(path):
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    if "transcript" not in df.columns or "message_id" not in df.columns:
        raise SystemExit(f"Input needs message_id and transcript columns. Found: {list(df.columns)}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Structured records from voice-note transcripts")
    parser.add_argument("-i", "--input", required=True, help=".xlsx or .csv with message_id and transcript columns")
    parser.add_argument("-o", "--output", default="results/extract_out.csv")
    args = parser.parse_args()

    df = read_input(args.input)
    texts = df["transcript"].astype(str).tolist()
    anonymized = scan_batch({"text": texts}, {"map": {}, "count": 0})["text"]

    records, seen = [], []  # seen: (message_id, activity, participants) of earlier events
    for message_id, text, anon in zip(df["message_id"], texts, anonymized):
        relevance, activity, participants = classify(text)
        duplicate_of = "none"
        if relevance == "Event" and participants is not None:
            for earlier_id, earlier_activity, earlier_participants in seen:
                if earlier_activity == activity and earlier_participants == participants:
                    duplicate_of = str(earlier_id)
                    break
            else:
                seen.append((message_id, activity, participants))
        records.append({"message_id": message_id, "relevance": relevance, "activity_type": activity,
                        "participants": "" if participants is None else participants,
                        "duplicate_of": duplicate_of, "anonymized_text": tag(anon)})

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pd.DataFrame(records).to_csv(args.output, index=False)
    print(f"Wrote {len(records)} records to {args.output}")
    print(pd.DataFrame(records)[["message_id", "relevance", "activity_type", "participants", "duplicate_of"]].to_string(index=False))


if __name__ == "__main__":
    main()
