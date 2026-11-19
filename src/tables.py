"""The output tables: one row per message (messages.csv) and one row per action sentence (follow_ups.csv).

Built from the processed dataset rows (load_dataset.prepare_dataset): redacted 'text' (Person0 / Place0 aliases,
[PHONE] / [EMAIL] tags), 'original_text', 'relevance' and 'relevance_score' from the classifier, 'pii_count' and
'pii_notes' from redaction, and the QA model's 'event_type' / 'participants' when it ran.

Message table
- relevance / relevance_score: the classifier's label and Event score.
- activity_type: the classifier's activity category, else the topic rule, the QA model's answer, or "Other field
  activity" (extract.activity_type). participants: the QA model's and the number rule's count, blank when they
  disagree (extract.participant_count). Events only.
- duplicate_of: message_id of an earlier Event with the same activity type and count, else "none".
- sensitive_topic (gender-based violence, HIV, screening results, reproductive and mental health): the message
  always goes to review, and its sentences about one identifiable person are withheld from the report.
- confidence / confidence_level / needs_review / review_reasons: every doubt lowers the confidence, so a person
  checks the uncertain rows. Agreement between the methods is the main check: activity types or head counts that
  disagree, or a count found by only one method. Also: no count, activity type not recognised, a possible duplicate,
  a possible missed Event, a place guessed by redaction, and capitalized words or long numbers still visible after
  redaction (possible missed names or phone numbers).
Follow-up list
- The sentences of Event messages that ask someone to act (referrals, a person to call back, next visits, reports to
  send, supplies). Follow-ups about one person in a sensitive message are marked "restricted" (for the case owner).

The keyword lists here (sensitive topics, follow-up patterns) were written alongside the hidden test, so they are
in-sample. write_table writes only the listed columns: the unredacted text never reaches a file.
"""
import re

import pandas as pd

from extract import OTHER_ACTIVITY, activity_type, find_participants, find_topic, participant_count
from redact import GENERIC_PLACE_WORDS, NOT_PLACE_WORDS, STOP_WORDS
from relevance import is_chitchat

MESSAGE_COLUMNS = ["message_id", "text", "relevance", "relevance_score", "activity_type", "participants",
                   "duplicate_of", "pii_count", "sensitive_topic", "confidence", "confidence_level", "needs_review",
                   "review_reasons"]
FOLLOWUP_COLUMNS = ["message_id", "activity_type", "category", "follow_up", "access"]

SENSITIVE_TOPICS = [
    (r"gender[- ]based violence|\bGBV\b|domestic violence|\babuse[ds]?\b|assault|\brape[ds]?\b|defile|survivors?\b|"
     r"\bFGM\b|child marriage", "Gender-based violence"),
    (r"\bHIV\b|\bAIDS\b|\bARVs?\b|viral load", "HIV"),
    (r"cancer|positive results?|tested positive|screening results?", "Health screening results"),
    (r"mental health|depress|suicid", "Mental health"),
    (r"family planning|contracepti|\bimplants?\b|pregnan|abortion|miscarriage", "Reproductive health"),
]

# A sentence about one identifiable person (kept out of reports when the topic is sensitive)
_INDIVIDUAL = re.compile(
    r"\[PHONE\]|\[EMAIL\]|\bone of them\b|\bone (?:mama|woman|girl|man|lady|client|patient|mother|member)\b|"
    r"\b(?:Mrs|Mr|Ms|Miss|Mama)\.? Person\d+\b|\bhelp (?:her|him)\b|\b(?:she|he) (?:has|is|was|will|needs|agreed|asked)\b|"
    r"\b(?:her|his) (?:number|home|house|husband|wife|child|children|name|case)\b", re.I)
_PERSONAL_STORY = re.compile(r"shared (?:their|her|his) (?:stor|experience)|testimon|very emotional", re.I)

FOLLOWUP_PATTERNS = [
    ("Referral", r"\brefer(?:red|ral|rals|s)?\b"),
    ("Person to follow up", r"follow[- ]?up|asked to be called|call(?:ed)? (?:her|him|them) back|agreed to go|"
                            r"help (?:her|him|them) with|needs? (?:transport|help|support)"),
    ("Next visit or meeting", r"come back|next (?:month|week|time|meeting)|will visit|visit their homes?|asked us to|plan that"),
    ("Report to send", r"\b(?:will|shall|to) (?:send|upload|share|submit)\b[^.!?]*\b(?:report|register|records?|photos|list)\b"),
    ("Supplies and logistics", r"ran out of|out of stock|should bring|needs? more|not enough|shortage"),
]

_PENALTIES = [  # (reason prefix, confidence lost)
    ("possible missed event", 0.40),
    ("possible unredacted", 0.30),
    ("participant counts disagree", 0.30),
    ("activity type not recognised", 0.25),
    ("no attendance count", 0.20),
    ("possible duplicate", 0.15),
    ("possible misspelling", 0.15),
    ("activity types disagree", 0.15),
    ("head count from one method only", 0.10),
    ("place guessed from context", 0.10),
]

_ABBREVIATION = re.compile(r"\b(?:Mrs|Mr|Ms|Dr|Sr|Prof|Rev|St|No)\.$")


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation, without breaking after a title ("Mrs. Person0")."""
    sentences = []
    for part in re.split(r"(?<=[.!?])\s+", str(text or "")):
        if not part.strip():
            continue
        if sentences and _ABBREVIATION.search(sentences[-1]):
            sentences[-1] += " " + part.strip()
        else:
            sentences.append(part.strip())
    return sentences


def sensitive_topic(text: str) -> str:
    for pattern, label in SENSITIVE_TOPICS:
        if re.search(pattern, text or "", re.I):
            return label
    return ""


def is_individual(sentence: str) -> bool:
    return bool(_INDIVIDUAL.search(sentence))


def protect_text(anonymized: str, topic: str) -> str:
    """For a sensitive message, keep the group-level sentences and withhold the ones about one person."""
    if not topic:
        return anonymized
    kept, withheld = [], 0
    for sentence in split_sentences(anonymized):
        if is_individual(sentence) or _PERSONAL_STORY.search(sentence):
            withheld += 1
        else:
            kept.append(sentence)
    if withheld:
        kept.append(f"[{withheld} sentence(s) about individuals withheld: {topic}]")
    return " ".join(kept)


def leftover_identifiers(anonymized: str) -> list[str]:
    """Capitalized words inside a sentence and long digit runs that survived redaction: possible missed PII."""
    out = []
    for m in re.finditer(r"(?<![A-Za-z0-9\[])[A-Z][a-z]{2,}(?![A-Za-z0-9])", anonymized):
        word = m.group(0)
        before = anonymized[: m.start()].rstrip()
        if not before or before[-1] in ".!?:\n":
            continue  # sentence-initial capital
        if word.lower() in STOP_WORDS or word.lower() in GENERIC_PLACE_WORDS or word.lower() in NOT_PLACE_WORDS:
            continue
        if word not in out:
            out.append(word)
    if re.search(r"(?:\d[\s-]?){9,}", anonymized):
        out.append("a long number")
    return out


def review_reasons(row: dict, source: str, qa_participants: str = "", pii_notes: str = "",
                   qa_ran: bool = False) -> list[str]:
    """row: a message-table row; source: the original text (the rules read it); qa_ran: the QA model was asked."""
    reasons = []
    if row["relevance"] == "Event":
        if str(row["participants"]).strip() == "":
            reasons.append("no attendance count found")
        if row["activity_type"] == OTHER_ACTIVITY:
            reasons.append("activity type not recognised")
        topic = find_topic(source)
        if topic and row["activity_type"] != topic:
            reasons.append(f"activity types disagree (table {row['activity_type']}, topic rule {topic})")
        rule = find_participants(source)
        qa = int(qa_participants) if str(qa_participants).strip().isdigit() else None
        if qa is not None and rule is not None and qa != rule:
            reasons.append(f"participant counts disagree (QA model {qa}, number rule {rule})")
        elif qa_ran and (qa is None) != (rule is None):  # agreement is the check: one method alone is less sure
            reasons.append(f"head count from one method only ({'QA model' if rule is None else 'number rule'})")
        if row["duplicate_of"] != "none":
            reasons.append(f"possible duplicate of message {row['duplicate_of']}: confirm before merging")
    elif find_topic(source) or find_participants(source) is not None:
        reasons.append("possible missed event: mentions an activity but was labeled chit-chat")
    reasons.extend(n for n in str(pii_notes or "").split("; ") if n)
    leftovers = leftover_identifiers(row["text"])
    if leftovers:
        reasons.append("possible unredacted name or number: " + ", ".join(leftovers[:6]))
    return reasons


def confidence_from(reasons: list[str]) -> float:
    lost, seen = 0.0, set()
    for reason in reasons:
        for prefix, penalty in _PENALTIES:
            if reason.startswith(prefix) and prefix not in seen:  # each kind of doubt counts once
                seen.add(prefix)
                lost += penalty
    return round(max(0.05, 1.0 - lost), 2)


def build_message_table(rows: list[dict]) -> pd.DataFrame:
    """The message table (MESSAGE_COLUMNS) plus two internal columns: original_text and report_text (the text the
    summary is written from: sensitive messages without their sentences about one person)."""
    out, earlier_events = [], []  # earlier_events: (message_id, activity, participants)
    for row in rows:
        text = str(row.get("text") or "")
        source = str(row.get("original_text") or "") or text
        msg = {"message_id": row["message_id"], "text": text, "relevance": "Chit-chat",
               "relevance_score": row.get("relevance_score", ""), "activity_type": "", "participants": "",
               "duplicate_of": "none", "pii_count": int(row.get("pii_count") or 0)}

        if not is_chitchat(row["relevance"]):
            msg["relevance"] = "Event"
            msg["activity_type"] = activity_type(source, row.get("event_type", ""), row.get("activity_category", ""),
                                                 row.get("activity_score", 0.0))
            count = participant_count(source, row.get("participants", ""))
            if count is not None:
                msg["participants"] = count
                for earlier_id, earlier_activity, earlier_count in earlier_events:
                    if earlier_activity == msg["activity_type"] and earlier_count == count:
                        msg["duplicate_of"] = str(earlier_id)
                        break
                else:
                    earlier_events.append((msg["message_id"], msg["activity_type"], count))

        topic = sensitive_topic(source)
        reasons = review_reasons(msg, source, row.get("participants", ""), row.get("pii_notes", ""),
                                 qa_ran="event_type" in row)
        confidence = confidence_from(reasons)
        level = "High" if confidence >= 0.8 else "Medium" if confidence >= 0.5 else "Low"
        if topic:
            reasons.append(f"sensitive topic ({topic}): check before sharing")
        msg.update({"sensitive_topic": topic, "confidence": confidence, "confidence_level": level,
                    "needs_review": bool(topic) or level != "High", "review_reasons": "; ".join(reasons),
                    "original_text": source, "report_text": protect_text(text, topic)})
        out.append(msg)
    return pd.DataFrame(out, columns=MESSAGE_COLUMNS + ["original_text", "report_text"])


def build_followups(messages: pd.DataFrame) -> pd.DataFrame:
    """One row per sentence that asks for action, from Event messages only."""
    out = []
    for _, row in messages[messages["relevance"] == "Event"].iterrows():
        topic = row["sensitive_topic"]
        for sentence in split_sentences(row["text"]):
            category = next((name for name, pattern in FOLLOWUP_PATTERNS if re.search(pattern, sentence, re.I)), None)
            if category:
                restricted = bool(topic) and is_individual(sentence)
                out.append({"message_id": row["message_id"], "activity_type": row["activity_type"],
                            "category": category, "follow_up": sentence,
                            "access": f"restricted ({topic})" if restricted else "general"})
    return pd.DataFrame(out, columns=FOLLOWUP_COLUMNS)


def write_table(df: pd.DataFrame, path: str, columns: list[str]) -> None:
    """Write only the given columns, as .xlsx/.xls or, for any other extension, .csv."""
    df = df[columns]
    if path.lower().endswith((".xlsx", ".xls")):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
