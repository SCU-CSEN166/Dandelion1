"""What a supervisor needs on top of the TA-format table: a review queue, a follow-up list, and sensitive-topic protection.

Built from the table made by ta_table.build_ta_table (it only reads the anonymized text, never the raw transcript).

- Review queue (build_review_table): every message gets a confidence score and the reasons it was lowered, so a
  person checks the uncertain rows instead of trusting everything. Reasons include: no attendance count, activity
  type not recognised, the QA model and the number rule disagreeing, a possible duplicate, a place guessed from
  context or from a likely speech-recognition misspelling, and capitalized words or long numbers that are still
  visible after redaction (possible missed names or phone numbers).
- Sensitive topics (gender-based violence, HIV, screening results, reproductive and mental health): the message is
  always sent to review, and report_text keeps only the group-level sentences. Sentences about one identifiable
  person ("One of them is Mrs. [NAME] ...", "Her number is [PHONE]") and personal stories are withheld from the
  report, even though the names are already replaced by tags.
- Follow-up list (build_followups): the sentences that ask someone to act (referrals, a person to call back,
  next visits, reports still to send, supplies that ran out). Follow-ups about one person in a sensitive message
  are marked "restricted" so they go to the case owner and not into the general report.
"""
import re

import pandas as pd

from redact import GENERIC_PLACE_WORDS, NOT_PLACE_WORDS, STOP_WORDS
from ta_table import find_participants, find_topic

REVIEW_COLUMNS = ["message_id", "relevance", "activity_type", "confidence", "confidence_level", "needs_review",
                  "review_reasons", "sensitive_topic", "report_text"]
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
    r"\b(?:Mrs|Mr|Ms|Miss|Mama)\.? \[NAME\]|\bhelp (?:her|him)\b|\b(?:she|he) (?:has|is|was|will|needs|agreed|asked)\b|"
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
    ("place guessed from context", 0.10),
]


_ABBREVIATION = re.compile(r"\b(?:Mrs|Mr|Ms|Dr|Sr|Prof|Rev|St|No)\.$")


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation, without breaking after a title ("Mrs. [NAME]")."""
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


def review_reasons(row: dict, qa_participants: str = "", pii_notes: str = "") -> list[str]:
    text = str(row.get("expected_anonymized_text") or "")
    source = str(row.get("transcript") or "") or text
    reasons = []
    if row.get("relevance") == "Event":
        if str(row.get("participants", "")).strip() == "":
            reasons.append("no attendance count found")
        if row.get("activity_type") == "Other field activity":
            reasons.append("activity type not recognised")
        rule = find_participants(source)
        if str(qa_participants).strip().isdigit() and rule is not None and int(qa_participants) != rule:
            reasons.append(f"participant counts disagree (QA model {qa_participants}, number rule {rule})")
        if str(row.get("duplicate_of", "none")) not in ("none", ""):
            reasons.append(f"possible duplicate of message {row['duplicate_of']}: confirm before merging")
    elif find_topic(source) or find_participants(source) is not None:
        reasons.append("possible missed event: mentions an activity but was labeled chit-chat")
    reasons.extend(n for n in str(pii_notes or "").split("; ") if n)
    leftovers = leftover_identifiers(text)
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


def build_review_table(ta_table: pd.DataFrame, rows: list[dict] | None = None) -> pd.DataFrame:
    """rows: the processed dataset rows in the same order as ta_table (for the QA count and the redaction notes)."""
    rows = rows or [{}] * len(ta_table)
    out = []
    for (_, row), extra in zip(ta_table.iterrows(), rows):
        anonymized = str(row["expected_anonymized_text"])
        topic = sensitive_topic(str(row.get("transcript") or "") or anonymized)
        reasons = review_reasons(row.to_dict(), extra.get("participants", ""), extra.get("pii_notes", ""))
        confidence = confidence_from(reasons)
        level = "High" if confidence >= 0.8 else "Medium" if confidence >= 0.5 else "Low"
        if topic:
            reasons.append(f"sensitive topic ({topic}): check before sharing")
        out.append({
            "message_id": row["message_id"], "relevance": row["relevance"], "activity_type": row["activity_type"],
            "confidence": confidence, "confidence_level": level,
            "needs_review": bool(topic) or level != "High",
            "review_reasons": "; ".join(reasons), "sensitive_topic": topic,
            "report_text": protect_text(anonymized, topic),
        })
    return pd.DataFrame(out, columns=REVIEW_COLUMNS)


def build_followups(ta_table: pd.DataFrame) -> pd.DataFrame:
    """One row per sentence that asks for action, from Event messages only (anonymized text)."""
    out = []
    for _, row in ta_table.iterrows():
        if row["relevance"] != "Event":
            continue
        anonymized = str(row["expected_anonymized_text"])
        topic = sensitive_topic(str(row.get("transcript") or "") or anonymized)
        for sentence in split_sentences(anonymized):
            category = next((name for name, pattern in FOLLOWUP_PATTERNS if re.search(pattern, sentence, re.I)), None)
            if not category:
                continue
            restricted = bool(topic) and is_individual(sentence)
            out.append({"message_id": row["message_id"], "activity_type": row["activity_type"], "category": category,
                        "follow_up": sentence, "access": f"restricted ({topic})" if restricted else "general"})
    return pd.DataFrame(out, columns=FOLLOWUP_COLUMNS)
