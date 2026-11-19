"""Preprocessing utilities for cleaning chat transcripts (WhatsApp metadata, system banners, placeholders)."""
import re


# Patterns for WhatsApp system notices and placeholders
CLEANUP_EXPRESSIONS = [
    # Encryption notices
    r"^(?:\[.*?\]\s*(?:[^:\n]+:\s*)?|.*?\s*-\s*)?Messages and calls are end-to-end encrypted[^\r\n]*(?:\r?\n|$)",
    # Contact cards and attachments
    r"^(?:\[.*?\]\s*(?:[^:\n]+:\s*)?|.*?\s*-\s*)?\*+\s+is a contact\.[^\r\n]*(?:\r?\n|$)",
    r"^(?:\[.*?\]\s*(?:[^:\n]+:\s*)?|.*?\s*-\s*)?<attached:[^\r\n]*?>[^\r\n]*(?:\r?\n|$)",
    # Media omitted notifications
    r"^(?:\[.*?\]\s*(?:[^:\n]+:\s*)?|.*?\s*-\s*)?(?:image|video|audio|document|sticker|GIF|Contact card)\s+omitted[^\r\n]*(?:\r?\n|$)",
    # Message deletion notices
    r"^(?:\[.*?\]\s*(?:[^:\n]+:\s*)?|.*?\s*-\s*)?(?:This message was deleted|You deleted this message)\.?[^\r\n]*(?:\r?\n|$)",
]

_CLEANUP_RE = re.compile("|".join(f"(?:{p})" for p in CLEANUP_EXPRESSIONS), re.MULTILINE | re.IGNORECASE)


def clean_chat_text(text: str) -> str:
    """Strip metadata, boilerplate system banners, and placeholders from chat text."""
    if not text:
        return ""
    cleaned = text.replace("‎", "").replace("‏", "")
    cleaned = _CLEANUP_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def preprocess_dataset(dataset, text_col: str = "text", batch_size: int = 1000):
    """Strip metadata, boilerplate system banners, and placeholders from dataset text."""
    def _clean_batch(batch):
        return {text_col: [clean_chat_text(t) for t in batch[text_col]]}

    return dataset.map(
        _clean_batch,
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=False,
    )
