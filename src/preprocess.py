"""Preprocessing utilities for cleaning chat transcripts (WhatsApp metadata, system banners, placeholders)."""
import re

try:
    import hyperscan
    _HS_AVAILABLE = True
except ImportError:
    hyperscan = None
    _HS_AVAILABLE = False


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

# Precompile Hyperscan database once at module load
_hs_db = None
if _HS_AVAILABLE:
    _hs_db = hyperscan.Database()
    _hs_flags = [
        hyperscan.HS_FLAG_SOM_LEFTMOST
        | hyperscan.HS_FLAG_MULTILINE
        | hyperscan.HS_FLAG_CASELESS
        | hyperscan.HS_FLAG_UTF8
    ] * len(CLEANUP_EXPRESSIONS)
    _hs_db.compile(expressions=CLEANUP_EXPRESSIONS, flags=_hs_flags)


def _hyperscan_clean(text: str) -> str:
    """Fast boilerplate removal using compiled Hyperscan database."""
    cleaned = text.replace("\u200e", "").replace("\u200f", "")
    data = cleaned.encode("utf-8")

    matches: list[tuple[int, int]] = []
    _hs_db.scan(data, match_event_handler=lambda id, start, end, *_: matches.append((start, end)))

    if not matches:
        while "\n\n\n" in cleaned:
            cleaned = cleaned.replace("\n\n\n", "\n\n")
        return cleaned.strip()

    matches.sort(key=lambda m: (m[0], m[1]))

    # Merge overlapping or adjacent match ranges
    merged: list[tuple[int, int]] = []
    for start, end in matches:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    # Single-pass O(N) assembly of non-matched byte slices
    chunks: list[bytes] = []
    last_pos = 0
    for start, end in merged:
        if start > last_pos:
            chunks.append(data[last_pos:start])
        last_pos = max(last_pos, end)
    if last_pos < len(data):
        chunks.append(data[last_pos:])

    result = b"".join(chunks).decode("utf-8", errors="replace")
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")
    return result.strip()


def _regex_clean(text: str) -> str:
    """Fallback boilerplate removal using standard re module."""
    cleaned = text.replace("\u200e", "").replace("\u200f", "")
    pattern = re.compile("|".join(f"(?:{p})" for p in CLEANUP_EXPRESSIONS), re.MULTILINE | re.IGNORECASE)
    cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clean_chat_text(text: str) -> str:
    """Strip metadata, boilerplate system banners, and placeholders from chat text."""
    if not text:
        return ""
    if _HS_AVAILABLE and _hs_db is not None:
        return _hyperscan_clean(text)
    return _regex_clean(text)


def preprocess_dataset(dataset, text_col: str = "text", batch_size: int = 1000):
    """Strip metadata, boilerplate system banners, and placeholders from dataset text."""
    def _clean_batch(batch):
        col = text_col if text_col in batch else ("transcript" if "transcript" in batch else next((c for c in batch if c in ["dialogue", "message", "query", "content", "source_text", "sentence", "post", "line"]), "text"))
        cleaned_lines = [clean_chat_text(t) if t else "" for t in batch.get(col, [])]
        res = {col: cleaned_lines}
        if col != "text":
            res["text"] = cleaned_lines
        return res

    return dataset.map(
        _clean_batch,
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=False,
    )

