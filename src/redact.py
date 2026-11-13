import hyperscan
from datasets import DatasetDict

db = hyperscan.Database()
db.compile(
    expressions=[
        r"\b\w+@\w+\.\w+\b",  # Email addresses
        r"(?:\+?\d{1,3})? \(?\d{3}\)? \d{3}[-\s]\d{4}",  # Phone numbers
        r"\b\S+:\s" # Names
    ],
    flags=[hyperscan.HS_FLAG_SOM_LEFTMOST] * 3,
)

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

        redacted_texts.append(bytes_text.decode(errors="replace"))

    return {"text": redacted_texts}

def process_dataset_pii(dataset: DatasetDict, batch_size: int = 1000) -> DatasetDict:
    namemap = {"map": dict(), "count": 0}
    processed_dataset = dataset.map(lambda batch: scan_batch(batch, namemap), batched=True, batch_size=batch_size, load_from_cache_file=False)
    return processed_dataset
