import hyperscan
from datasets import DatasetDict

db = hyperscan.Database()
db.compile(
    expressions=[
        r"\b\w+@\w+\.\w+\b",  # Email addresses
        r"(?:\+?\d{1,3})? \(?\d{3}\)? \d{3}[-\s]\d{4}",  # Phone numbers
        r"\b[A-Z][a-z]\S+\b" # Names
    ],
    flags = [hyperscan.HS_FLAG_SOM_LEFTMOST] * 3,
)

def scan_batch(batch):
    redacted_texts: list[bytes] = []
    for text in batch["text"]:
        line_matches = []
        bytes_text = text.encode("utf-8")
        db.scan(bytes_text, match_event_handler=lambda id, start, end, *_: line_matches.append((id, start, end)))

        for _, start, end in line_matches:
            #print(f"Redacting text from index {start} to {end}: {bytes_text[start:end].decode('utf-8')} in text {bytes_text.decode('utf-8')}")
            bytes_text = bytes_text[:start] + '*'.encode("utf-8") * (end - start) + bytes_text[end:]
        
        redacted_texts.append(bytes_text.decode("utf-8"))

    return {"text": redacted_texts}

def process_dataset_pii(dataset: DatasetDict, batch_size: int = 1000) -> DatasetDict:
    processed_dataset = dataset.map(scan_batch, batched=True, batch_size=batch_size, load_from_cache_file=False)
    return processed_dataset