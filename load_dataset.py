from datasets import load_dataset
from redact import process_dataset_pii

def prepare_dataset(dataset_name: str):
    raw = load_dataset(dataset_name)
    redacted = process_dataset_pii(raw)
    return redacted