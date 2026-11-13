from datasets import load_dataset
from redact import process_dataset_pii

def prepare_dataset(dataset_name: str):
    if ".csv" in dataset_name:
        raw = load_dataset("csv", data_files=dataset_name)
    else:
        raw = load_dataset(dataset_name)
    redacted = process_dataset_pii(raw)
    return redacted