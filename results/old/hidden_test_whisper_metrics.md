# Hidden-test metrics on our own Whisper transcriptions

Question from the TA: were the metrics computed on our own Whisper transcriptions or on the provided transcripts?
This file has both. The Whisper rows use the transcription of the 10 audio files in `data/hidden_test_whisper_transcripts.csv`
(Whisper output from commit 61bb7df); the provided-transcript row is the table from PR #5. All three are scored against the
TA's expected-output table (`data/hidden_test_outputs.xlsx`) with `src/eval_ta_table.py`.

| Table | Input text | Code |
|---|---|---|
| `results/old/hidden_test_whisper_ta_table_main.csv` | our Whisper transcripts | `main` (PR #5), before PR #6 |
| `results/old/hidden_test_whisper_ta_table.csv` | our Whisper transcripts | PR #6 (speech-error handling) |
| `results/old/hidden_test_ta_table.csv` | the provided transcripts | PR #6 |

Reproduce:
```
python src/main.py -d data/hidden_test_whisper_transcripts.csv --extract_info --ta_table_out results/old/hidden_test_whisper_ta_table.csv
python src/eval_ta_table.py -p results/old/hidden_test_whisper_ta_table.csv -k data/hidden_test_outputs.xlsx --metrics_out results/old/hidden_test_whisper_metrics.md
```

Notes: PII tags are scored by aligning our anonymized text with the expected text word by word, because on Whisper text a
misspelled name is not the same string as the key's item. "PII item recall" checks that each key item is gone from our text,
which wrongly passes a misspelled name that was left visible, so on Whisper text read the aligned tag numbers. Event is the
positive class for relevance precision/recall/F1. Anonymized-text exact match cannot match when the transcripts differ.
The rules were written after seeing the test and the key, so all scores are in-sample. Only 10 messages, so one message
moves a score a lot.

### 1. Our Whisper transcripts, code on main (PR #5)

| Metric | Value | Detail |
|---|---:|---|
| Relevance accuracy | 1.000 | 10 messages |
| Relevance precision / recall / F1 (Event = positive) | 1.000 / 1.000 / 1.000 | tp 8, fp 0, fn 0 |
| Activity type accuracy | 1.000 | 8/8 events, word overlap >= 0.5 |
| Participants accuracy | 1.000 | 10/10 |
| duplicate_of accuracy | 1.000 | 10/10 |
| duplicate detection precision / recall / F1 | 1.000 / 1.000 / 1.000 | |
| pii_count accuracy | 0.400 | 4/10 |
| PII item recall | 0.826 | 38/46 items gone from our text (partial 5, missed 3) |
| PII tags precision / recall / F1 (aligned with expected text) | 0.974 / 0.804 / 0.881 | 37 of 46 expected tags matched, 38 tags of ours |
| Anonymized text exact match | 0/10 | not meaningful when the transcripts differ from the key's |
### 2. Our Whisper transcripts, PR #6 code

| Metric | Value | Detail |
|---|---:|---|
| Relevance accuracy | 1.000 | 10 messages |
| Relevance precision / recall / F1 (Event = positive) | 1.000 / 1.000 / 1.000 | tp 8, fp 0, fn 0 |
| Activity type accuracy | 1.000 | 8/8 events, word overlap >= 0.5 |
| Participants accuracy | 1.000 | 10/10 |
| duplicate_of accuracy | 1.000 | 10/10 |
| duplicate detection precision / recall / F1 | 1.000 / 1.000 / 1.000 | |
| pii_count accuracy | 0.700 | 7/10 |
| PII item recall | 0.957 | 44/46 items gone from our text (partial 2, missed 0) |
| PII tags precision / recall / F1 (aligned with expected text) | 0.978 / 0.957 / 0.967 | 44 of 46 expected tags matched, 45 tags of ours |
| Anonymized text exact match | 0/10 | not meaningful when the transcripts differ from the key's |
### 3. Provided transcripts, PR #6 code (for comparison; in-sample)

| Metric | Value | Detail |
|---|---:|---|
| Relevance accuracy | 1.000 | 10 messages |
| Relevance precision / recall / F1 (Event = positive) | 1.000 / 1.000 / 1.000 | tp 8, fp 0, fn 0 |
| Activity type accuracy | 1.000 | 8/8 events, word overlap >= 0.5 |
| Participants accuracy | 1.000 | 10/10 |
| duplicate_of accuracy | 1.000 | 10/10 |
| duplicate detection precision / recall / F1 | 1.000 / 1.000 / 1.000 | |
| pii_count accuracy | 1.000 | 10/10 |
| PII item recall | 1.000 | 46/46 items gone from our text (partial 0, missed 0) |
| PII tags precision / recall / F1 (aligned with expected text) | 1.000 / 1.000 / 1.000 | 46 of 46 expected tags matched, 46 tags of ours |
| Anonymized text exact match | 10/10 | same transcripts as the key, so exact match is meaningful here |
