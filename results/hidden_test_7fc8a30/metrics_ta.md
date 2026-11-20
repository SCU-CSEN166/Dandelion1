### Hidden test, TA transcripts, commit 7fc8a30 (after refactor, before fixes)

| Metric | Value | Detail |
|---|---:|---|
| Relevance accuracy | 0.900 | 10 messages |
| Relevance precision / recall / F1 (Event = positive) | 1.000 / 0.875 / 0.933 | tp 7, fp 0, fn 1 |
| Activity type accuracy | 0.500 | 4/8 events, word overlap >= 0.5 |
| Participants accuracy | 0.900 | 9/10 |
| duplicate_of accuracy | 0.800 | 8/10 |
| duplicate detection precision / recall / F1 | n/a / 0.000 / n/a | |
| pii_count accuracy | 1.000 | 10/10 |
| PII item recall | 1.000 | 46/46 items gone from our text (partial 0, missed 0) |
| PII tags precision / recall / F1 (aligned with expected text) | 1.000 / 1.000 / 1.000 | 46 of 46 expected tags matched, 46 tags of ours |
| Anonymized text exact match | 10/10 | not meaningful when the transcripts differ from the key's |
