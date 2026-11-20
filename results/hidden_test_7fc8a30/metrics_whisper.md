### Hidden test, Whisper transcripts, commit 7fc8a30 (after refactor, before fixes)

| Metric | Value | Detail |
|---|---:|---|
| Relevance accuracy | 0.800 | 10 messages |
| Relevance precision / recall / F1 (Event = positive) | 1.000 / 0.750 / 0.857 | tp 6, fp 0, fn 2 |
| Activity type accuracy | 0.500 | 4/8 events, word overlap >= 0.5 |
| Participants accuracy | 0.800 | 8/10 |
| duplicate_of accuracy | 0.800 | 8/10 |
| duplicate detection precision / recall / F1 | n/a / 0.000 / n/a | |
| pii_count accuracy | 0.700 | 7/10 |
| PII item recall | 0.957 | 44/46 items gone from our text (partial 2, missed 0) |
| PII tags precision / recall / F1 (aligned with expected text) | 0.978 / 0.957 / 0.967 | 44 of 46 expected tags matched, 45 tags of ours |
| Anonymized text exact match | 0/10 | not meaningful when the transcripts differ from the key's |
