### Hidden test, Whisper transcripts, after fixes (threshold 0.99, topic rule first)

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
