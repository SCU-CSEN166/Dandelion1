# Hidden test set: baseline results (current pipeline, before any fixes)

Pipeline version: `main` at f9cf794. Redaction: `src/redact.py`. Summarizer: Qwen/Qwen3-0.6B, run on CPU.
Input: 10 voice-note transcripts (hidden test set). The transcripts, labels and generated report are not included
here, because they are the hidden test data.

## Redaction (PII recall)
Labels were made by me (Ali) by reading the 10 transcripts: 46 personal items in total
(21 names, 10 phone numbers, 15 place or facility names). Scored with `src/pii_eval.py`.

| Group | Items | Caught | Recall |
|---|---:|---:|---:|
| Names | 21 | 0 | 0.00 |
| Phone numbers | 10 | 0 | 0.00 |
| Places | 15 | 0 | 0.00 |
| All | 46 | 0 | 0.00 |

Ordinary words kept: 0.998 (one over-redaction: a normal word followed by a colon was treated as a speaker name).

Why: the phone pattern expects a US-style layout, so Kenyan numbers like `07xx xxx xxx` do not match; the name pattern
only fires on `Name:` speaker labels, which voice notes do not have; nothing detects places.

## Summary (Qwen3-0.6B report from all 10 transcripts)
- A person's name appeared in the report, so redaction leaked into the output.
- Several facts were wrong: a count of women served was reported as a count of sessions, "referred" became "refused",
  the currency was wrong, and a percentage was invented.
- Only about 8 of the 10 messages were covered. Several outreach events were missing.
- Small talk (transport trouble, a lost phone charger) was listed as a field blocker.
- The model's internal reasoning (a `<think>` block) was included in the output.
- Repeated reports of the same event were not clearly merged.

## Caveats
10 messages, one run, a very small model, and labels made by one person. Treat these as rough baseline numbers.

## Planned fixes (not done yet)
1. Kenyan phone pattern (`07xx xxx xxx`, `+254...`).
2. Name detection from titles (Sister, Mrs., Mama...) and introductions ("this is ...").
3. Kenyan place-name list built from other FarmerChat rows, not the labeled dev set.
4. Strip the `<think>` block from summarizer output.
