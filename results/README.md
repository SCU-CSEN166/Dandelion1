# Results

`old/` keeps the outputs and metrics of earlier versions of the code, as a record of the project's progress

| Files in `old/` | What they are | Made by (at the time) |
|---|---|---|
| `hidden_test_*` | Pipeline outputs and metrics on the TA's 10 hidden-test messages (provided and Whisper transcripts) | `main.py`, `eval_ta_table.py` |
| `edacc_kenyan_*.csv` | Speech-recognition runs on Kenyan-accent English (EdAcc), with WER | `transcribe.py` |
| `samsum_*.json`, `knkarthick_samsum_*.json`, `tldr_*.json`, `CarperAI_*.json` | Summary evaluations (ROUGE, NLI fact recall) | `eval_summaries.py`, `eval.py` |
| `pii_eval*.json` | Redaction recall on labeled PII data | `pii_eval.py` |
| `dandelion*.csv` | Early pipeline output tables | `main.py` |

| Folder | What it is |
|---|---|
| `hidden_test_7fc8a30/` | Hidden-test tables and metrics right after the refactor (commit 7fc8a30): relevance 9/10, activity 8 → 4/8 |
| `hidden_test_fixed/` | The same after the fixes (Event threshold 0.99, topic rule before the classifier): back to 10/10 and 8/8 |

Each holds `messages_ta.csv` / `messages_whisper.csv` (main.py on the TA's and on our Whisper transcripts, without
`-s`) and their `metrics_*.md` from `eval/eval_hidden_test.py --metrics_out`.

## Report quality on the hidden test (`hidden_test_reports/`)

`report.md` from three summarizer setups on the TA transcripts, each scored by `eval/eval_summary.py` (judge:
Qwen/Qwen2.5-7B-Instruct; key facts: the 6 distinct Events of the TA key). Run on WAVE (GPU, 32 GB RAM) on 2026-11-20.

| Folder | Summarizer | Faithfulness | Completeness | Conciseness | Sentences |
|---|---|---:|---:|---:|---:|
| `0.6b_with_chitchat/` | Qwen3-0.6B, all messages | 0.375 | 0.50 (3/6) | 0.50 | 8 |
| `0.6b_filter_relevant/` | Qwen3-0.6B, `--filter_relevant` | 0.50 | 0.50 (3/6) | 0.30 | 10 |
| `7b_filter_relevant/` | Qwen2.5-7B-Instruct, `--filter_relevant` | 0.70 | 0.83 (5/6) | 0.30 | 20 |

What the reports got wrong (judge's "unsupported" sentences, checked by hand against the transcripts):
- 0.6B with chit-chat: a matatu breakdown from a chit-chat apology (message 3) is put into the menstrual hygiene
  session; "come back next month" is moved from the family planning report to the wrong session; a "delayed return
  due to rain" is invented. The judge also rejected one supported sentence (message 10 does ask for more chairs and
  water), so the true faithfulness is about 4/8.
- 0.6B filtered: no chit-chat any more, but it still covers 3 of 6 events and invents follow-ups ("connected women to
  counseling"). The judge rejected "60 attendees" at the menstrual hygiene session, which message 5 supports.
- 7B filtered: five of the six Events, each with its correct head count; the slips are smaller ("long queues at both
  screening camps" when there was one camp reported twice; "rainfall affected attendance" when the message says the
  turnout was very good despite the rain; reports "sent" that nobody mentions).
- All three leave out the table banking meeting (message 4, 25 members).

Caveats: the 7B summarizer is scored by the same 7B model, which may favour its own writing. Conciseness counts
follow-up sentences as filler because the key facts are Events only. The 7B report's title line was scored as a
sentence. With 6 facts and 8-20 sentences these are rough numbers. The judge's mistakes above are why
`summary_eval.json` keeps every verdict for a person to check.

The current code writes its outputs to the folder given with `main.py -o` (the git-ignored `outputs/` by
convention)