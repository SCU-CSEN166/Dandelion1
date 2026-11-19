# Results

`old/` keeps the outputs and metrics of earlier versions of the code, as a record of the project's progress

| Files in `old/` | What they are | Made by (at the time) |
|---|---|---|
| `hidden_test_*` | Pipeline outputs and metrics on the TA's 10 hidden-test messages (provided and Whisper transcripts) | `main.py`, `eval_ta_table.py` |
| `edacc_kenyan_*.csv` | Speech-recognition runs on Kenyan-accent English (EdAcc), with WER | `transcribe.py` |
| `samsum_*.json`, `knkarthick_samsum_*.json`, `tldr_*.json`, `CarperAI_*.json` | Summary evaluations (ROUGE, NLI fact recall) | `eval_summaries.py`, `eval.py` |
| `pii_eval*.json` | Redaction recall on labeled PII data | `pii_eval.py` |
| `dandelion*.csv` | Early pipeline output tables | `main.py` |

The current code writes its outputs to the folder given with `main.py -o` (the git-ignored `outputs/` by
convention)