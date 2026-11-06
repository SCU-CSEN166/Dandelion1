"""Score the summarizer on SAMSum (messenger-style dialogues with human summaries).

Example:
  python eval_summaries.py -s google/flan-t5-large -n 50
  python eval_summaries.py -s google/flan-t5-large -n 50 --redact
"""
import argparse
import json
import os
import random
from functools import lru_cache

import summarize  # import torch/transformers before datasets (avoids a native-library crash on macOS)
from datasets import load_dataset
from rouge_score import rouge_scorer

# summarize_text() reloads the model on every call. Cache the loads so a
# multi-example evaluation doesn't reload the model each time.
summarize.AutoTokenizer.from_pretrained = lru_cache(maxsize=None)(summarize.AutoTokenizer.from_pretrained)
summarize.AutoModelForSeq2SeqLM.from_pretrained = lru_cache(maxsize=None)(summarize.AutoModelForSeq2SeqLM.from_pretrained)


def main():
    parser = argparse.ArgumentParser(description="Evaluate summarization on SAMSum")
    parser.add_argument("-s", "--summarizer", type=str, required=True)
    parser.add_argument("-n", "--num", type=int, default=50)
    parser.add_argument("-l", "--length", type=int, default=10, help="Minimum summary length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--redact", action="store_true", help="Run our regex redaction before summarizing")
    parser.add_argument("-o", "--output", type=str, default="results/samsum_eval.json")
    args = parser.parse_args()

    ds = load_dataset("knkarthick/samsum", split="test")
    random.seed(args.seed)
    indices = random.sample(range(len(ds)), args.num)
    dialogues = [ds[i]["dialogue"] for i in indices]
    references = [ds[i]["summary"] for i in indices]

    if args.redact:
        from redact import scan_batch
        dialogues = scan_batch({"text": dialogues})["text"]

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    examples = []

    for idx, dialogue, reference in zip(indices, dialogues, references):
        prediction = summarize.summarize_text(dialogue, args.summarizer, args.length, 0)
        scores = scorer.score(reference, prediction)
        for key in totals:
            totals[key] += scores[key].fmeasure
        examples.append({"index": idx, "input": dialogue, "reference": reference, "prediction": prediction,
                         "rougeL": scores["rougeL"].fmeasure})

    means = {key: round(value / args.num, 4) for key, value in totals.items()}
    print(f"Model: {args.summarizer}  n={args.num}  redact={args.redact}")
    print(f"ROUGE-1 {means['rouge1']}  ROUGE-2 {means['rouge2']}  ROUGE-L {means['rougeL']}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"model": args.summarizer, "n": args.num, "redact": args.redact, "seed": args.seed,
                   "mean_scores": means, "examples": examples}, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
