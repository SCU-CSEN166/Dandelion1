"""evaluation module for summarization: ROUGE scores and NLI outcome recall"""
import argparse
import json
import os
import re
import pandas as pd
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from redact import scan_batch
from summarize import summarize_text

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]


def load_eval_data(
    dataset: str,
    split: str = "test",
    config: str | None = None,
    text_col: str | None = None,
    ref_col: str | None = None,
    strip_prefix: str | None = None,
) -> pd.DataFrame:
    """Load text and reference columns from local file or Hugging Face dataset."""
    if os.path.isfile(dataset):
        ext = os.path.splitext(dataset)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(dataset, encoding_errors="replace")
        elif ext in (".json", ".jsonl"):
            df = pd.read_json(dataset, lines=ext == ".jsonl")
        elif ext == ".parquet":
            df = pd.read_parquet(dataset)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    else:
        df = load_dataset(dataset, config, split=split).to_pandas()

    cols = {c.lower(): c for c in df.columns}
    text_cands = ["dialogue", "text", "post", "content", "document", "article", "prompt", "input"]
    ref_cands = ["summary", "highlights", "tldr", "reference", "target", "output"]

    t_col = text_col or next((cols[c] for c in text_cands if c in cols), None)
    r_col = ref_col or next((cols[c] for c in ref_cands if c in cols), None)

    if not t_col or not r_col:
        raise SystemExit(f"Could not auto-detect text/ref columns. Available: {list(df.columns)}")

    df = df.dropna(subset=[t_col, r_col]).drop_duplicates(subset=[t_col])
    text = df[t_col].astype(str)
    if strip_prefix:
        text = text.str.removeprefix(strip_prefix)

    return pd.DataFrame({"text": text, "reference": df[r_col].astype(str)})


def split_facts(text: str) -> list[str]:
    """Split reference summary into individual fact statements."""
    facts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 3]
    return facts or [text.strip()]


class NLIChecker:
    """Zero-shot fact verification using an NLI classification model."""

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail_id = labels["entailment"]

    @torch.no_grad()
    def check_entailment(self, premise: str, hypotheses: list[str]) -> list[bool]:
        if not hypotheses:
            return []
        enc = self.tokenizer(
            [premise] * len(hypotheses),
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        return (self.model(**enc).logits.argmax(dim=-1) == self.entail_id).tolist()


def mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate summarization model on dialogue/chat datasets.")
    parser.add_argument("-s", "--summarizer", required=True, help="Hugging Face model name or local path")
    parser.add_argument("-d", "--dataset", default="knkarthick/samsum", help="Dataset name or path (default: knkarthick/samsum)")
    parser.add_argument("--config", help="Dataset configuration or subset name")
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    parser.add_argument("--text_col", help="Column name containing source text")
    parser.add_argument("--ref_col", help="Column name containing reference summary")
    parser.add_argument("--strip_prefix", help="Prefix to remove from inputs")
    parser.add_argument("-n", "--num", "--num_samples", dest="num", type=int, default=50, help="Number of samples (default: 50, 0 = all)")
    parser.add_argument("-l", "--length", type=int, default=10, help="Minimum summary length (default: 10)")
    parser.add_argument("-il", "--input_length", type=int, default=1024, help="Max input tokens before chunking")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--redact", action="store_true", help="Apply PII redaction before summarizing")
    parser.add_argument("--nli", default="facebook/bart-large-mnli", help="NLI model for outcome recall (default: facebook/bart-large-mnli)")
    parser.add_argument("--no_nli", action="store_true", help="Disable NLI outcome recall for fast ROUGE-only eval")
    parser.add_argument("-o", "--output", help="Output JSON path")
    args = parser.parse_args()

    # Load and sample dataset
    df = load_eval_data(args.dataset, args.split, args.config, args.text_col, args.ref_col, args.strip_prefix)
    if args.num and 0 < args.num < len(df):
        df = df.sample(args.num, random_state=args.seed).reset_index(drop=True)

    # Optional PII Redaction
    posts = list(df["text"])
    if args.redact:
        namemap = {"map": {}, "count": 0}
        posts = scan_batch({"text": posts}, namemap)["text"]

    # Initialize scorers
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=True)
    use_nli = not args.no_nli and args.nli and args.nli.lower() != "none"
    nli = NLIChecker(args.nli) if use_nli else None

    examples = []
    print(f"Evaluating '{args.summarizer}' on '{args.dataset}' (n={len(df)}, redact={args.redact}, nli={use_nli})...")

    for i, (post, ref) in enumerate(zip(posts, df["reference"])):
        pred = summarize_text(post, args.summarizer, min_length=args.length, split_overflowing=args.input_length)
        r_scores = scorer.score(ref, pred)

        item = {
            "index": i,
            "input": post,
            "reference": ref,
            "prediction": pred,
            "rouge": {k: {"fmeasure": r_scores[k].fmeasure, "recall": r_scores[k].recall, "precision": r_scores[k].precision} for k in ROUGE_KEYS},
        }

        if nli:
            facts = split_facts(ref)
            hits = nli.check_entailment(pred, facts)
            item["facts"] = [{"fact": f, "found": h} for f, h in zip(facts, hits)]
            item["nli_recall"] = mean(hits)

        examples.append(item)
        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            log_str = f"[{i + 1}/{len(df)}] ROUGE-1 F1={r_scores['rouge1'].fmeasure:.3f}"
            if nli:
                log_str += f"  NLI R={item['nli_recall']:.2f}"
            print(log_str)

    # Aggregate metrics
    mean_rouge = {
        k: {
            "fmeasure": mean([e["rouge"][k]["fmeasure"] for e in examples]),
            "recall": mean([e["rouge"][k]["recall"] for e in examples]),
            "precision": mean([e["rouge"][k]["precision"] for e in examples]),
        }
        for k in ROUGE_KEYS
    }

    results = {
        "model": args.summarizer,
        "dataset": args.dataset,
        "n": len(examples),
        "redact": args.redact,
        "seed": args.seed,
        "mean_rouge": mean_rouge,
        "examples": examples,
    }

    if nli:
        all_hits = [f["found"] for e in examples for f in e.get("facts", [])]
        results["nli_model"] = args.nli
        results["nli_outcome_recall"] = mean(all_hits)
        results["nli_outcome_recall_per_post"] = mean([e.get("nli_recall", 0.0) for e in examples])

    # Resolve output path
    if not args.output:
        data_tag = os.path.splitext(os.path.basename(args.dataset))[0] if os.path.isfile(args.dataset) else f"{args.dataset}_{args.split}"
        clean_name = re.sub(r"[^\w.-]", "_", f"{data_tag}_{args.summarizer}")
        args.output = os.path.join("results", f"{clean_name}.json")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n--- Summary Results ---")
    print(f"ROUGE-1 F1: {mean_rouge['rouge1']['fmeasure']:.4f} (P: {mean_rouge['rouge1']['precision']:.4f}, R: {mean_rouge['rouge1']['recall']:.4f})")
    print(f"ROUGE-2 F1: {mean_rouge['rouge2']['fmeasure']:.4f} (P: {mean_rouge['rouge2']['precision']:.4f}, R: {mean_rouge['rouge2']['recall']:.4f})")
    print(f"ROUGE-L F1: {mean_rouge['rougeL']['fmeasure']:.4f} (P: {mean_rouge['rougeL']['precision']:.4f}, R: {mean_rouge['rougeL']['recall']:.4f})")
    if nli:
        print(f"NLI Outcome Recall: {results['nli_outcome_recall']:.4f}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
