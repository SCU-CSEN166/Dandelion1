import os
import re
import json
import argparse
import pandas as pd
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from redact import scan_batch
from summarize import summarize_text

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]
TEXT_COLS = ["text", "story", "document", "article", "dialogue", "post", "content", "prompt", "source", "input"]
REF_COLS = ["summary", "highlights", "tldr", "abstract", "reference", "chosen", "target", "output"]


def read_table(dataset: str, config: str | None, split: str) -> pd.DataFrame:
    if os.path.isfile(dataset):
        ext = os.path.splitext(dataset)[1].lower()
        if ext == ".csv":
            return pd.read_csv(dataset, encoding_errors="replace")
        if ext in (".json", ".jsonl"):
            return pd.read_json(dataset, lines=ext == ".jsonl")
        if ext == ".parquet":
            return pd.read_parquet(dataset)
        raise ValueError(f"Unsupported file type: {ext} (use .csv, .json, .jsonl, or .parquet)")
    return load_dataset(dataset, config, split=split).to_pandas()


def pick_col(df: pd.DataFrame, given: str | None, candidates: list[str], kind: str) -> str:
    if given:
        if given not in df.columns:
            raise SystemExit(f"Column '{given}' not found. Available: {list(df.columns)}")
        return given
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    raise SystemExit(f"Couldn't guess the {kind} column. Pass --{kind}_col. Available: {list(df.columns)}")


def load_eval_data(args) -> pd.DataFrame:
    df = read_table(args.dataset, args.config, args.split)
    text_col = pick_col(df, args.text_col, TEXT_COLS, "text")
    ref_col = pick_col(df, args.ref_col, REF_COLS, "ref")
    print(f"Using text column '{text_col}' and reference column '{ref_col}'")
    df = df.dropna(subset=[text_col, ref_col]).drop_duplicates(text_col)
    text = df[text_col].astype(str)
    if args.strip_prefix:
        text = text.str.removeprefix(args.strip_prefix)
    return pd.DataFrame({"text": text, "reference": df[ref_col].astype(str)})


def split_facts(text: str) -> list[str]:
    facts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 3]
    return facts or [text.strip()]


class NLI:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail_id = labels["entailment"]

    @torch.no_grad()
    def entailed(self, premise: str, hypotheses: list[str]) -> list[bool]:
        enc = self.tokenizer([premise] * len(hypotheses), hypotheses, return_tensors="pt",
                             padding=True, truncation=True).to(self.device)
        return (self.model(**enc).logits.argmax(-1) == self.entail_id).tolist()


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    p = argparse.ArgumentParser(description="Evaluate summarization recall against reference summaries.")
    p.add_argument("-s", "--summarizer", required=True, help="Summarization model, e.g. google/flan-t5-large")
    p.add_argument("-d", "--dataset", required=True, help="Hugging Face dataset name or local .csv/.json/.jsonl/.parquet file")
    p.add_argument("--config", help="Hugging Face dataset config/subset name, if it has one")
    p.add_argument("--split", default="test", help="Hugging Face split (ignored for local files)")
    p.add_argument("--text_col", help="Column with the text to summarize (auto-detected if omitted)")
    p.add_argument("--ref_col", help="Column with the reference summary (auto-detected if omitted)")
    p.add_argument("--strip_prefix", help="Text to remove from the start of every input, e.g. an instruction")
    p.add_argument("-n", "--num_samples", type=int, default=200, help="Posts to evaluate (0 = all)")
    p.add_argument("-l", "--length", type=int, default=10, help="Minimum summary length (keep short; refs are 1-2 sentences)")
    p.add_argument("-il", "--input_length", type=int, default=1024, help="Max input tokens; longer posts are summarized in blocks")
    p.add_argument("--nli", default="facebook/bart-large-mnli", help="NLI model for outcome recall")
    p.add_argument("-o", "--output", help="Output JSON path (default: dataset_outputs/<dataset>_<split>_<summarizer>.json)")
    args = p.parse_args()
    if not args.output:
        if os.path.isfile(args.dataset):
            data_name = os.path.splitext(os.path.basename(args.dataset))[0]
        else:
            data_name = f"{args.dataset}_{args.split}"
        name = re.sub(r"[^\w.-]", "_", f"{data_name}_{args.summarizer}")
        args.output = os.path.join("dataset_outputs", f"{name}.json")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    df = load_eval_data(args)
    if args.num_samples:
        df = df.sample(min(args.num_samples, len(df)), random_state=0)

    posts = scan_batch({"text": list(df.text)}, bare_senders=True)["text"]
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=True)
    nli = NLI(args.nli)

    examples = []
    for i, (post, ref) in enumerate(zip(posts, df.reference)):
        pred = summarize_text(post, args.summarizer, args.length, args.input_length)
        rouge = scorer.score(ref, pred)
        facts = split_facts(ref)
        hits = nli.entailed(pred, facts)
        examples.append({
            "post": post,
            "reference": ref,
            "prediction": pred,
            "rouge": {k: {"recall": rouge[k].recall, "precision": rouge[k].precision} for k in ROUGE_KEYS},
            "facts": [{"fact": f, "found": h} for f, h in zip(facts, hits)],
            "nli_recall": mean(hits),
        })
        print(f"[{i + 1}/{len(df)}] rouge1 R={rouge['rouge1'].recall:.2f}  nli R={mean(hits):.2f}")

    all_hits = [f["found"] for e in examples for f in e["facts"]]
    results = {
        "summarizer": args.summarizer,
        "nli_model": args.nli,
        "dataset": args.dataset if os.path.isfile(args.dataset) else f"{args.dataset}:{args.split}",
        "n": len(examples),
        "rouge": {k: {m: mean([e["rouge"][k][m] for e in examples]) for m in ("recall", "precision")}
                  for k in ROUGE_KEYS},
        "nli_outcome_recall": mean(all_hits),           # facts found / total facts
        "nli_outcome_recall_per_post": mean([e["nli_recall"] for e in examples]),
        "examples": examples,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in results.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main();