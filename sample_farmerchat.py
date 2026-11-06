"""Sample Kenyan farmer queries from DigiGreen FarmerChat for hand labeling.

Only the `query` column is used (the `response` column is written by a bot).
Output is a JSONL file with empty label fields to fill in by hand:
  pii_labels : list of {"text": "...", "type": "NAME|LOCATION|PHONE|EMAIL|OTHER"}
  key_info   : short note of what a good report should keep from the message
"""
import argparse
import json
import random

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Sample FarmerChat Kenya queries for labeling")
    parser.add_argument("-n", "--num", type=int, default=50, help="Number of queries to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=str, default="data/farmerchat_kenya_sample.jsonl")
    args = parser.parse_args()

    ds = load_dataset("DigiGreen/farmerchat-queries", "kenya", split="train")
    print(f"Loaded {len(ds)} Kenya rows. Columns: {ds.column_names}")

    random.seed(args.seed)
    indices = random.sample(range(len(ds)), args.num)

    with open(args.output, "w", encoding="utf-8") as f:
        for i in indices:
            row = ds[i]
            record = {
                "row_index": i,
                "query": row["query"],
                "county": row.get("user_geo_level2", ""),
                "district": row.get("user_geo_level3", ""),
                "pii_labels": [],
                "key_info": "",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {args.num} queries to {args.output}")


if __name__ == "__main__":
    main()
