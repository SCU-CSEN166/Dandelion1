import sys
import argparse
import torch
from load_dataset import prepare_dataset
from summarize import summarize_text

def main():
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available")

    parser = argparse.ArgumentParser(description="Process a WhatsApp dataset and summarize its content.")
    parser.add_argument("-d", "--dataset", type=str, help="Name of the dataset to process, either locally stored or from Huggingface")
    parser.add_argument("-s", "--summarizer", type=str, help="Name of the summarization model to use, either locally or from Huggingface (such as 'facebook/bart-large-cnn')")
    parser.add_argument("-l", "--length", type=int, default=100, help="Minimum length of the summary")
    parser.add_argument("-m", "--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate in the summary")
    parser.add_argument("-b", "--use_blocks", action="store_true")
    parser.add_argument("-o", "--output", type=str, help="File to write the summary to")
    parser.add_argument("--input_out", type=str, help="File to write the input text to")

    args = parser.parse_args()

    if not args.dataset:
        print("Dataset name is required")
        sys.exit(1)

    dataset_name = args.dataset
    dataset = prepare_dataset(dataset_name)

    text_input = "\n".join(dataset["train"]["text"])
    if (args.input_out):
        with open(args.input_out, "w", encoding="utf-8") as f:
            f.write(text_input)

    if (args.summarizer and args.output):
        summary = summarize_text(text_input, args.summarizer, args.length, args.use_blocks, max_new_tokens=args.max_new_tokens)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(summary)

if __name__ == "__main__":
    main()