import sys
import argparse
import pandas as pd
import torch
from load_dataset import prepare_dataset
from summarize import summarize_text

def main():
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available")

    parser = argparse.ArgumentParser(description="Process a WhatsApp dataset and/or voice notes and summarize their content.")
    parser.add_argument("-d", "--dataset", type=str, help="Dataset to process: a Hugging Face dataset name, or a local .csv, .tsv, .json, .jsonl or .xlsx file")
    parser.add_argument("--text_col", type=str, help="Column holding the text in a local file or dataset (guessed if not given: text, transcript, message, ...)")
    parser.add_argument("-a", "--audio", nargs="+", help="Audio/video files, folders or globs to transcribe with Whisper (mp4, m4a, mp3, wav, ...). Can be combined with --dataset")
    parser.add_argument("--whisper_model", type=str, default="openai/whisper-small", help="Whisper model for --audio, e.g. openai/whisper-large-v3")
    parser.add_argument("--language", type=str, default="english", help="Language Whisper should transcribe in ('auto' to detect)")
    parser.add_argument("--transcripts_out", type=str, help="CSV file to write the raw (unredacted) transcripts of --audio to")
    parser.add_argument("-s", "--summarizer", type=str, help="Name of the summarization model to use, either locally or from Huggingface (such as 'facebook/bart-large-cnn')")
    parser.add_argument("-l", "--length", type=int, default=100, help="Minimum length of the summary")
    parser.add_argument("-m", "--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate in the summary")
    parser.add_argument("-b", "--use_blocks", action="store_true")
    parser.add_argument("-o", "--output", type=str, help="File to write the summary to")
    parser.add_argument("--input_out", type=str, help="File to write the input text to")

    args = parser.parse_args()

    if not args.dataset and not args.audio:
        print("Give a dataset (-d) or audio files (-a)")
        sys.exit(1)

    audio_texts = None
    if args.audio:
        from audio import transcribe_files
        transcripts = transcribe_files(args.audio, args.whisper_model, args.language)
        audio_texts = [text for _, text in transcripts]
        if args.transcripts_out:
            pd.DataFrame(transcripts, columns=["file", "text"]).to_csv(args.transcripts_out, index=False)

    dataset = prepare_dataset(args.dataset, text_col=args.text_col, extra_texts=audio_texts)

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