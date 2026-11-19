"""Speech-recognition word error rate (WER) of a Whisper model on a labeled speech dataset.

Default: Swahili (Kenya) from Google FLEURS. Writes each clip's transcript ("text") next to its reference and the
dataset's other columns; for a dataset with reference summaries (e.g. SLUE-TED "abstract"), score summaries of the
transcripts with eval_summary.py on that CSV.

Run:  python eval/eval_asr.py -m openai/whisper-medium -d edinburghcstr/edacc --config default --split test \
          --ref_col text --accent Kenyan --language english -n 50 -o results/edacc_kenyan_whisper-medium.csv
"""
import argparse
import io
import os
import sys

import pandas as pd
from datasets import Audio, load_dataset
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from transcribe import load_asr, transcribe  # noqa: E402


def load_clip(audio: dict) -> dict:
    import soundfile as sf  # pip install soundfile
    arr, sr = sf.read(io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"], dtype="float32")
    return {"raw": arr.mean(axis=1) if arr.ndim > 1 else arr, "sampling_rate": sr}  # stereo -> mono


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-m", "--model", default="openai/whisper-small", help="e.g. openai/whisper-large-v3")
    p.add_argument("-d", "--dataset", default="google/fleurs")
    p.add_argument("--config", default="sw_ke", help="FLEURS language, e.g. sw_ke (Swahili-Kenya), en_us")
    p.add_argument("--language", default="swahili", help="Language Whisper should transcribe in")
    p.add_argument("--split", default="test")
    p.add_argument("--ref_col", default="transcription")
    p.add_argument("--gen_config", help="Borrow the generation config of this model, for fine-tunes with an outdated one, e.g. openai/whisper-medium")
    p.add_argument("--accent", help="Keep only rows whose 'accent' column contains this, e.g. 'Kenyan' (EdAcc)")
    p.add_argument("--hints", help="Spelling hints for Whisper (see src/transcribe.py)")
    p.add_argument("-n", "--num_samples", type=int, default=50, help="0 = all")
    p.add_argument("-o", "--output", default="results/asr_out.csv")
    args = p.parse_args()

    # streaming: only downloads the clips we use, instead of every split (sw_ke train alone is ~3 GB)
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))  # decoded below, so skipped rows are never decoded

    def keep(r):
        # EdAcc marks unscored segments (read-aloud passages) with this ref; Whisper transcribes them -> pure insertions
        if "IGNORE_TIME_SEGMENT_IN_SCORING" in (r[args.ref_col] or ""):
            return False
        return not args.accent or args.accent.lower() in (r.get("accent") or "").lower()

    rows = list(ds.filter(keep).take(args.num_samples) if args.num_samples else ds.filter(keep))
    hyp = transcribe(load_asr(args.model, args.gen_config), [load_clip(r["audio"]) for r in rows], args.language,
                     args.hints)
    ref = [r[args.ref_col] for r in rows]

    from jiwer import wer  # pip install jiwer
    norm = BasicTextNormalizer()
    print(f"WER: {wer([norm(r) for r in ref], [norm(h) for h in hyp]):.3f}  ({len(ref)} clips, {args.model})")

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("audio", args.ref_col)} for r in rows])
    df.insert(0, "text", hyp)
    df.insert(1, "reference", ref)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
