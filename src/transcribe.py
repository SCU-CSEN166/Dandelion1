import argparse
import io
import soundfile as sf
import pandas as pd
from datasets import Audio, load_dataset
from jiwer import wer
from transformers import GenerationConfig, pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

p = argparse.ArgumentParser()
p.add_argument("-m", "--model", default="openai/whisper-small", help="e.g. openai/whisper-large-v3")
p.add_argument("-d", "--dataset", default="google/fleurs")
p.add_argument("--config", default="sw_ke", help="FLEURS language, e.g. sw_ke (Swahili-Kenya), en_us")
p.add_argument("--language", default="swahili", help="Language Whisper should transcribe in")
p.add_argument("--split", default="test")
p.add_argument("--ref_col", default="transcription")
p.add_argument("--gen_config", help="Borrow generation config from this model, for fine-tunes with an outdated one, e.g. openai/whisper-medium")
p.add_argument("--accent", help="Keep only rows whose 'accent' column contains this, e.g. 'Kenyan' (EdAcc)")
p.add_argument("-n", "--num_samples", type=int, default=50, help="0 = all")
p.add_argument("-o", "--output", default="results/asr_out.csv")
p.add_argument("-s", "--summarizer", help="Also redact + summarize + score the transcripts with eval.py, e.g. google/flan-t5-large")
p.add_argument("--sum_ref_col", default="abstract", help="Column holding the reference summary (SLUE-TED: abstract)")
p.add_argument("--files", nargs="+", help="Transcribe local audio/video files instead of a dataset (mp4, m4a, mp3, wav...; needs ffmpeg). No WER.")
args = p.parse_args()
if args.files and args.summarizer:
    p.error("-s needs reference summaries, which --files doesn't have")

if not args.files:
    # streaming: only downloads the clips we use, instead of every split (sw_ke train alone is ~3 GB)
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    # decode=False first: we decode ourselves below, and the filter then never decodes audio for skipped rows
    ds = ds.cast_column("audio", Audio(decode=False))

    def keep(r):
        # EdAcc marks unscored segments (read-aloud passages) with this ref; Whisper transcribes them -> pure insertions
        if "IGNORE_TIME_SEGMENT_IN_SCORING" in (r[args.ref_col] or ""):
            return False
        return not args.accent or args.accent.lower() in (r.get("accent") or "").lower()

    ds = ds.filter(keep)  # one filter only: chaining filters on a stream breaks in this datasets version
    rows = list(ds.take(args.num_samples) if args.num_samples else ds)

# chunk_length_s: Whisper only hears 30 s at a time; needed for long audio (SLUE-TED talks), no-op for short clips
asr = pipeline("automatic-speech-recognition", model=args.model, device_map="auto", chunk_length_s=30)
if args.gen_config:  # older fine-tunes lack the language/task tables Whisper's generate() needs
    asr.model.generation_config = GenerationConfig.from_pretrained(args.gen_config)
def load_audio(a):
    arr, sr = sf.read(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"], dtype="float32")
    return {"raw": arr.mean(axis=1) if arr.ndim > 1 else arr, "sampling_rate": sr}  # stereo -> mono; pipeline resamples

# a file path makes the pipeline decode with ffmpeg, so any format ffmpeg reads works (incl. video)
inputs = args.files or [load_audio(r["audio"]) for r in rows]
out = asr(inputs, batch_size=8, generate_kwargs={"language": args.language, "task": "transcribe"})
hyp = [o["text"].strip() for o in out]

if args.files:
    df = pd.DataFrame({"text": hyp, "file": args.files})
else:
    ref = [r[args.ref_col] for r in rows]
    norm = BasicTextNormalizer()
    print(f"WER: {wer([norm(r) for r in ref], [norm(h) for h in hyp]):.3f}  ({len(ref)} clips, {args.model})")

    # keep the dataset's other columns too (e.g. SLUE-TED "abstract" = reference summary for eval.py)
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("audio", args.ref_col)} for r in rows])
    df.insert(0, "text", hyp)
    df.insert(1, "reference", ref)
df.to_csv(args.output, index=False)
print(f"Saved {args.output}")

if args.summarizer:
    # ponytail: reuses eval.py's CLI instead of refactoring it into functions
    import sys, gc, torch, eval as ev
    del asr; gc.collect(); torch.cuda.empty_cache()  # free Whisper's VRAM for the summarizer
    sys.argv = ["eval.py", "-d", args.output, "-s", args.summarizer,
                "--text_col", "text", "--ref_col", args.sum_ref_col, "-n", "0"]
    ev.main()
