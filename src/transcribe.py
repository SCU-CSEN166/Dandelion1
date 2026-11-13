import argparse
import glob
import os
import io
import shutil
import subprocess
import numpy as np
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
p.add_argument("-s", "--summarizer", help="Also redact + summarize the transcripts, e.g. google/flan-t5-large (datasets: also scored by eval.py)")
p.add_argument("--sum_ref_col", default="abstract", help="Column holding the reference summary (SLUE-TED: abstract)")
p.add_argument("--files", nargs="+", help="Transcribe local audio/video files instead of a dataset (mp4, m4a, mp3, wav...; needs ffmpeg or pip install imageio-ffmpeg). No WER.")
args = p.parse_args()
if args.files:
    # expand globs/folders here: notebooks and unmatched shell globs pass "*.mp4" through literally
    exts = ("*.mp4", "*.m4a", "*.mp3", "*.wav", "*.flac")
    args.files = sorted(f for x in args.files
                        for f in ([g for e in exts for g in glob.glob(os.path.join(x, e))] if os.path.isdir(x) else glob.glob(x)))
    if not args.files:
        p.error("--files matched nothing; check the path (a leading / means the filesystem root)")
    print(f"Transcribing {len(args.files)} files")

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

def load_file(path):
    # ffmpeg pulls the audio track out of any container (mp4, m4a, ...); imageio-ffmpeg ships a binary when the system has none
    exe = shutil.which("ffmpeg")
    if not exe:
        import imageio_ffmpeg  # pip install imageio-ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    # ponytail: whole file decoded into RAM (~230 MB per hour of audio), stream it if files get huge
    r = subprocess.run([exe, "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                       capture_output=True)
    if r.returncode:
        raise RuntimeError(f"ffmpeg failed on {path}: {r.stderr.decode(errors='replace')}")
    return {"raw": np.frombuffer(r.stdout, np.float32), "sampling_rate": 16000}

inputs = [load_file(f) for f in args.files] if args.files else [load_audio(r["audio"]) for r in rows]
out = asr(inputs, batch_size=8, generate_kwargs={"language": args.language, "task": "transcribe"})
hyp = [o["text"].strip() for o in out]

if args.files:
    df = pd.DataFrame({"text": hyp, "file": args.files})
    if args.summarizer:
        # your own files have no reference summaries, so redact + summarize only, no scoring
        import gc, inspect, torch
        from redact import scan_batch
        from summarize import summarize_text
        del asr; gc.collect(); torch.cuda.empty_cache()  # free Whisper's VRAM for the summarizer
        if len(inspect.signature(scan_batch).parameters) > 1:  # newer redact.py: names -> Person0, Person1, ...
            redacted = scan_batch({"text": hyp}, {"map": {}, "count": 0})["text"]
        else:
            redacted = scan_batch({"text": hyp})["text"]
        df["redacted"] = redacted
        df["summary"] = [summarize_text(t, args.summarizer, 10, 1024) for t in redacted]  # 1024: long talks go in blocks
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

if args.summarizer and not args.files:
    # ponytail: reuses eval.py's CLI instead of refactoring it into functions
    import sys, gc, torch, eval as ev
    del asr; gc.collect(); torch.cuda.empty_cache()  # free Whisper's VRAM for the summarizer
    sys.argv = ["eval.py", "-d", args.output, "-s", args.summarizer,
                "--text_col", "text", "--ref_col", args.sum_ref_col, "-n", "0"]
    ev.main()
