"""Transcribe voice notes (audio or video files) with Whisper into a table that main.py reads.

Output columns: message_id (1, 2, ... in file-name order), audio_file, transcript. Then run the pipeline on it:
  python src/transcribe.py data/voice_notes/ -o data/voice_notes.csv --hints "Rongai, Nakuru, Kampi ya Moto"
  python src/main.py -d data/voice_notes.csv -o outputs/voice_notes -s <summarizer>

Needs ffmpeg to read audio (or: pip install imageio-ffmpeg).
"""
import argparse
import glob
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
from transformers import GenerationConfig, pipeline

AUDIO_EXTENSIONS = ("*.mp4", "*.m4a", "*.mp3", "*.wav", "*.flac", "*.ogg", "*.opus")


def find_audio_files(paths: list[str]) -> list[str]:
    """Expand folders and glob patterns (notebooks and unmatched shell globs pass "*.mp4" through literally)."""
    found = []
    for p in paths:
        if os.path.isdir(p):
            found += [f for ext in AUDIO_EXTENSIONS for f in glob.glob(os.path.join(p, ext))]
        else:
            found += glob.glob(p)
    return sorted(found)


def load_audio_file(path: str) -> dict:
    """Decode any audio or video file to 16 kHz mono with ffmpeg."""
    exe = shutil.which("ffmpeg")
    if not exe:
        import imageio_ffmpeg  # pip install imageio-ffmpeg: ships an ffmpeg binary when the system has none
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    # the whole file is decoded into RAM (~230 MB per hour of audio); stream it if files get huge
    r = subprocess.run([exe, "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                       capture_output=True)
    if r.returncode:
        raise RuntimeError(f"ffmpeg failed on {path}: {r.stderr.decode(errors='replace')}")
    return {"raw": np.frombuffer(r.stdout, np.float32), "sampling_rate": 16000}


def load_asr(model: str, gen_config: str | None = None):
    """A Whisper pipeline. chunk_length_s: Whisper hears 30 s at a time, so longer audio is cut into chunks.
    gen_config: borrow the generation config of another model, for fine-tunes with an outdated one."""
    asr = pipeline("automatic-speech-recognition", model=model, device_map="auto", chunk_length_s=30)
    if gen_config:
        asr.model.generation_config = GenerationConfig.from_pretrained(gen_config)
    return asr


def transcribe(asr, inputs: list[dict], language: str, hints: str | None = None) -> list[str]:
    """Transcribe decoded audio. hints: names and places (or a text file of them) that Whisper should spell the way
    they are written there ("Girissa" -> "Garissa"); Whisper reads roughly the last 200 tokens."""
    gen_kwargs = {"language": language, "task": "transcribe"}
    if hints:
        text = open(hints, encoding="utf-8").read() if os.path.isfile(hints) else hints
        gen_kwargs["prompt_ids"] = asr.tokenizer.get_prompt_ids(" ".join(text.split()), return_tensors="pt").to(
            asr.model.device)
    return [o["text"].strip() for o in asr(inputs, batch_size=8, generate_kwargs=gen_kwargs)]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("files", nargs="+", help="Audio/video files, folders or glob patterns")
    p.add_argument("-o", "--output", required=True, help="CSV to write (input for main.py -d)")
    p.add_argument("-m", "--model", default="openai/whisper-small", help="e.g. openai/whisper-medium")
    p.add_argument("--language", default="english", help="Language Whisper should transcribe in")
    p.add_argument("--hints", help="Spelling hints: a text file, or a comma-separated list of local names and places")
    p.add_argument("--gen_config", help="Borrow the generation config of this model (for outdated fine-tunes)")
    args = p.parse_args()

    files = find_audio_files(args.files)
    if not files:
        p.error("no audio files matched; check the path (a leading / means the filesystem root)")
    print(f"Transcribing {len(files)} files")
    texts = transcribe(load_asr(args.model, args.gen_config), [load_audio_file(f) for f in files], args.language,
                       args.hints)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pd.DataFrame({"message_id": range(1, len(files) + 1), "audio_file": [os.path.basename(f) for f in files],
                  "transcript": texts}).to_csv(args.output, index=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
