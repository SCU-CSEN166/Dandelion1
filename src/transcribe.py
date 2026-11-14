"""Speech-to-text with Whisper-medium (best on EdAcc Kenyan dev: WER 0.219 vs small 0.241, large-v3 0.228).

    from transcribe import transcribe, unload
    results = transcribe(["dandelion_audio/"])   # -> [(path, text), ...]
    unload()                                     # free GPU memory before the summarizer
"""
import gc
import glob
import os
import shutil
import subprocess
from functools import cache

import numpy as np
import torch
from transformers import pipeline

MODEL = "openai/whisper-medium"
AUDIO_EXTS = ("*.mp4", "*.m4a", "*.mp3", "*.wav", "*.flac", "*.ogg", "*.opus")
SAMPLE_RATE = 16000


@cache
def _asr():
    # chunk_length_s: Whisper only hears 30 s at a time; needed for long recordings, no-op for short voice notes
    return pipeline("automatic-speech-recognition", model=MODEL, device_map="auto", chunk_length_s=30)


def find_audio(paths: list[str]) -> list[str]:
    """Expand folders and globs (e.g. "dandelion_audio/" or "audio/*.mp4") into a sorted list of files."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += [f for e in AUDIO_EXTS for f in glob.glob(os.path.join(p, e))]
        else:
            files += glob.glob(p)
    if not files:
        raise FileNotFoundError(f"No audio files found in {paths} (a leading / means the filesystem root)")
    return sorted(files)


def load_audio(path: str) -> dict:
    """Decode any audio/video file to 16 kHz mono float32 via ffmpeg (bundled by imageio-ffmpeg if the system has none)."""
    exe = shutil.which("ffmpeg")
    if not exe:
        import imageio_ffmpeg  # pip install imageio-ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run([exe, "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
                       capture_output=True)
    if r.returncode:
        raise RuntimeError(f"ffmpeg failed on {path}: {r.stderr.decode(errors='replace')}")
    return {"raw": np.frombuffer(r.stdout, np.float32), "sampling_rate": SAMPLE_RATE}


def transcribe(paths: list[str], language: str = "english") -> list[tuple[str, str]]:
    """Transcribe audio/video files, folders or globs. Returns [(file, text), ...] in sorted file order."""
    files = find_audio(paths)
    print(f"Transcribing {len(files)} files with {MODEL}")
    out = _asr()([load_audio(f) for f in files], batch_size=8,
                 generate_kwargs={"language": language, "task": "transcribe"})
    return [(f, o["text"].strip()) for f, o in zip(files, out)]


def unload():
    """Drop the Whisper model and free its GPU memory."""
    _asr.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
