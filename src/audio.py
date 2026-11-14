"""Speech-to-text for local audio and video files (mp4, m4a, mp3, wav, ...) using Whisper.

ffmpeg pulls the audio track out of any container. If the system has no ffmpeg, install the bundled one:
  pip install imageio-ffmpeg
"""
import gc
import glob
import os
import shutil
import subprocess

import numpy as np

AUDIO_EXTENSIONS = (".mp4", ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".webm", ".aac", ".mov")


def expand_audio_paths(paths: list[str]) -> list[str]:
    """Turn files, folders and globs into a sorted list of audio/video files."""
    found = []
    for item in paths:
        if os.path.isdir(item):
            for name in os.listdir(item):
                if name.lower().endswith(AUDIO_EXTENSIONS):
                    found.append(os.path.join(item, name))
        else:
            found.extend(glob.glob(item))
    found = sorted({f for f in found if os.path.isfile(f)})
    if not found:
        raise FileNotFoundError(f"No audio files found for: {paths}")
    return found


def load_audio_file(path: str) -> dict:
    """Decode any audio/video file to 16 kHz mono float32, the format Whisper expects."""
    exe = shutil.which("ffmpeg")
    if not exe:
        import imageio_ffmpeg  # pip install imageio-ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run([exe, "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                            capture_output=True)
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed on {path}: {result.stderr.decode(errors='replace')}")
    return {"raw": np.frombuffer(result.stdout, np.float32), "sampling_rate": 16000}


def transcribe_files(paths: list[str], model: str = "openai/whisper-small", language: str | None = "english",
                     batch_size: int = 8) -> list[tuple[str, str]]:
    """Transcribe audio files. Returns [(file, transcript), ...] in sorted file order."""
    from transformers import pipeline  # imported here so loading text data does not need the audio stack

    files = expand_audio_paths(paths)
    print(f"Transcribing {len(files)} audio file(s) with {model}")
    try:
        asr = pipeline("automatic-speech-recognition", model=model, device_map="auto", chunk_length_s=30)
    except ValueError:  # device_map needs the 'accelerate' package; fall back to the default device without it
        asr = pipeline("automatic-speech-recognition", model=model, chunk_length_s=30)
    generate_kwargs = {"task": "transcribe"}
    if language and language.lower() != "auto":
        generate_kwargs["language"] = language
    out = asr([load_audio_file(f) for f in files], batch_size=batch_size, generate_kwargs=generate_kwargs)
    transcripts = [o["text"].strip() for o in out]
    del asr
    gc.collect()  # free Whisper's memory before the summarizer loads
    return list(zip(files, transcripts))
