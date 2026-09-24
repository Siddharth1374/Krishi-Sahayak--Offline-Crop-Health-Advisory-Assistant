"""
speech/asr.py
--------------
Offline speech-to-text so illiterate farmers can just speak their question.

Uses faster-whisper (CTranslate2 backend) for a simple portable CPU/NPU path.
For maximum on-device performance on Snapdragon, swap in the AI Hub-optimized
Whisper ONNX build (qai_hub_models.models.whisper_*) run through the QNN EP,
mirroring the pattern used in vision/classify.py.
"""

import os

try:
    from faster_whisper import WhisperModel
except ImportError as e:
    raise SystemExit("Install faster-whisper: pip install faster-whisper") from e

MODEL_SIZE = os.environ.get("KS_ASR_MODEL", "small")  # tiny/base/small work offline well
DEVICE = os.environ.get("KS_ASR_DEVICE", "cpu")        # "cpu" is safe/portable default
COMPUTE_TYPE = os.environ.get("KS_ASR_COMPUTE", "int8")  # quantized for speed


class SpeechRecognizer:
    def __init__(self):
        print(f"[asr] Loading Whisper '{MODEL_SIZE}' ({DEVICE}, {COMPUTE_TYPE}) ...")
        self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """language: ISO code (e.g. 'hi' for Hindi) or None to auto-detect."""
        segments, info = self.model.transcribe(audio_path, language=language, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments)
        print(f"[asr] Detected language: {info.language} (p={info.language_probability:.2f})")
        return text.strip()


def record_from_mic(duration_sec: int = 6, out_path: str = "data/tmp_input.wav") -> str:
    """Simple mic capture helper (optional, for a live demo)."""
    import sounddevice as sd
    import numpy as np
    import wave

    samplerate = 16000
    print(f"[asr] Recording {duration_sec}s from microphone...")
    audio = sd.rec(int(duration_sec * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    return out_path


if __name__ == "__main__":
    import sys

    recognizer = SpeechRecognizer()
    audio_file = sys.argv[1] if len(sys.argv) > 1 else record_from_mic()
    print("Transcript:", recognizer.transcribe(audio_file))
