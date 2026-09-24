"""
speech/tts.py
--------------
Offline text-to-speech so the advisor's answer can be spoken aloud.

Default: pyttsx3 (fully offline, works everywhere, but limited language/voice
quality). For real local-language support (Hindi, Bhojpuri, Marathi, etc.),
swap this for Meta's MMS-TTS models (open-source, 1000+ languages) exported
via AI Hub / ONNX for NPU playback.
"""

import os

try:
    import pyttsx3
except ImportError as e:
    raise SystemExit("Install pyttsx3: pip install pyttsx3") from e


class Speaker:
    def __init__(self, rate: int = 160, voice_hint: str | None = None):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        if voice_hint:
            for voice in self.engine.getProperty("voices"):
                if voice_hint.lower() in voice.name.lower():
                    self.engine.setProperty("voice", voice.id)
                    break

    def speak(self, text: str):
        self.engine.say(text)
        self.engine.runAndWait()

    def save_to_file(self, text: str, out_path: str = "data/tmp_output.mp3"):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.engine.save_to_file(text, out_path)
        self.engine.runAndWait()
        return out_path


if __name__ == "__main__":
    import sys

    speaker = Speaker()
    text = " ".join(sys.argv[1:]) or "Your tomato plant shows signs of early blight."
    speaker.speak(text)
