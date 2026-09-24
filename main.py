"""
main.py
-------
Krishi Sahayak — full offline pipeline:

    photo -> vision classifier -> diagnosis
    voice question -> ASR -> text question
    diagnosis + question + RAG context -> local LLM -> advice
    advice -> TTS -> spoken answer

Two ways to run:
    python main.py --cli path/to/leaf.jpg "How much fungicide should I use?"
    python main.py --ui              # launches a local Gradio web UI (offline, localhost only)
"""

import argparse
import sys

from vision.classify import CropDiseaseClassifier
from llm.advisor import CropAdvisor
from speech.asr import SpeechRecognizer
from speech.tts import Speaker


class KrishiSahayak:
    def __init__(self, enable_voice: bool = True):
        print("[app] Loading models (this happens once)...")
        self.classifier = CropDiseaseClassifier()
        self.advisor = CropAdvisor()
        self.enable_voice = enable_voice
        if enable_voice:
            self.asr = SpeechRecognizer()
            self.speaker = Speaker()

    def diagnose_image(self, image_path: str):
        return self.classifier.predict(image_path, top_k=1)[0]

    def answer(self, image_path: str, question_text: str = "", question_audio: str | None = None):
        diagnosis = self.diagnose_image(image_path)
        diagnosis_str = f"{diagnosis['label']} ({diagnosis['confidence']*100:.0f}% confidence)"

        if question_audio and self.enable_voice:
            question_text = self.asr.transcribe(question_audio)

        if not question_text:
            question_text = "What is wrong and what should I do?"

        advice = self.advisor.advise(diagnosis_str, question_text)
        return diagnosis_str, question_text, advice

    def answer_and_speak(self, image_path: str, question_audio: str):
        diagnosis_str, question_text, advice = self.answer(image_path, question_audio=question_audio)
        if self.enable_voice:
            self.speaker.speak(advice)
        return diagnosis_str, question_text, advice


def run_cli(image_path: str, question: str):
    app = KrishiSahayak(enable_voice=False)
    diagnosis_str, question_text, advice = app.answer(image_path, question_text=question)
    print("\n--- Diagnosis ---")
    print(diagnosis_str)
    print("\n--- Question ---")
    print(question_text)
    print("\n--- Advice ---")
    print(advice)


def run_ui():
    import gradio as gr

    app = KrishiSahayak(enable_voice=True)

    def process(image, audio, typed_question):
        if image is None:
            return "Please upload or capture a crop photo.", "", ""
        question_text = typed_question or ""
        diagnosis_str, q_text, advice = app.answer(
            image, question_text=question_text, question_audio=audio
        )
        try:
            app.speaker.speak(advice)
        except Exception as e:
            print(f"[tts] Skipped speaking (no audio device?): {e}")
        return diagnosis_str, q_text, advice

    with gr.Blocks(title="Krishi Sahayak") as demo:
        gr.Markdown("# 🌾 Krishi Sahayak — Offline Crop Health Advisor")
        gr.Markdown("Runs fully on-device. No internet required.")
        with gr.Row():
            image_in = gr.Image(type="filepath", label="Photo of leaf/crop")
            audio_in = gr.Audio(type="filepath", label="Ask by voice (optional)")
        text_in = gr.Textbox(label="Or type your question (optional)")
        submit = gr.Button("Get Advice", variant="primary")
        diagnosis_out = gr.Textbox(label="Diagnosis")
        question_out = gr.Textbox(label="Understood question")
        advice_out = gr.Textbox(label="Advice", lines=6)

        submit.click(
            process,
            inputs=[image_in, audio_in, text_in],
            outputs=[diagnosis_out, question_out, advice_out],
        )

    # server_name="0.0.0.0" if you need it reachable on the local village network;
    # keep default 127.0.0.1 for a single offline kiosk machine.
    demo.launch(inbrowser=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Krishi Sahayak")
    parser.add_argument("--ui", action="store_true", help="Launch local web UI")
    parser.add_argument("--cli", nargs="*", help="CLI mode: <image_path> [question text]")
    args = parser.parse_args()

    if args.ui:
        run_ui()
    elif args.cli:
        image_path = args.cli[0]
        question = " ".join(args.cli[1:]) if len(args.cli) > 1 else ""
        run_cli(image_path, question)
    else:
        parser.print_help()
        sys.exit(1)
