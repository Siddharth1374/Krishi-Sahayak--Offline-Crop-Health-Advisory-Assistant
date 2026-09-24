# Krishi Sahayak

Offline crop disease advisor built for the Snapdragon HP AI PC challenge. Basically the idea is: a farmer takes a photo of a sick-looking leaf, points a mic at it and asks a question, and gets back spoken advice — without needing internet at all. Everything runs on-device using the NPU on Snapdragon-powered HP laptops.

I built this because a lot of the existing crop-advisory apps (Plantix etc.) need a live internet connection, and a huge chunk of farmland just doesn't have reliable connectivity. So the whole point here is offline-first.

## What it actually does

1. You point a camera/webcam at a leaf
2. A vision model (MobileNetV3, exported/optimized through Qualcomm AI Hub) classifies what disease or pest it probably is
3. You can ask a follow-up question by voice — "how much should I spray?" type stuff
4. A small local LLM answers, using some reference docs (agri guides) I stuffed into a local knowledge base so it doesn't just make up dosages
5. It reads the answer back out loud

## Status / what's real vs what's a stub right now

Being honest here since this is still WIP:

- Vision pipeline (`vision/classify.py`) — code is done and uses the QNN execution provider when available, falls back to CPU. **You need to plug in your own trained `.onnx` model** — I did not ship model weights (they're huge + I haven't finished fine-tuning on a full disease dataset yet, only tested on a handful of tomato/potato classes).
- LLM advisor (`llm/advisor.py`) — works with any GGUF quantized model, tested with Llama-3.2-3B-Instruct. Currently runs on CPU via llama-cpp-python. NPU acceleration for this part is still TODO (would need to move to onnxruntime-genai + QNN, haven't gotten to it).
- RAG / knowledge base — just TF-IDF right now, not a real embedding-based search. It's dumb but it works and it's fully offline with zero extra downloads. Only one sample doc (`data/knowledge/sample_tomato_guide.txt`) is in there right now, need to add more crops.
- ASR/TTS — Whisper + pyttsx3, both CPU. Voice quality on TTS is pretty robotic honestly, would like to swap for MMS-TTS at some point for actual local language support (Hindi etc.)
- UI — there's a basic Gradio interface (`main.py --ui`), nothing fancy, just enough to demo the flow.

So tl;dr: the pipeline works end to end once you supply the two model files, but there's real work left before this is production-ready for actual farmers (proper dataset, proper language support, NPU-accelerated LLM, testing in the field basically).

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # (source venv/bin/activate on mac/linux)
pip install -r requirements.txt
```

You'll also need:

**A vision model.** Either fine-tune one yourself starting from `qai_hub_models.models.mobilenet_v3_small` on a labeled crop-disease dataset (I used a subset of PlantVillage), or grab something pre-trained and export/optimize it for NPU via the AI Hub CLI:

```bash
pip install qai-hub qai-hub-models
qai-hub configure --api_token YOUR_TOKEN
```

Drop the resulting `.onnx` at `models/crop_disease_mobilenetv3.onnx` and update `models/labels.json` to match your actual classes (mine are just placeholder tomato/potato/corn/wheat/rice labels right now).

**An LLM.** I used:
```bash
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models/
```
Any similar small quantized instruct model should work fine, just point `KS_LLM_MODEL` at it if the filename's different.

## Running it

```bash
python main.py --cli path/to/leaf.jpg "how much fungicide do I need?"
```

or the web UI version:

```bash
python main.py --ui
```

## Folder layout

```
krishi_sahayak/
├── main.py                 -> entry point, CLI + gradio UI
├── vision/classify.py       -> disease/pest classifier, QNN/NPU
├── llm/advisor.py           -> local LLM + RAG grounding
├── rag/knowledge_base.py    -> offline TF-IDF retrieval
├── speech/asr.py            -> whisper speech-to-text
├── speech/tts.py            -> offline TTS
├── data/knowledge/          -> reference docs for RAG (add more here)
├── models/                  -> put your .onnx / .gguf files here (not in repo)
└── requirements.txt
```

## Why Snapdragon / NPU

Two reasons this matters for the use case, not just for the sake of the challenge brief:

- Power draw — NPU inference is way cheaper on battery than running the same models on CPU, which matters if this ends up on a solar-charged kiosk in a village with unreliable power.
- Latency — a farmer standing in a field isn't going to wait 30 seconds for a cloud round trip that doesn't even exist there in the first place.

Haven't finished formal benchmarks (CPU vs QNN inference time/power) yet but that's next on the list — planning to add numbers here once I've run them on the actual dev kit.

## Known issues / things to fix next

- Labels file is a placeholder, needs real class list once the model's actually fine-tuned
- No error handling yet if webcam/mic isn't available
- pyttsx3 voice is pretty rough for anything other than English
- Knowledge base only has one sample file — needs way more crop-specific docs to be actually useful
- Haven't tested on a real Snapdragon dev kit yet, only checked that QNN EP is detected correctly in code — need to validate actual NPU offload happens
