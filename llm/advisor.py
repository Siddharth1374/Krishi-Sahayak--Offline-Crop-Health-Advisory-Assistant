"""
llm/advisor.py
---------------
Generates farmer-facing treatment advice using a local, quantized LLM
(Llama-3.2-3B-Instruct or Phi-3.5-mini, GGUF quantized), grounded with
retrieved context from the local knowledge base to reduce hallucination
on dosages/timings.

Runs fully offline via llama-cpp-python. For maximum NPU acceleration on
Snapdragon, swap this backend for onnxruntime-genai with the QNN EP once
your chosen model has an AI Hub / ONNX GenAI build available.

Download a quantized model once, e.g.:
    huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
        Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models/

Then set KS_LLM_MODEL to that path.
"""

import os
from rag.knowledge_base import KnowledgeBase

try:
    from llama_cpp import Llama
except ImportError as e:
    raise SystemExit("Install llama-cpp-python: pip install llama-cpp-python") from e

LLM_MODEL_PATH = os.environ.get(
    "KS_LLM_MODEL", "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
)
N_CTX = int(os.environ.get("KS_LLM_CTX", "4096"))
N_THREADS = int(os.environ.get("KS_LLM_THREADS", str(os.cpu_count() or 4)))

SYSTEM_PROMPT = """You are Krishi Sahayak, an offline farming advisor for smallholder \
farmers. You explain plant disease/pest problems simply, in plain language, with \
concrete, safe treatment steps (organic options first when reasonable, then chemical \
options with dosage and safety precautions). Keep answers short (under 150 words), \
practical, and easy to read aloud. If retrieved reference context is provided, base \
your answer on it and do not invent dosages that are not supported by it. If you are \
unsure or the situation sounds severe, tell the farmer to consult a local agriculture \
extension officer."""


class CropAdvisor:
    def __init__(self, model_path: str = LLM_MODEL_PATH, use_rag: bool = True):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"LLM model not found at {model_path}. Download a quantized GGUF "
                f"model and set KS_LLM_MODEL to its path."
            )
        print(f"[llm] Loading {model_path} ...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            verbose=False,
        )
        self.kb = KnowledgeBase() if use_rag else None

    def _build_prompt(self, diagnosis: str, farmer_question: str, context_chunks) -> list[dict]:
        context_text = ""
        if context_chunks:
            joined = "\n---\n".join(c.text for c in context_chunks)
            context_text = f"\n\nReference material (use this to ground your answer):\n{joined}"

        user_content = (
            f"Detected condition (from image analysis): {diagnosis}\n"
            f"Farmer's question: {farmer_question}"
            f"{context_text}"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def advise(self, diagnosis: str, farmer_question: str = "What should I do?") -> str:
        context_chunks = []
        if self.kb is not None:
            context_chunks = self.kb.retrieve(f"{diagnosis} {farmer_question}", top_k=3)

        messages = self._build_prompt(diagnosis, farmer_question, context_chunks)
        response = self.llm.create_chat_completion(
            messages=messages, max_tokens=300, temperature=0.3
        )
        return response["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    advisor = CropAdvisor()
    answer = advisor.advise(
        diagnosis="Tomato_Early_Blight (92% confidence)",
        farmer_question="How much fungicide should I use and how often?",
    )
    print("\nAdvice:\n", answer)
