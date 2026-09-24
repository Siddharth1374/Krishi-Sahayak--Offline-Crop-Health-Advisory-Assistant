"""
vision/classify.py
-------------------
Crop disease / pest classification using an ONNX model accelerated on the
Snapdragon NPU via the QNN Execution Provider.

Model: MobileNetV3 (or EfficientNet-Lite) fine-tuned on PlantVillage,
exported and optimized through Qualcomm AI Hub for Hexagon NPU.

Get the model (example, run once on your machine with AI Hub CLI configured):

    qai-hub submit-compile-job \
        --model mobilenet_v3_small \
        --device "Snapdragon X Elite CRD" \
        --target-runtime onnx

Or use a model already published in qai_hub_models, e.g.:

    from qai_hub_models.models.mobilenet_v3_small import Model
    model = Model.from_pretrained()
    model.export(...)   # produces an .onnx you point ONNX_MODEL_PATH to

This script does NOT ship model weights. Point ONNX_MODEL_PATH at your
exported/fine-tuned .onnx file before running.
"""

import os
import json
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError as e:
    raise SystemExit("Install onnxruntime-qnn: pip install onnxruntime-qnn") from e

ONNX_MODEL_PATH = os.environ.get(
    "KS_VISION_MODEL", "models/crop_disease_mobilenetv3.onnx"
)
LABELS_PATH = os.environ.get("KS_VISION_LABELS", "models/labels.json")
INPUT_SIZE = (224, 224)  # match your exported model's expected input


def _load_labels(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback demo labels (PlantVillage-style) — replace with your real class list
    return [
        "Healthy",
        "Tomato_Early_Blight",
        "Tomato_Late_Blight",
        "Tomato_Leaf_Mold",
        "Potato_Early_Blight",
        "Potato_Late_Blight",
        "Corn_Common_Rust",
        "Corn_Gray_Leaf_Spot",
        "Wheat_Yellow_Rust",
        "Pepper_Bacterial_Spot",
    ]


class CropDiseaseClassifier:
    """Loads the ONNX model on the NPU (QNN EP) with graceful CPU fallback."""

    def __init__(self, model_path: str = ONNX_MODEL_PATH, labels_path: str = LABELS_PATH):
        self.labels = _load_labels(labels_path)
        self.session = self._load_session(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def _load_session(self, model_path: str) -> "ort.InferenceSession":
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Export/fine-tune a classifier "
                f"via Qualcomm AI Hub and set KS_VISION_MODEL to its path."
            )

        # Prefer the Hexagon NPU (QNN); fall back to CPU if unavailable
        # (e.g. developing on a non-Snapdragon machine).
        providers = []
        available = ort.get_available_providers()
        if "QNNExecutionProvider" in available:
            providers.append((
                "QNNExecutionProvider",
                {
                    "backend_path": "QnnHtp.dll" if os.name == "nt" else "libQnnHtp.so",
                    "htp_performance_mode": "high_performance",
                },
            ))
        providers.append("CPUExecutionProvider")

        print(f"[vision] Available EPs: {available}")
        return ort.InferenceSession(model_path, providers=providers)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB").resize(INPUT_SIZE)
        arr = np.asarray(img).astype(np.float32) / 255.0
        # ImageNet-style normalization — adjust to match your training pipeline
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        return np.expand_dims(arr, 0)

    def predict(self, image_path: str, top_k: int = 3):
        image = Image.open(image_path)
        input_tensor = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        logits = outputs[0][0]
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()

        top_idx = np.argsort(probs)[::-1][:top_k]
        return [
            {"label": self.labels[i], "confidence": float(probs[i])}
            for i in top_idx
            if i < len(self.labels)
        ]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python classify.py <image_path>")
        sys.exit(1)

    clf = CropDiseaseClassifier()
    results = clf.predict(sys.argv[1])
    print(json.dumps(results, indent=2))
