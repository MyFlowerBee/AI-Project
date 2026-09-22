"""
RetinaAI - Prediction Service
--------------------------------
This module defines a common interface (`Predictor`) with two
implementations:

  1. DemoPredictor  - a clearly-labeled MOCK predictor used when no
     trained model is available. It returns plausible-looking but
     RANDOMLY GENERATED results so the rest of the application (UI,
     API, history, Grad-CAM display, etc.) can be fully exercised.
     Results from this predictor are NEVER real medical predictions.

  2. RealModelPredictor - a stub showing exactly where a real trained
     TensorFlow/Keras model (e.g. EfficientNet/ResNet transfer
     learning) would be loaded and called. It raises a clear error
     until a real model file is actually supplied.

Swap which one is active via the `mode` field in requests, or the
APP_MODE environment variable. AI Model Mode should only be enabled
once a real, trained model is present.
"""

from abc import ABC, abstractmethod
import numpy as np
import hashlib

DR_CLASSES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
]


class Predictor(ABC):
    @abstractmethod
    def predict(self, model_input: np.ndarray, seed_bytes: bytes = b"") -> dict:
        """Return {"prediction": str, "confidence": float, "class_probabilities": dict}"""
        raise NotImplementedError


class DemoPredictor(Predictor):
    """
    MOCK / DEMO predictor.

    Produces a deterministic-but-varied fake probability distribution
    derived from a hash of the image bytes (so the same image gives
    the same demo result, which is less confusing during a viva/demo,
    without needing any real model weights).

    THESE RESULTS ARE SIMULATED AND MUST NEVER BE PRESENTED AS REAL
    CLINICAL PREDICTIONS.
    """

    def predict(self, model_input: np.ndarray, seed_bytes: bytes = b"") -> dict:
        # Derive a stable seed from the image bytes so demo results
        # are reproducible per-image, not just pure random noise.
        digest = hashlib.sha256(seed_bytes or model_input.tobytes()).digest()
        seed = int.from_bytes(digest[:4], "big")
        rng = np.random.default_rng(seed)

        # Generate a plausible-looking probability distribution
        raw = rng.dirichlet(alpha=[1.5, 1.0, 1.0, 0.8, 0.6])
        # Slightly sharpen it so one class tends to dominate, like a
        # real classifier's softmax output would.
        raw = raw ** 1.8
        raw = raw / raw.sum()

        class_probabilities = {cls: float(round(p, 4)) for cls, p in zip(DR_CLASSES, raw)}
        top_class = max(class_probabilities, key=class_probabilities.get)
        confidence = class_probabilities[top_class]

        return {
            "prediction": top_class,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "mode": "demo",
        }


class RealModelPredictor(Predictor):
    """
    Placeholder for a real trained CNN / transfer-learning model
    (e.g. TensorFlow/Keras EfficientNetB0 fine-tuned on a labeled
    fundus-image dataset such as APTOS 2019 or EyePACS).

    To activate:
      1. Train or obtain a model and export it (e.g. `model.h5` or a
         SavedModel directory).
      2. Place it under ml/model/weights/.
      3. Implement `load_model()` and `predict()` below using
         tf.keras.models.load_model(...) and model.predict(...).
      4. Switch APP_MODE to "model" (see backend/main.py).
    """

    def __init__(self, weights_path: str = "ml/model/weights/dr_model.h5"):
        self.weights_path = weights_path
        self.model = None

    def load_model(self):
        raise NotImplementedError(
            "No trained model is loaded. Add a trained .h5/SavedModel file at "
            f"'{self.weights_path}' and implement load_model()/predict() before "
            "enabling AI Model Mode."
        )

    def predict(self, model_input: np.ndarray, seed_bytes: bytes = b"") -> dict:
        raise NotImplementedError(
            "AI Model Mode is not available: no trained Diabetic Retinopathy "
            "model has been supplied for this project. Use Demo Mode instead, "
            "or plug in a trained model here."
        )


def get_predictor(mode: str = "demo") -> Predictor:
    if mode == "model":
        return RealModelPredictor()
    return DemoPredictor()
