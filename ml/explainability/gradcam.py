"""
RetinaAI - Explainable AI (Grad-CAM)
---------------------------------------
Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the
regions of an input image that most influenced a CNN's prediction.

Because AI Model Mode has no real trained model in this project yet,
`demo_gradcam_overlay()` generates a clearly-labeled SIMULATED heatmap
(a smooth synthetic blob) so the full UI/UX and API contract can be
demonstrated end-to-end. `real_gradcam()` shows exactly where the real
Grad-CAM implementation (using TensorFlow's GradientTape against the
model's last conv layer) would go once a trained model exists.
"""

from PIL import Image
import numpy as np


def _synthetic_heatmap(size, seed: int) -> np.ndarray:
    """Generate a smooth, plausible-looking fake activation map."""
    h, w = size[1], size[0]
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]

    heat = np.zeros((h, w), dtype=np.float32)
    n_blobs = rng.integers(1, 3)
    for _ in range(n_blobs):
        cx, cy = rng.uniform(0.25, 0.75) * w, rng.uniform(0.25, 0.75) * h
        sigma = rng.uniform(0.12, 0.22) * min(h, w)
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
        heat += blob

    heat = heat / heat.max()
    return heat


def _apply_colormap(heat: np.ndarray) -> np.ndarray:
    """Simple red-yellow 'jet-like' colormap without extra dependencies."""
    r = np.clip(1.5 - np.abs(4 * heat - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * heat - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * heat - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def demo_gradcam_overlay(base_image: Image.Image, seed_bytes: bytes, alpha: float = 0.45) -> Image.Image:
    """
    Return a NEW image: the base fundus image with a simulated
    heatmap overlay. Labeled everywhere in the UI as demo/simulated.
    """
    import hashlib
    seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:4], "big")

    base = base_image.convert("RGB")
    heat = _synthetic_heatmap(base.size, seed)
    heat_rgb = _apply_colormap(heat)
    heat_img = Image.fromarray(heat_rgb).resize(base.size)

    blended = Image.blend(base, heat_img, alpha=alpha)
    return blended


def real_gradcam(model, image_array: np.ndarray, last_conv_layer_name: str):
    """
    Placeholder for a real Grad-CAM implementation.

    Real implementation sketch (once a trained tf.keras model exists):

        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_array)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    Not implemented here because no trained model is available for
    this project yet.
    """
    raise NotImplementedError(
        "Real Grad-CAM requires a trained model with a known final "
        "convolutional layer. Supply a trained model to enable this."
    )
