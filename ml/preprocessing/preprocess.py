"""
RetinaAI - Image Preprocessing Pipeline
-----------------------------------------
Modular preprocessing so it can be swapped out to match whatever
trained model / dataset is eventually plugged in.

Pipeline:
  1. Validate the file is a real, readable image.
  2. Convert to RGB.
  3. Resize to the model's expected input size (default 224x224).
  4. Normalize pixel values to [0, 1].
  5. Return both a display-ready preview and a model-ready array.
"""

from PIL import Image, UnidentifiedImageError
import numpy as np
import io

DEFAULT_INPUT_SIZE = (224, 224)
MAX_FILE_SIZE_MB = 15
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}


class PreprocessingError(Exception):
    """Raised when an uploaded file can't be safely preprocessed."""
    pass


def validate_and_load(file_bytes: bytes) -> Image.Image:
    """Validate size/format and load the image, raising a clear error otherwise."""
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise PreprocessingError(
            f"Image is too large ({size_mb:.1f} MB). Max allowed is {MAX_FILE_SIZE_MB} MB."
        )

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.load()
    except UnidentifiedImageError:
        raise PreprocessingError("File is not a valid or supported image format.")
    except Exception as exc:
        raise PreprocessingError(f"Could not read image: {exc}")

    fmt = (img.format or "").upper()
    if fmt not in ALLOWED_FORMATS:
        raise PreprocessingError(
            f"Unsupported image format '{fmt}'. Supported: {', '.join(sorted(ALLOWED_FORMATS))}."
        )

    return img


def preprocess_image(file_bytes: bytes, target_size=DEFAULT_INPUT_SIZE):
    """
    Full preprocessing pipeline.

    Returns:
        dict with:
          - "original_preview": PIL.Image (RGB, unresized, for display)
          - "model_input": np.ndarray, shape (1, H, W, 3), normalized to [0,1]
          - "resized_preview": PIL.Image (RGB, resized, for display)
    """
    img = validate_and_load(file_bytes)

    # Convert to RGB (handles grayscale, RGBA, palette images, etc.)
    rgb_img = img.convert("RGB")

    # Keep an unresized copy for display purposes
    original_preview = rgb_img.copy()

    # Resize to model input size
    resized = rgb_img.resize(target_size, Image.BILINEAR)

    # Normalize to [0, 1] float32 array with batch dimension
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    model_input = np.expand_dims(arr, axis=0)

    return {
        "original_preview": original_preview,
        "resized_preview": resized,
        "model_input": model_input,
    }
