"""
RetinaAI Backend - FastAPI Application
==========================================
Academic / research prototype for AI-assisted Diabetic Retinopathy
screening support. NOT a certified medical device. See README.md and
the /about content in the frontend for full disclaimers.

Run with:
    uvicorn backend.main:app --reload --port 8000
(from the retina-ai/ project root)
"""

import sys
import os
import io
import base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root on path

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ml.preprocessing.preprocess import preprocess_image, PreprocessingError
from ml.model.predictor import get_predictor, DR_CLASSES
from ml.explainability.gradcam import demo_gradcam_overlay
from backend.database.db import init_db, insert_analysis, get_history, get_analysis_by_id

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# "demo"  -> uses the clearly-labeled mock predictor (default, safe)
# "model" -> would use a real trained model (raises until one is supplied)
APP_MODE = os.environ.get("APP_MODE", "demo")

# Confidence threshold below which a result is flagged for mandatory
# clinical review. Configurable, NOT a clinically validated threshold.
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.80"))

MAX_HISTORY_LIMIT = 200

app = FastAPI(
    title="RetinaAI API",
    description="AI-assisted Diabetic Retinopathy detection - academic prototype.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relax for local student-project use
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    mode: str
    confidence_threshold: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_base64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _review_status(confidence: float) -> str:
    if confidence >= CONFIDENCE_THRESHOLD:
        return "Clinical Review Recommended"
    return "Low Confidence - Further Evaluation Required"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "mode": APP_MODE, "confidence_threshold": CONFIDENCE_THRESHOLD}


@app.post("/preprocess")
async def preprocess_endpoint(file: UploadFile = File(...)):
    """Validate + preprocess an image and return preview data (no prediction)."""
    file_bytes = await file.read()
    try:
        result = preprocess_image(file_bytes)
    except PreprocessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "original_preview": _image_to_base64(result["original_preview"]),
        "resized_preview": _image_to_base64(result["resized_preview"]),
        "input_shape": list(result["model_input"].shape),
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...), mode: str = Query(default=None)):
    """
    Full pipeline: validate -> preprocess -> predict -> confidence check
    -> Grad-CAM (demo) -> persist to history -> return everything the UI needs.
    """
    active_mode = mode or APP_MODE
    file_bytes = await file.read()

    try:
        processed = preprocess_image(file_bytes)
    except PreprocessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    predictor = get_predictor(active_mode)
    try:
        result = predictor.predict(processed["model_input"], seed_bytes=file_bytes)
    except NotImplementedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    confidence = result["confidence"]
    status = _review_status(confidence)

    # Grad-CAM (demo-only until a real model exists)
    gradcam_img = None
    gradcam_note = None
    if active_mode == "demo":
        overlay = demo_gradcam_overlay(processed["original_preview"], seed_bytes=file_bytes)
        gradcam_img = _image_to_base64(overlay)
        gradcam_note = (
            "SIMULATED heatmap (Demo Mode - no trained model). Highlighted regions "
            "are illustrative only and do not reflect a real model's attention."
        )
    else:
        gradcam_note = "Grad-CAM requires a trained model; not available in this deployment."

    saved = insert_analysis(
        image_filename=file.filename or "uploaded_image",
        prediction=result["prediction"],
        confidence=confidence,
        class_probabilities=result["class_probabilities"],
        review_status=status,
        mode=active_mode,
    )

    return {
        "analysis_id": saved["id"],
        "timestamp": saved["timestamp"],
        "mode": active_mode,
        "prediction": result["prediction"],
        "confidence": confidence,
        "class_probabilities": result["class_probabilities"],
        "review_status": status,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "original_image": _image_to_base64(processed["original_preview"]),
        "gradcam_image": gradcam_img,
        "gradcam_note": gradcam_note,
        "disclaimer": (
            "This AI result is for research/educational decision support and "
            "must not be used as a standalone medical diagnosis."
        ),
    }


@app.post("/gradcam")
async def gradcam_endpoint(file: UploadFile = File(...)):
    """Standalone Grad-CAM generation for an already-uploaded image (demo mode)."""
    file_bytes = await file.read()
    try:
        processed = preprocess_image(file_bytes)
    except PreprocessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    overlay = demo_gradcam_overlay(processed["original_preview"], seed_bytes=file_bytes)
    return {
        "gradcam_image": _image_to_base64(overlay),
        "note": "SIMULATED heatmap (Demo Mode). Requires a trained model for real Grad-CAM.",
    }


@app.get("/history")
def history_endpoint(limit: int = Query(default=50, le=MAX_HISTORY_LIMIT)):
    return {"items": get_history(limit=limit)}


@app.get("/history/{analysis_id}")
def history_detail(analysis_id: str):
    item = get_analysis_by_id(analysis_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return item


@app.get("/classes")
def classes_endpoint():
    return {"classes": DR_CLASSES}
