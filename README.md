# RetinaAI — AI-Based Diabetic Retinopathy Detection System

**Academic / research prototype.** RetinaAI demonstrates an end-to-end AI-assisted
workflow for screening retinal (fundus) images for signs of Diabetic Retinopathy (DR):
upload → preprocessing → CNN classification → confidence/uncertainty check → Grad-CAM
explainability → dashboard → clinical-review recommendation.

> ⚠️ **This is not a certified medical device.** It has not been clinically validated
> and must never be used as a standalone diagnostic tool. It is designed to support,
> not replace, a qualified ophthalmologist.

---

## 1. Problem Statement

Diabetic Retinopathy is a leading cause of preventable blindness in people with diabetes.
Early detection via regular retinal screening dramatically improves outcomes, but
screening capacity is limited in many regions. AI-assisted triage can help prioritize
images for human review — this project is a student/academic demonstration of that
concept, built with responsible-AI guardrails (uncertainty flagging, explainability,
human oversight) baked in from the start.

## 2. Objectives

- Build a working, demonstrable pipeline for DR screening support.
- Show explainable AI (Grad-CAM) rather than an opaque prediction.
- Flag low-confidence predictions instead of presenting every output as certain.
- Keep the "real model" and "demo model" code paths clearly, structurally separate.
- Document everything needed to plug in a real trained model later.

## 3. System Architecture

```
USER
  ↓
Upload Fundus Image (React frontend)
  ↓
Image Preprocessing (validate → RGB → resize 224×224 → normalize)
  ↓
CNN / Transfer-Learning Model  (Demo Mode: simulated | AI Model Mode: real, if supplied)
  ↓
DR Classification (5 classes) + Confidence Score
  ↓
Confidence/Uncertainty Analysis (configurable threshold, default 0.80)
  ↓
Grad-CAM Explainability (heatmap overlay)
  ↓
Result Dashboard (prediction, confidence, probability chart, heatmap, recommendation)
  ↓
History stored in SQLite
```

### DR Classes
1. No Diabetic Retinopathy
2. Mild DR
3. Moderate DR
4. Severe DR
5. Proliferative DR

## 4. Technology Stack

| Layer          | Tech                                                        |
|----------------|---------------------------------------------------------------|
| Frontend       | React 18 (CDN, no build step), Tailwind CSS (CDN)             |
| Backend        | Python, FastAPI, Uvicorn                                      |
| AI (Demo Mode) | NumPy-based simulated predictor (clearly labeled, not a model)|
| AI (Model Mode)| Designed for TensorFlow/Keras CNN / transfer learning (EfficientNet/ResNet) — stubbed, not included |
| Explainability | Grad-CAM (simulated in Demo Mode; real implementation sketch provided for Model Mode) |
| Image Processing | Pillow, NumPy (OpenCV noted for future use)                 |
| Database       | SQLite                                                         |
| Charts         | Custom lightweight bar charts in the React frontend             |

## 5. Project Structure

```
retina-ai/
├── frontend/
│   └── index.html          # Full React SPA (Home, Upload, Results, History, About)
├── backend/
│   ├── main.py              # FastAPI app + all endpoints
│   └── database/
│       └── db.py            # SQLite persistence layer
├── ml/
│   ├── preprocessing/
│   │   └── preprocess.py    # Validation, resize, normalize
│   ├── model/
│   │   └── predictor.py     # DemoPredictor (mock) + RealModelPredictor (stub)
│   └── explainability/
│       └── gradcam.py       # Demo heatmap + real Grad-CAM sketch
├── uploads/                 # (runtime scratch space, empty by default)
├── requirements.txt
└── README.md
```

## 6. Installation & Running

### Backend
```bash
cd retina-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn backend.main:app --reload --port 8000
```
The API will be live at `http://localhost:8000` (interactive docs at `/docs`).

### Frontend
No build step required — it's a single self-contained HTML file using React via CDN.
```bash
# from the retina-ai/frontend directory
python -m http.server 5500
```
Then open `http://localhost:5500` in your browser. If your backend runs on a different
host/port, set it before the page loads:
```html
<script>window.RETINA_AI_API_BASE = "http://localhost:8000";</script>
```
(add this `<script>` tag just above the main app `<script>` in `index.html`).

## 7. API Documentation

| Method | Endpoint          | Description                                                |
|--------|-------------------|--------------------------------------------------------------|
| GET    | `/health`          | Service status, active mode, confidence threshold            |
| POST   | `/preprocess`       | Validate + preprocess an image, return preview only          |
| POST   | `/predict`          | Full pipeline: preprocess → predict → Grad-CAM → save to history |
| POST   | `/gradcam`          | Standalone Grad-CAM generation (demo)                          |
| GET    | `/history`          | List past analyses                                              |
| GET    | `/history/{id}`     | Fetch one past analysis                                          |
| GET    | `/classes`          | List the 5 DR classes                                            |

Example `/predict` response:
```json
{
  "prediction": "Moderate DR",
  "confidence": 0.87,
  "class_probabilities": {
    "No DR": 0.03, "Mild DR": 0.05, "Moderate DR": 0.87,
    "Severe DR": 0.04, "Proliferative DR": 0.01
  },
  "review_status": "Clinical Review Recommended",
  "mode": "demo"
}
```
The exact numbers above are illustrative of the response *shape* only — Demo Mode
generates a different, image-derived pseudo-random distribution per image, and none of
it is a real medical prediction.

## 8. Demo Mode vs. AI Model Mode

- **Demo Mode (default):** Uses `DemoPredictor`, a deterministic-per-image but
  otherwise randomly generated classifier output, plus a simulated Grad-CAM heatmap.
  Exercises the entire UI/API/DB pipeline without needing a trained model. Every
  Demo Mode result is labeled in the UI ("DEMO MODE (simulated results)").
- **AI Model Mode:** Routes to `RealModelPredictor`, which currently raises a clear
  `NotImplementedError` because no trained model is bundled with this project. See
  "Connecting a Real Model" below.

Switch modes from the dropdown in the frontend navbar, or via the `?mode=` query
param on `/predict`.

## 9. Connecting a Real Trained Model

1. Train a CNN (recommended: transfer learning with EfficientNetB0/ResNet50) on a
   labeled fundus-image dataset such as **APTOS 2019 Blindness Detection** or
   **EyePACS** (both are external resources — not included in or trained by this
   project's design documents).
2. Export it (`model.h5` or a Keras `SavedModel` directory) into `ml/model/weights/`.
3. In `ml/model/predictor.py`, implement `RealModelPredictor.load_model()` and
   `.predict()` using `tf.keras.models.load_model(...)` and `model.predict(...)`.
4. In `ml/explainability/gradcam.py`, implement `real_gradcam()` following the
   sketch already in the docstring (uses `tf.GradientTape` against the model's
   last convolutional layer).
5. Run the backend with `APP_MODE=model` or select "AI Model Mode" in the UI.

## 10. Dataset Requirements (for training a real model)

- Labeled fundus images across all 5 DR severity classes.
- Class balance matters — DR datasets are typically skewed toward "No DR"; consider
  class weighting or augmentation.
- Diverse patient demographics and imaging equipment/conditions to reduce bias.
- A held-out validation/test split, distinct from training data, for honest evaluation.

## 11. Explainable AI (Grad-CAM)

Grad-CAM highlights which regions of the input image most influenced a prediction —
useful for sanity-checking that the model is looking at the retina rather than
artifacts (e.g., camera vignetting). In Demo Mode, the heatmap is a labeled synthetic
overlay; the real implementation path is documented in `ml/explainability/gradcam.py`.

## 12. Confidence / Uncertainty Handling

`CONFIDENCE_THRESHOLD` (default `0.80`, configurable via environment variable) governs
whether a result is shown as "Clinical Review Recommended" or flagged as
"Low Confidence - Further Evaluation Required." This is an **application-level
heuristic**, not a clinically validated cutoff.

## 13. Limitations

- No real trained model or clinical dataset ships with this project — Demo Mode
  results are simulated and must never be treated as medical predictions.
- Not clinically validated; not a certified medical device.
- Grad-CAM in Demo Mode is illustrative only, not derived from real model gradients.
- Class imbalance, dataset bias, and generalization across populations/devices are
  known open problems for any real DR classifier and are not addressed here.

## 14. Responsible AI

- AI should support healthcare professionals, not replace them.
- Dataset bias can materially affect model performance across populations.
- Low-confidence cases are always flagged rather than silently presented as certain.
- Only minimal data is stored (no patient identifiers) — see `backend/database/db.py`.
- The system must not be used as a standalone diagnostic tool.

## 15. Sustainable AI Design (Future Scope)

The architecture keeps preprocessing lightweight and modular so a smaller model
(e.g., MobileNet-class) could be swapped in for edge/fog deployment in
resource-constrained screening settings. No environmental-impact claims are made, as
none have been measured for this prototype.

## 16. Future Scope

- Integrate a real trained CNN and real Grad-CAM.
- Add authentication and role-based access for clinicians vs. reviewers.
- Batch upload / bulk screening workflows.
- Periodic retraining pipeline with monitoring for data drift.
- Formal clinical validation study before any real-world pilot use.

---

*This project's design draws on general concepts from AI/ML for medical imaging,
explainable AI, Grad-CAM interpretability, human clinical oversight, and
privacy/bias considerations in healthcare AI — implemented here as an original
student prototype, not a reproduction of any specific external paper or dataset.*
