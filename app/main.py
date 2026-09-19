"""
Jumia Sentiment Analysis — FastAPI Application
================================================
Serves:
  - A home page with a live review-prediction form
  - An upload page for batch CSV predictions
  - A dashboard page with model comparison metrics, charts, and
    upload history
  - A JSON API for predictions, batch predictions, and metrics
    (used by the frontend JS)

Run with (from the project root):
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000 in your browser.
"""

import json
import os

import joblib
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import batch as batch_module

# ---------------------------------------------------------------------
# Paths — resolved relative to this file so it works no matter where
# uvicorn is launched from.
# ---------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMPLATES_DIR = os.path.join(APP_DIR, "templates")
STATIC_DIR = os.path.join(APP_DIR, "static")
HISTORY_CHARTS_DIR = os.path.join(STATIC_DIR, "history_charts")
HISTORY_RESULTS_DIR = os.path.join(STATIC_DIR, "history_results")
DB_PATH = os.path.join(APP_DIR, "history.db")

os.makedirs(HISTORY_CHARTS_DIR, exist_ok=True)
os.makedirs(HISTORY_RESULTS_DIR, exist_ok=True)

app = FastAPI(title="Jumia Sentiment Analysis")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

batch_module.init_db(DB_PATH)

# ---------------------------------------------------------------------
# Load trained artifacts once, at startup
# ---------------------------------------------------------------------
vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

MODELS = {
    "naive_bayes": joblib.load(os.path.join(MODELS_DIR, "naive_bayes.pkl")),
    "logistic_regression": joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl")),
}

with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
    METRICS = json.load(f)

MODEL_LABELS = {
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
}


# ---------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------
class PredictionRequest(BaseModel):
    review_text: str
    model: str = "logistic_regression"


class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    model_used: str


# ---------------------------------------------------------------------
# Core prediction logic (shared by the API and the form endpoint)
# ---------------------------------------------------------------------
def predict_sentiment(review_text: str, model_name: str) -> PredictionResponse:
    if model_name not in MODELS:
        model_name = "logistic_regression"

    model = MODELS[model_name]
    vec = vectorizer.transform([review_text])

    if model_name == "naive_bayes":
        vec_input = vec.toarray()
    else:
        vec_input = scaler.transform(vec)

    pred = model.predict(vec_input)[0]
    proba = model.predict_proba(vec_input)[0]
    confidence = float(max(proba))

    sentiment = label_encoder.inverse_transform([pred])[0]

    return PredictionResponse(
        sentiment=sentiment,
        confidence=round(confidence * 100, 1),
        model_used=MODEL_LABELS[model_name],
    )


# ---------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"model_labels": MODEL_LABELS},
    )


@app.get("/upload")
def upload_page(request: Request):
    return templates.TemplateResponse(
        request,
        "upload.html",
        {
            "model_labels": MODEL_LABELS,
            "max_rows": batch_module.MAX_ROWS,
            "max_size_mb": batch_module.MAX_FILE_SIZE_BYTES // (1024 * 1024),
        },
    )


@app.get("/dashboard")
def dashboard(request: Request):
    history = batch_module.get_batch_history(DB_PATH, limit=10)
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "metrics": METRICS,
            "model_labels": MODEL_LABELS,
            "history": history,
        },
    )


# ---------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------
@app.post("/api/predict", response_model=PredictionResponse)
def api_predict(payload: PredictionRequest):
    return predict_sentiment(payload.review_text, payload.model)


@app.get("/api/metrics")
def api_metrics():
    return METRICS


@app.post("/api/batch-predict")
async def api_batch_predict(
    file: UploadFile = File(...),
    model: str = Form("logistic_regression"),
):
    file_bytes = await file.read()

    try:
        df = batch_module.validate_upload(file_bytes, file.filename)
    except batch_module.BatchValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if model not in MODELS:
        model = "logistic_regression"

    df, model_name = batch_module.run_batch_prediction(
        df, model, MODELS, vectorizer, scaler, label_encoder
    )

    summary = batch_module.save_batch_results(
        df,
        filename=file.filename,
        model_name=model_name,
        model_label=MODEL_LABELS[model_name],
        db_path=DB_PATH,
        charts_dir=HISTORY_CHARTS_DIR,
        results_dir=HISTORY_RESULTS_DIR,
    )

    # Preview: first 20 rows, so the response stays small even for large batches
    preview_cols = ["review_text", "predicted_sentiment", "confidence_pct"]
    preview = df[preview_cols].head(20).to_dict(orient="records")
    summary["preview"] = preview

    return summary


@app.get("/api/batch-download/{batch_id}")
def api_batch_download(batch_id: str):
    full_path, download_name = batch_module.get_batch_results_path(
        DB_PATH, batch_id, HISTORY_RESULTS_DIR
    )

    if full_path is None or not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Batch not found.")

    return FileResponse(
        full_path,
        media_type="text/csv",
        filename=download_name,
    )
