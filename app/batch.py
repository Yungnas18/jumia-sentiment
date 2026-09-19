"""
Batch upload support for Jumia Sentiment Analysis
====================================================
Handles:
  - Validating uploaded CSV files (size, row count, required column)
  - Running batch predictions through a trained model
  - Persisting a lightweight history of past uploads to SQLite
  - Generating a per-batch summary pie chart (same visual style as
    the main dashboard) and a downloadable results CSV

No user accounts/authentication — history is just a running log of
past uploads, visible to anyone using the app. This is intentional:
see the project README for the reasoning.
"""

import csv
import io
import os
import sqlite3
import uuid
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_ROWS = 5000
REQUIRED_COLUMN = "review_text"


class BatchValidationError(Exception):
    """Raised when an uploaded file fails validation. Message is user-facing."""
    pass


def init_db(db_path: str):
    """Create the batch_history table if it doesn't already exist."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS batch_history (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            model_used TEXT NOT NULL,
            total_reviews INTEGER NOT NULL,
            positive_count INTEGER NOT NULL,
            negative_count INTEGER NOT NULL,
            avg_confidence REAL NOT NULL,
            chart_path TEXT NOT NULL,
            results_path TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def validate_upload(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Validates an uploaded CSV's size, structure, and row count.
    Returns a cleaned DataFrame if valid, otherwise raises
    BatchValidationError with a clear, user-facing message.
    """
    if not filename.lower().endswith(".csv"):
        raise BatchValidationError("Only .csv files are supported.")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise BatchValidationError(
            f"File is {size_mb:.1f}MB, which exceeds the 5MB limit. "
            f"Please upload a smaller batch."
        )

    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        raise BatchValidationError(
            "Couldn't read this file as a CSV. Please check the file is a "
            "valid, comma-separated CSV file."
        )

    # Normalize column names for a case/whitespace-tolerant match
    normalized = {c.strip().lower(): c for c in df.columns}
    if REQUIRED_COLUMN not in normalized:
        raise BatchValidationError(
            f"Your CSV must include a column named '{REQUIRED_COLUMN}'. "
            f"Found columns: {', '.join(df.columns)}"
        )

    review_col = normalized[REQUIRED_COLUMN]
    df = df.rename(columns={review_col: REQUIRED_COLUMN})
    df = df.dropna(subset=[REQUIRED_COLUMN]).reset_index(drop=True)
    df[REQUIRED_COLUMN] = df[REQUIRED_COLUMN].astype(str).str.strip()
    df = df[df[REQUIRED_COLUMN] != ""].reset_index(drop=True)

    if len(df) == 0:
        raise BatchValidationError(
            f"No usable rows found — the '{REQUIRED_COLUMN}' column is empty."
        )

    if len(df) > MAX_ROWS:
        raise BatchValidationError(
            f"This file has {len(df)} rows, which exceeds the {MAX_ROWS}-row "
            f"limit for a single batch. Please split it into smaller files."
        )

    return df


def run_batch_prediction(
    df: pd.DataFrame,
    model_name: str,
    models: dict,
    vectorizer,
    scaler,
    label_encoder,
) -> pd.DataFrame:
    """
    Runs the chosen model across every review in the batch at once
    (vectorized, not row-by-row) and returns the DataFrame with
    predicted_sentiment and confidence_pct columns added.
    """
    if model_name not in models:
        model_name = "logistic_regression"

    model = models[model_name]
    vecs = vectorizer.transform(df[REQUIRED_COLUMN].tolist())

    if model_name == "naive_bayes":
        model_input = vecs.toarray()
    else:
        model_input = scaler.transform(vecs)

    preds = model.predict(model_input)
    probs = model.predict_proba(model_input)
    confidences = probs.max(axis=1)

    df = df.copy()
    df["predicted_sentiment"] = label_encoder.inverse_transform(preds)
    df["confidence_pct"] = (confidences * 100).round(1)

    return df, model_name


def save_batch_results(
    df: pd.DataFrame,
    filename: str,
    model_name: str,
    model_label: str,
    db_path: str,
    charts_dir: str,
    results_dir: str,
) -> dict:
    """
    Saves the batch's pie chart, downloadable results CSV, and a
    summary row in SQLite. Returns the summary dict used by the API
    response and the dashboard.
    """
    batch_id = uuid.uuid4().hex[:12]
    uploaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    total = len(df)
    positive_count = int((df["predicted_sentiment"] == "Positive").sum())
    negative_count = total - positive_count
    avg_confidence = round(float(df["confidence_pct"].mean()), 1)

    # Pie chart — same visual style as the main dashboard's sentiment_pie.png
    chart_filename = f"{batch_id}.png"
    chart_path = os.path.join(charts_dir, chart_filename)
    plt.figure(figsize=(5, 5))
    counts = [positive_count, negative_count]
    labels = ["Positive", "Negative"]
    colors = ["#2ecc71", "#e74c3c"]
    # Guard against an all-one-class batch (pie chart can't plot a 0 slice cleanly)
    nonzero_counts = [c for c in counts if c > 0]
    nonzero_labels = [l for l, c in zip(labels, counts) if c > 0]
    nonzero_colors = [col for col, c in zip(colors, counts) if c > 0]
    plt.pie(nonzero_counts, labels=nonzero_labels, colors=nonzero_colors,
            autopct="%1.1f%%", startangle=140)
    plt.gca().add_artist(plt.Circle((0, 0), 0.5, color="white"))
    plt.title(f"Batch: {filename}")
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()

    # Downloadable results CSV
    results_filename = f"{batch_id}_results.csv"
    results_path = os.path.join(results_dir, results_filename)
    df.to_csv(results_path, index=False, quoting=csv.QUOTE_MINIMAL)

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO batch_history
            (id, filename, uploaded_at, model_used, total_reviews,
             positive_count, negative_count, avg_confidence,
             chart_path, results_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            batch_id, filename, uploaded_at, model_label, total,
            positive_count, negative_count, avg_confidence,
            chart_filename, results_filename,
        ),
    )
    conn.commit()
    conn.close()

    return {
        "batch_id": batch_id,
        "filename": filename,
        "uploaded_at": uploaded_at,
        "model_used": model_label,
        "total_reviews": total,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "avg_confidence": avg_confidence,
        "chart_url": f"/static/history_charts/{chart_filename}",
        "download_url": f"/api/batch-download/{batch_id}",
    }


def get_batch_history(db_path: str, limit: int = 10) -> list:
    """Returns the most recent batches, newest first, for the dashboard."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM batch_history ORDER BY uploaded_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            "id": row["id"],
            "filename": row["filename"],
            "uploaded_at": row["uploaded_at"],
            "model_used": row["model_used"],
            "total_reviews": row["total_reviews"],
            "positive_count": row["positive_count"],
            "negative_count": row["negative_count"],
            "avg_confidence": row["avg_confidence"],
            "chart_url": f"/static/history_charts/{row['chart_path']}",
            "download_url": f"/api/batch-download/{row['id']}",
        })
    return history


def get_batch_results_path(db_path: str, batch_id: str, results_dir: str):
    """Looks up the results CSV path for a given batch_id, for download."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT filename, results_path FROM batch_history WHERE id = ?",
        (batch_id,),
    ).fetchone()
    conn.close()

    if row is None:
        return None, None

    full_path = os.path.join(results_dir, row["results_path"])
    download_name = f"sentiment_results_{row['filename']}"
    return full_path, download_name
