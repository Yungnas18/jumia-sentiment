"""
Jumia Sentiment Analysis — Offline Training Script
====================================================
Run this once (or whenever the dataset changes) to:
  1. Clean and label the review data
  2. Train Naive Bayes and Logistic Regression models
  3. Evaluate both with cross-validation + a held-out test set
  4. Save trained models/vectorizer to models/ (as .pkl files)
  5. Save all evaluation metrics to models/metrics.json
  6. Save all charts (pie chart, word clouds, ROC curves, confusion
     matrices) to app/static/charts/ so the FastAPI app can serve them

Usage:
    python train_model.py
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,
)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "jumia_reviews_df_multi_page.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHARTS_DIR = os.path.join(BASE_DIR, "app", "static", "charts")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# STEP 1 — Load and label data
# ---------------------------------------------------------------------
print("Loading dataset...")
jumia = pd.read_csv(DATA_PATH)
jumia = jumia.rename(columns={"Rating(of 5)": "rating"})
jumia = jumia.dropna(subset=["Review Body", "Review Topic"]).reset_index(drop=True)

# rating >= 3 -> Positive, else Negative
jumia["sentiment"] = np.where(jumia["rating"] >= 3, "Positive", "Negative")

print(f"Loaded {len(jumia)} reviews")
print(jumia["sentiment"].value_counts())

# ---------------------------------------------------------------------
# STEP 2 — Charts: rating distribution + word clouds
# ---------------------------------------------------------------------
nltk.download("stopwords", quiet=True)
base_stopwords = set(stopwords.words("english"))
base_stopwords.update(["br", "href"])

sentiment_counts = jumia["sentiment"].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(
    sentiment_counts.values, labels=sentiment_counts.index,
    colors=["#2ecc71", "#e74c3c"], autopct="%1.1f%%", startangle=140,
)
plt.gca().add_artist(plt.Circle((0, 0), 0.5, color="white"))
plt.title("Distribution of Jumia Review Sentiment")
plt.savefig(os.path.join(CHARTS_DIR, "sentiment_pie.png"), bbox_inches="tight")
plt.close()

for label, subset in [("all", jumia), ("positive", jumia[jumia.sentiment == "Positive"]),
                       ("negative", jumia[jumia.sentiment == "Negative"])]:
    if len(subset) == 0:
        continue
    text = " ".join(subset["Review Body"])
    wc = WordCloud(stopwords=base_stopwords, background_color="white",
                    width=900, height=500).generate(text)
    plt.figure(figsize=(9, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(os.path.join(CHARTS_DIR, f"wordcloud_{label}.png"), bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------------
# STEP 3 — Clean text
# ---------------------------------------------------------------------
def remove_punctuation(text):
    return "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"'))

jumia["Review Body"] = jumia["Review Body"].apply(remove_punctuation)
jumia["Review Topic"] = jumia["Review Topic"].apply(remove_punctuation)
jumia["combined_text"] = jumia["Review Topic"] + " " + jumia["Review Body"]

# ---------------------------------------------------------------------
# STEP 4 — Train/test split (stratified, so the tiny negative class
# is represented proportionally in both sets)
# ---------------------------------------------------------------------
X_text = jumia["combined_text"]
y = jumia["sentiment"]

X_train_text, X_test_text, y_train_raw, y_test_raw = train_test_split(
    X_text, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

vectorizer = CountVectorizer(token_pattern=r"\b\w+\b", max_features=3000)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test = le.transform(y_test_raw)
# le.classes_ -> ['Negative', 'Positive'], so Positive = 1, Negative = 0

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ---------------------------------------------------------------------
# STEP 5 — Train models (with class_weight/priors adjusted for imbalance)
# ---------------------------------------------------------------------
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "naive_bayes": GaussianNB(),
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    print(f"\nTraining {name}...")

    if name == "naive_bayes":
        Xtr, Xte = X_train.toarray(), X_test.toarray()
    else:
        Xtr, Xte = X_train_scaled, X_test_scaled

    # 5-fold cross-validation on the training set (F1, since classes are imbalanced)
    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="f1")

    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    probs = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else preds

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    metrics = {
        "precision": round(precision_score(y_test, preds, zero_division=0), 3),
        "recall": round(recall_score(y_test, preds, zero_division=0), 3),
        "f1": round(f1_score(y_test, preds, zero_division=0), 3),
        "accuracy": round(accuracy_score(y_test, preds), 3),
        "roc_auc": round(roc_auc, 3),
        "cv_f1_mean": round(cv_scores.mean(), 3),
        "cv_f1_std": round(cv_scores.std(), 3),
    }
    results[name] = metrics
    print(metrics)

    # ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {name.replace('_', ' ').title()}")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(CHARTS_DIR, f"roc_{name}.png"), bbox_inches="tight")
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {name.replace('_', ' ').title()}")
    plt.savefig(os.path.join(CHARTS_DIR, f"confusion_{name}.png"), bbox_inches="tight")
    plt.close()

    # Save model
    joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))

# ---------------------------------------------------------------------
# STEP 6 — Save shared artifacts
# ---------------------------------------------------------------------
joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))

results["dataset_info"] = {
    "total_reviews": len(jumia),
    "positive_count": int((jumia["sentiment"] == "Positive").sum()),
    "negative_count": int((jumia["sentiment"] == "Negative").sum()),
    "train_size": int(X_train.shape[0]),
    "test_size": int(X_test.shape[0]),
}

with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nAll done.")
print(f"Models saved to:  {MODELS_DIR}")
print(f"Charts saved to:  {CHARTS_DIR}")
print(f"Metrics saved to: {os.path.join(MODELS_DIR, 'metrics.json')}")
