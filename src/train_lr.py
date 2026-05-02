"""
train_lr.py
-----------
Logistic Regression baseline model using TF-IDF features.
Trains, evaluates, and saves the model + vectorizer.

Run from project root:
    python src/train_lr.py
"""

import os
import joblib
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import load_amazon, preprocess_df, split_data, load_processed
from src.evaluate import compute_metrics, plot_confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = "data/processed/amazon_clean.csv"
MODEL_SAVE_DIR = "models/saved"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def train():
    # ---- Load data ---------------------------------------------------------
    if os.path.exists(DATA_PATH):
        from src.preprocess import load_processed
        df = load_processed(DATA_PATH)
    else:
        print("Processed data not found. Run notebook 01 first, or set DATA_PATH.")
        return

    # ---- Split -------------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # ---- Build pipeline ----------------------------------------------------
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),      # unigrams + bigrams
            sublinear_tf=True,       # apply log normalization to TF
            min_df=3,                # ignore very rare terms
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            multi_class="multinomial",
            solver="lbfgs",
            n_jobs=-1,
            random_state=42,
        )),
    ])

    # ---- Train -------------------------------------------------------------
    print("\nTraining Logistic Regression...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # ---- Evaluate on validation set ----------------------------------------
    y_val_pred = pipeline.predict(X_val)
    val_results = compute_metrics(y_val, y_val_pred, model_name="LR (Validation)")
    plot_confusion_matrix(
        y_val, y_val_pred,
        model_name="Logistic Regression — Validation",
        save_path=os.path.join(MODEL_SAVE_DIR, "lr_confusion_val.png")
    )

    # ---- Evaluate on test set ----------------------------------------------
    y_test_pred = pipeline.predict(X_test)
    test_results = compute_metrics(y_test, y_test_pred, model_name="LR (Test)")
    plot_confusion_matrix(
        y_test, y_test_pred,
        model_name="Logistic Regression — Test",
        save_path=os.path.join(MODEL_SAVE_DIR, "lr_confusion_test.png")
    )

    # ---- Save model --------------------------------------------------------
    model_path = os.path.join(MODEL_SAVE_DIR, "lr_pipeline.pkl")
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved → {model_path}")

    return pipeline, test_results


def predict(text: str, pipeline=None) -> dict:
    """
    Predict sentiment for a single review string.
    Loads model from disk if pipeline is not passed in.
    """
    if pipeline is None:
        model_path = os.path.join(MODEL_SAVE_DIR, "lr_pipeline.pkl")
        pipeline = joblib.load(model_path)

    label_map_inv = {0: "negative", 1: "neutral", 2: "positive"}

    # Get probabilities
    proba = pipeline.predict_proba([text])[0]
    pred_label = int(pipeline.predict([text])[0])

    return {
        "sentiment": label_map_inv[pred_label],
        "confidence": float(np.max(proba)),
        "probabilities": {
            "negative": float(proba[0]),
            "neutral": float(proba[1]),
            "positive": float(proba[2]),
        }
    }


if __name__ == "__main__":
    train()

