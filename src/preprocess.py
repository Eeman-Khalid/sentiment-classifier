"""
preprocess.py
-------------
Shared preprocessing pipeline for:
  - Amazon Product Reviews (primary dataset)
  - Women's E-Commerce Clothing Reviews (secondary / cross-validation)

Usage:
    from src.preprocess import load_amazon, load_clothing, preprocess_df
"""

import re
import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Download required NLTK data 
for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
# Keep negations they matter for sentiment
NEGATIONS = {"no", "not", "nor", "never", "neither", "barely", "hardly", "scarcely"}
STOP_WORDS -= NEGATIONS

lemmatizer = WordNetLemmatizer()


# Label mapping helpers

def rating_to_sentiment_amazon(rating: float) -> str:
    """
    Amazon ratings are 1–5 stars.
      5        → positive
      3, 4     → neutral       (4 is included here because reviews often
                                say 'good but not great' — keeps class balance)
      1, 2     → negative
    """
    if rating == 5:
        return "positive"
    elif rating in (3, 4):
        return "neutral"
    else:
        return "negative"


def rating_to_sentiment_clothing(rating: float) -> str:
    """
    Clothing dataset ratings are also 1–5.
    Same mapping as Amazon for consistency.
    """
    if rating == 5:
        return "positive"
    elif rating in (3, 4):
        return "neutral"
    else:
        return "negative"


LABEL_MAP = {"positive": 2, "neutral": 1, "negative": 0}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

# Text cleaning

def clean_text(text: str) -> str:
    """
    Full text cleaning pipeline:
      1. Lowercase
      2. Remove HTML tags
      3. Remove URLs
      4. Remove non-alphabetic characters (keep spaces)
      5. Tokenize
      6. Remove stopwords (keeping negations)
      7. Lemmatize
      8. Rejoin
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # strip URLs
    text = re.sub(r"[^a-z\s]", " ", text)          # keep only letters
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS]

    return " ".join(tokens)

# Dataset loaders

def load_amazon(filepath: str) -> pd.DataFrame:
    """
    Load and standardise the Amazon Product Reviews dataset.

    Expected columns used:
        - 'reviews.text'   : raw review text
        - 'reviews.rating' : numeric rating 1–5

    Returns a DataFrame with columns: [text, sentiment, label]
    """
    df = pd.read_csv(filepath)

    # Flexible column detection — Kaggle CSV sometimes has slight name variations
    text_col = next((c for c in df.columns if "text" in c.lower()), None)
    rating_col = next((c for c in df.columns if "rating" in c.lower()), None)

    if text_col is None or rating_col is None:
        raise ValueError(
            f"Could not find text/rating columns in Amazon CSV. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[[text_col, rating_col]].copy()
    df.columns = ["text", "rating"]

    # Drop rows where text or rating is missing
    df.dropna(subset=["text", "rating"], inplace=True)

    # Ensure rating is numeric
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df.dropna(subset=["rating"], inplace=True)
    df["rating"] = df["rating"].astype(int)

    # Keep only valid ratings
    df = df[df["rating"].between(1, 5)]

    # Map to sentiment
    df["sentiment"] = df["rating"].apply(rating_to_sentiment_amazon)
    df["label"] = df["sentiment"].map(LABEL_MAP)

    df.reset_index(drop=True, inplace=True)
    print(f"[Amazon] Loaded {len(df):,} reviews")
    print(df["sentiment"].value_counts().to_string())
    return df


def load_clothing(filepath: str) -> pd.DataFrame:
    """
    Load and standardise the Women's Clothing Reviews dataset.

    Expected columns used:
        - 'Review Text' : raw review text
        - 'Rating'      : numeric rating 1–5

    Returns a DataFrame with columns: [text, sentiment, label]
    """
    df = pd.read_csv(filepath)

    text_col = next(
        (c for c in df.columns if "review" in c.lower() and "text" in c.lower()), None
    )
    rating_col = next((c for c in df.columns if "rating" in c.lower()), None)

    if text_col is None or rating_col is None:
        raise ValueError(
            f"Could not find text/rating columns in Clothing CSV. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[[text_col, rating_col]].copy()
    df.columns = ["text", "rating"]

    df.dropna(subset=["text", "rating"], inplace=True)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df.dropna(subset=["rating"], inplace=True)
    df["rating"] = df["rating"].astype(int)
    df = df[df["rating"].between(1, 5)]

    df["sentiment"] = df["rating"].apply(rating_to_sentiment_clothing)
    df["label"] = df["sentiment"].map(LABEL_MAP)

    df.reset_index(drop=True, inplace=True)
    print(f"[Clothing] Loaded {len(df):,} reviews")
    print(df["sentiment"].value_counts().to_string())
    return df


# Preprocessing orchestrator

def preprocess_df(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Apply clean_text() to the specified column.
    Drops rows that become empty after cleaning.
    """
    print("Cleaning text... (this may take a minute)")
    df = df.copy()
    df["clean_text"] = df[text_col].apply(clean_text)

    before = len(df)
    df = df[df["clean_text"].str.strip() != ""]
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} empty rows after cleaning.")

    df.reset_index(drop=True, inplace=True)
    return df

# Train / val / test split

def split_data(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    label_col: str = "label",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    """
    Stratified split → train / val / test.
    Default: 70% train | 15% val | 15% test

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df[text_col].values
    y = df[label_col].values

    # First split off test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Then split remaining into train + val
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=random_state
    )

    print(f"\nSplit sizes → Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# Save / load processed data


def save_processed(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved processed data → {out_path}")


def load_processed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded processed data from {path} ({len(df):,} rows)")
    return df