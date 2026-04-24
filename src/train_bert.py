"""
train_bert.py
-------------
DistilBERT fine-tuning for 3-class sentiment classification.
Designed to run on Kaggle with GPU. Saves model to models/saved/bert/.

Run on Kaggle:
    !python src/train_bert.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import load_processed, split_data
from src.evaluate import compute_metrics, plot_confusion_matrix


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG = {
    "data_path": "data/processed/amazon_clean.csv",
    "model_save_dir": "models/saved/bert",
    "model_name": "distilbert-base-uncased",
    "num_labels": 3,
    "max_seq_len": 128,
    "batch_size": 32,
    "epochs": 4,
    "lr": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "patience": 2,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
os.makedirs(CONFIG["model_save_dir"], exist_ok=True)

LABEL_MAP_INV = {0: "negative", 1: "neutral", 2: "positive"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BertReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def eval_epoch(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), correct / total, np.array(all_preds), np.array(all_labels)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train():
    # ---- Load data ---------------------------------------------------------
    df = load_processed(CONFIG["data_path"])
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, text_col="text"   # BERT uses raw text — tokenizer handles subwords
    )

    # ---- Tokenizer ---------------------------------------------------------
    print(f"Loading tokenizer: {CONFIG['model_name']}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(CONFIG["model_name"])

    # ---- Datasets ----------------------------------------------------------
    print("Tokenizing datasets (this takes a few minutes)...")
    train_ds = BertReviewDataset(X_train, y_train, tokenizer, CONFIG["max_seq_len"])
    val_ds   = BertReviewDataset(X_val,   y_val,   tokenizer, CONFIG["max_seq_len"])
    test_ds  = BertReviewDataset(X_test,  y_test,  tokenizer, CONFIG["max_seq_len"])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

    # ---- Model -------------------------------------------------------------
    print(f"Loading model: {CONFIG['model_name']}")
    model = DistilBertForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=CONFIG["num_labels"],
    ).to(DEVICE)

    # ---- Optimizer + Scheduler ---------------------------------------------
    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    total_steps = len(train_loader) * CONFIG["epochs"]
    warmup_steps = int(CONFIG["warmup_ratio"] * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ---- Training loop -----------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader)

        print(
            f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_pretrained(CONFIG["model_save_dir"])
            tokenizer.save_pretrained(CONFIG["model_save_dir"])
            print("  ✓ Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoch}.")
                break

    # ---- Test evaluation ---------------------------------------------------
    print("\nLoading best model for test evaluation...")
    model = DistilBertForSequenceClassification.from_pretrained(CONFIG["model_save_dir"]).to(DEVICE)
    _, _, y_pred, y_true = eval_epoch(model, test_loader)

    results = compute_metrics(y_true, y_pred, model_name="DistilBERT (Test)")
    plot_confusion_matrix(
        y_true, y_pred,
        model_name="DistilBERT — Test",
        save_path=os.path.join(CONFIG["model_save_dir"], "bert_confusion_test.png")
    )

    return model, results


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(text: str, model=None, tokenizer=None) -> dict:
    """
    Predict sentiment for a single raw text string.
    Loads model + tokenizer from disk if not passed in.
    """
    save_dir = CONFIG["model_save_dir"]

    if tokenizer is None:
        tokenizer = DistilBertTokenizerFast.from_pretrained(save_dir)
    if model is None:
        model = DistilBertForSequenceClassification.from_pretrained(save_dir)
    model.eval()

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=CONFIG["max_seq_len"],
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoding)
        proba = torch.softmax(outputs.logits, dim=1)[0].numpy()
        pred = int(np.argmax(proba))

    return {
        "sentiment": LABEL_MAP_INV[pred],
        "confidence": float(np.max(proba)),
        "probabilities": {
            "negative": float(proba[0]),
            "neutral": float(proba[1]),
            "positive": float(proba[2]),
        }
    }


if __name__ == "__main__":
    train()