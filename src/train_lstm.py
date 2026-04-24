"""
train_lstm.py
-------------
Bidirectional LSTM model using PyTorch.
Designed to run on Kaggle (GPU). Saves model weights as lstm_model.pt.

Run on Kaggle:
    !python src/train_lstm.py

Or locally (CPU — slow but works for testing):
    python src/train_lstm.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import load_processed, split_data
from src.evaluate import compute_metrics, plot_confusion_matrix


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG = {
    "data_path": "data/processed/amazon_clean.csv",
    "model_save_dir": "models/saved",
    "vocab_size": 30_000,
    "embed_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "num_classes": 3,
    "max_seq_len": 150,
    "batch_size": 64,
    "epochs": 10,
    "lr": 1e-3,
    "patience": 3,           # early stopping patience
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
os.makedirs(CONFIG["model_save_dir"], exist_ok=True)


# ---------------------------------------------------------------------------
# Vocabulary builder
# ---------------------------------------------------------------------------

def build_vocab(texts, max_vocab=30_000):
    """Build a simple word → index vocabulary from training texts."""
    from collections import Counter
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    # Reserve 0=PAD, 1=UNK
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def encode(text, vocab, max_len):
    """Convert a cleaned text string to a fixed-length integer sequence."""
    tokens = text.split()[:max_len]
    ids = [vocab.get(t, 1) for t in tokens]   # 1 = UNK
    # Pad to max_len
    ids += [0] * (max_len - len(ids))
    return ids


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.encodings = [encode(t, vocab, max_len) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encodings[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        # bidirectional → hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        # lstm_out: (batch, seq_len, hidden*2)
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Use mean pooling over sequence (more stable than last hidden)
        pooled = lstm_out.mean(dim=1)
        out = self.fc(self.dropout(pooled))
        return out


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
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
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # ---- Vocabulary --------------------------------------------------------
    print("Building vocabulary...")
    vocab = build_vocab(X_train, max_vocab=CONFIG["vocab_size"])
    print(f"Vocab size: {len(vocab):,}")

    # Save vocab for inference
    vocab_path = os.path.join(CONFIG["model_save_dir"], "lstm_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)
    print(f"Vocabulary saved → {vocab_path}")

    # ---- Datasets & loaders ------------------------------------------------
    max_len = CONFIG["max_seq_len"]
    train_ds = ReviewDataset(X_train, y_train, vocab, max_len)
    val_ds   = ReviewDataset(X_val,   y_val,   vocab, max_len)
    test_ds  = ReviewDataset(X_test,  y_test,  vocab, max_len)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

    # ---- Model -------------------------------------------------------------
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=CONFIG["embed_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        num_classes=CONFIG["num_classes"],
        dropout=CONFIG["dropout"],
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    # ---- Training loop -----------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Early stopping + checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CONFIG["model_save_dir"], "lstm_model.pt"))
            print("  ✓ Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoch}.")
                break

    # ---- Test evaluation ---------------------------------------------------
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(
        torch.load(os.path.join(CONFIG["model_save_dir"], "lstm_model.pt"), map_location=DEVICE)
    )
    _, _, y_pred, y_true = eval_epoch(model, test_loader, criterion)
    results = compute_metrics(y_true, y_pred, model_name="BiLSTM (Test)")
    plot_confusion_matrix(
        y_true, y_pred,
        model_name="BiLSTM — Test",
        save_path=os.path.join(CONFIG["model_save_dir"], "lstm_confusion_test.png")
    )

    # Save config for inference
    with open(os.path.join(CONFIG["model_save_dir"], "lstm_config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    return model, results


def predict(text: str, model=None, vocab=None) -> dict:
    """
    Predict sentiment for a single cleaned or raw text string.
    Loads model + vocab from disk if not passed in.
    """
    import json
    from src.preprocess import clean_text

    save_dir = CONFIG["model_save_dir"]
    label_map_inv = {0: "negative", 1: "neutral", 2: "positive"}

    if vocab is None:
        with open(os.path.join(save_dir, "lstm_vocab.json")) as f:
            vocab = json.load(f)

    if model is None:
        model = BiLSTMClassifier(
            vocab_size=len(vocab),
            embed_dim=CONFIG["embed_dim"],
            hidden_dim=CONFIG["hidden_dim"],
            num_layers=CONFIG["num_layers"],
            num_classes=CONFIG["num_classes"],
            dropout=CONFIG["dropout"],
        )
        model.load_state_dict(
            torch.load(os.path.join(save_dir, "lstm_model.pt"), map_location="cpu")
        )
    model.eval()

    cleaned = clean_text(text)
    ids = encode(cleaned, vocab, CONFIG["max_seq_len"])
    tensor = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(tensor)
        proba = torch.softmax(logits, dim=1)[0].numpy()
        pred = int(np.argmax(proba))

    return {
        "sentiment": label_map_inv[pred],
        "confidence": float(np.max(proba)),
        "probabilities": {
            "negative": float(proba[0]),
            "neutral": float(proba[1]),
            "positive": float(proba[2]),
        }
    }


if __name__ == "__main__":
    train()