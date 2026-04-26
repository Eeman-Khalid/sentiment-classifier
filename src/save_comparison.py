"""
save_comparison.py
------------------
Merges lr_results.json + lstm_results.json + bert_results.json
into one comparison_results.json for the Streamlit UI.

Run from project root:
    python src/save_comparison.py
"""

import os, json

MODEL_DIR = "models/saved/"

def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def main():
    combined = {}

    # ── Logistic Regression ──────────────────────────────────────────────
    # Reads from lr_results.json (its own dedicated file, never overwritten)
    lr_data = load_json(os.path.join(MODEL_DIR, "lr_results.json"))
    if "Logistic Regression" in lr_data:
        combined["Logistic Regression"] = lr_data["Logistic Regression"]
        print(f"[OK] Logistic Regression: {combined['Logistic Regression']}")
    else:
        print("[MISSING] Logistic Regression — re-run notebook 02 (it now saves lr_results.json separately)")

    # ── BiLSTM ──────────────────────────────────────────────────────────
    lstm_data = load_json(os.path.join(MODEL_DIR, "lstm_results.json"))
    if "BiLSTM" in lstm_data:
        combined["BiLSTM"] = lstm_data["BiLSTM"]
        print(f"[OK] BiLSTM: {combined['BiLSTM']}")
    else:
        print("[MISSING] BiLSTM — download lstm_results.json from Kaggle output tab")

    # ── DistilBERT ──────────────────────────────────────────────────────
    bert_data = load_json(os.path.join(MODEL_DIR, "bert_results.json"))
    if "DistilBERT" in bert_data:
        combined["DistilBERT"] = bert_data["DistilBERT"]
        print(f"[OK] DistilBERT: {combined['DistilBERT']}")
    else:
        print("[MISSING] DistilBERT — download bert_results.json from Kaggle output tab")

    if not combined:
        print("\nNo results found. Train at least one model first.")
        return

    # Write final merged file (this is what Streamlit reads)
    out = os.path.join(MODEL_DIR, "comparison_results.json")
    with open(out, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved → {out}")
    print(json.dumps(combined, indent=2))

if __name__ == "__main__":
    main()
