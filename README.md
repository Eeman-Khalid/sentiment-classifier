# 🛍️ Product Review Sentiment Classification

**NLP Semester Project | FAST-NUCES**  
**Student:** Eeman Khalid · 22F-3173 · BAI-8A

---

## Overview

Multi-class sentiment classification (Positive / Neutral / Negative) on e-commerce product reviews using three progressively advanced NLP models:

| Model | Approach | Trained On |
|---|---|---|
| Logistic Regression | TF-IDF (baseline) | Local / Kaggle |
| BiLSTM | Custom word embeddings | Kaggle (GPU) |
| DistilBERT | Transformer fine-tuning | Kaggle (GPU) |

---

## Datasets

| Dataset | Role | Size | Source |
|---|---|---|---|
| Amazon Product Reviews | Primary (train/test) | ~34K rows | [Kaggle](https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset) |
| Women's Clothing Reviews | Secondary (cross-validation) | ~23K rows | [Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) |

---

## Project Structure

```
sentiment-classifier/
├── app/
│   └── streamlit_app.py      # Streamlit UI
├── data/
│   └── raw/                  # Put downloaded CSVs here (gitignored)
├── models/
│   └── saved/                # Trained model weights (gitignored)
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_logistic_regression.ipynb
│   ├── 03_lstm.ipynb
│   └── 04_distilbert.ipynb
├── src/
│   ├── preprocess.py
│   ├── train_lr.py
│   ├── train_lstm.py
│   ├── train_bert.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/sentiment-classifier.git
cd sentiment-classifier

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Workflow

### Step 1 — Download datasets
Place the CSV files in `data/raw/`:
- `data/raw/amazon_reviews.csv`
- `data/raw/clothing_reviews.csv`

### Step 2 — Preprocess (run on Kaggle or locally)
Upload `notebooks/01_preprocessing.ipynb` to Kaggle, add both datasets, and run all cells.  
Download the output CSVs (`amazon_clean.csv`, `clothing_clean.csv`) and place them in `data/processed/`.

### Step 3 — Train Logistic Regression (locally)
```bash
python src/train_lr.py
```

### Step 4 — Train LSTM & DistilBERT (on Kaggle with GPU)
Upload `notebooks/03_lstm.ipynb` and `notebooks/04_distilbert.ipynb` to Kaggle.  
After training, download model weights and place them:
- LSTM → `models/saved/lstm_model.pt` + `models/saved/lstm_vocab.json`
- DistilBERT → `models/saved/bert/` (folder)

### Step 5 — Run the UI
```bash
streamlit run app/streamlit_app.py
```

---

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score (weighted)
- Confusion Matrix (per model)
- Model Comparison Chart

---

## Base Paper

> *Amazon products reviews classification based on machine learning, deep learning methods and BERT* (2023)  
> TELKOMNIKA Telecommunication Computing Electronics and Control  
> [Link](https://www.telkomnika.uad.ac.id/index.php/TELKOMNIKA/article/view/24046/11728)

---

## Bonus Work
- Cross-dataset validation (Amazon → Clothing)
- Hyperparameter tuning (GridSearchCV for LR, manual for LSTM)
- Error analysis on misclassified samples