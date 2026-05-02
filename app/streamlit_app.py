"""
streamlit_app.py
----------------
Sentiment Classifier UI - Complete Version
Author: Eeman Khalid (22F-3173)
Run: streamlit run app/streamlit_app.py
"""

import os, sys, json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import joblib
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = "models/saved"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.header-box {
    text-align:center; padding:20px;
    background:linear-gradient(90deg,#1f1c2c,#928dab);
    border-radius:15px; color:white; margin-bottom:20px;
}
.result-positive { background:#d4edda; color:#155724; border:2px solid #28a745;
    padding:15px; border-radius:10px; text-align:center; font-size:1.3rem; font-weight:bold; }
.result-neutral  { background:#fff3cd; color:#856404; border:2px solid #ffc107;
    padding:15px; border-radius:10px; text-align:center; font-size:1.3rem; font-weight:bold; }
.result-negative { background:#f8d7da; color:#721c24; border:2px solid #dc3545;
    padding:15px; border-radius:10px; text-align:center; font-size:1.3rem; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h1>🧠 Sentiment Analysis System</h1>
    <h4>Eeman Khalid | 22F-3173 | NLP Semester Project</h4>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📌 Project Overview")
    st.markdown("""
### 🔹 Models
- Logistic Regression (Baseline)
- BiLSTM (Deep Learning)
- DistilBERT (Transformer)

### 🔹 Datasets
- Amazon Product Reviews (Primary)
- Women's Clothing Reviews (Cross-Val)

### 🔹 Classes
- 😊 Positive (5★)
- 😐 Neutral (3–4★)
- 😞 Negative (1–2★)

### 🔹 Techniques
- TF-IDF + Balanced Class Weights
- Oversampling (minority classes)
- Weighted CrossEntropyLoss
- Early Stopping
""")

LABEL_MAP_INV = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_NAMES   = ["negative", "neutral", "positive"]

# ═══════════════════════════════════════════════════════════════════
# MODEL LOADERS (cached — load once per session)
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading Logistic Regression...")
def load_lr():
    path = os.path.join(MODEL_DIR, "lr_pipeline.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_resource(show_spinner="Loading BiLSTM...")
def load_lstm():
    vocab_path  = os.path.join(MODEL_DIR, "lstm_vocab.json")
    model_path  = os.path.join(MODEL_DIR, "lstm_model.pt")
    config_path = os.path.join(MODEL_DIR, "lstm_config.json")
    if not all(os.path.exists(p) for p in [vocab_path, model_path, config_path]):
        return None, None, None

    with open(vocab_path)  as f: vocab = json.load(f)
    with open(config_path) as f: cfg   = json.load(f)

    class BiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                                batch_first=True, bidirectional=True,
                                dropout=dropout if num_layers > 1 else 0)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)
        def forward(self, x):
            out, _ = self.lstm(self.dropout(self.embedding(x)))
            return self.fc(self.dropout(out.mean(dim=1)))

    model = BiLSTMClassifier(
        vocab_size=len(vocab), embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"],
        num_classes=cfg["num_classes"], dropout=cfg["dropout"]
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, vocab, cfg

@st.cache_resource(show_spinner="Loading DistilBERT...")
def load_bert():
    bert_dir = os.path.join(MODEL_DIR, "bert")
    if not os.path.exists(bert_dir):
        return None, None
    try:
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        tokenizer = DistilBertTokenizerFast.from_pretrained(bert_dir)
        model     = DistilBertForSequenceClassification.from_pretrained(bert_dir)
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"BERT load error: {e}")
        return None, None

# ═══════════════════════════════════════════════════════════════════
# PREDICTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def predict_lr(text, pipeline):
    proba     = pipeline.predict_proba([text])[0]
    pred      = int(pipeline.predict([text])[0])
    return {
        "sentiment":     LABEL_MAP_INV[pred],
        "confidence":    float(np.max(proba)),
        "probabilities": {"negative": float(proba[0]),
                          "neutral":  float(proba[1]),
                          "positive": float(proba[2])}
    }

def predict_lstm(text, model, vocab, cfg):
    from src.preprocess import clean_text
    cleaned = clean_text(text)
    tokens  = cleaned.split()[:cfg["max_seq_len"]]
    ids     = [vocab.get(t, 1) for t in tokens]
    ids    += [0] * (cfg["max_seq_len"] - len(ids))
    tensor  = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        proba = torch.softmax(model(tensor), dim=1)[0].numpy()
    pred = int(np.argmax(proba))
    return {
        "sentiment":     LABEL_MAP_INV[pred],
        "confidence":    float(np.max(proba)),
        "probabilities": {"negative": float(proba[0]),
                          "neutral":  float(proba[1]),
                          "positive": float(proba[2])}
    }

def predict_bert(text, model, tokenizer):
    enc = tokenizer(text, truncation=True, padding="max_length",
                    max_length=128, return_tensors="pt")
    with torch.no_grad():
        proba = torch.softmax(model(**enc).logits, dim=1)[0].numpy()
    pred = int(np.argmax(proba))
    return {
        "sentiment":     LABEL_MAP_INV[pred],
        "confidence":    float(np.max(proba)),
        "probabilities": {"negative": float(proba[0]),
                          "neutral":  float(proba[1]),
                          "positive": float(proba[2])}
    }

# ── Helper: display result ────────────────────────────────────────
def show_result(result):
    sentiment  = result["sentiment"]
    confidence = result["confidence"]
    probs      = result["probabilities"]
    emoji      = {"positive": "😊", "neutral": "😐", "negative": "😞"}[sentiment]

    st.markdown(
        f'<div class="result-{sentiment}">'
        f'{emoji} {sentiment.upper()} &nbsp;|&nbsp; Confidence: {confidence:.1%}'
        f'</div>', unsafe_allow_html=True
    )
    st.markdown("####")

    fig = go.Figure(go.Bar(
        x=["Negative", "Neutral", "Positive"],
        y=[probs["negative"], probs["neutral"], probs["positive"]],
        marker_color=["#dc3545", "#ffc107", "#28a745"],
        text=[f"{v:.1%}" for v in [probs["negative"], probs["neutral"], probs["positive"]]],
        textposition="outside"
    ))
    fig.update_layout(
        title="Prediction Probabilities",
        yaxis=dict(range=[0, 1.15], tickformat=".0%"),
        height=350, showlegend=False,
        margin=dict(t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Helper: run prediction for selected model ─────────────────────
def run_prediction(text, model_choice):
    if model_choice == "Logistic Regression":
        pipeline = load_lr()
        if pipeline is None:
            st.error("❌ lr_pipeline.pkl not found in models/saved/")
            return None
        return predict_lr(text, pipeline)

    elif model_choice == "BiLSTM":
        model, vocab, cfg = load_lstm()
        if model is None:
            st.error("❌ LSTM files not found. Need: lstm_model.pt, lstm_vocab.json, lstm_config.json")
            return None
        return predict_lstm(text, model, vocab, cfg)

    else:  # DistilBERT
        model, tokenizer = load_bert()
        if model is None:
            st.error("❌ BERT model not found in models/saved/bert/")
            return None
        return predict_bert(text, model, tokenizer)

# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📝 Single Review", "📋 Batch Reviews", "📊 Model Comparison"])

# ── TAB 1: Single Review ──────────────────────────────────────────
with tab1:
    st.subheader("Classify a Single Review")

    model_choice = st.selectbox("Select Model", 
                                ["Logistic Regression", "BiLSTM", "DistilBERT"],
                                key="single_model")
    user_input   = st.text_area("Enter a product review:", height=150,
                                placeholder="e.g. The quality is amazing, arrived fast!")

    if st.button("🔍 Analyse", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a review.")
        else:
            with st.spinner(f"Running {model_choice}..."):
                try:
                    result = run_prediction(user_input, model_choice)
                    if result:
                        show_result(result)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# ── TAB 2: Batch Reviews ──────────────────────────────────────────
with tab2:
    st.subheader("Classify Multiple Reviews at Once")

    model_choice_b = st.selectbox("Select Model",
                                  ["Logistic Regression", "BiLSTM", "DistilBERT"],
                                  key="batch_model")
    batch_input    = st.text_area("Enter one review per line:", height=200,
                                  placeholder="Great product!\nTerrible quality.\nIt was okay.")

    if st.button("🔍 Analyse All", type="primary"):
        reviews = [r.strip() for r in batch_input.strip().split("\n") if r.strip()]
        if not reviews:
            st.warning("Please enter at least one review.")
        else:
            results = []
            progress = st.progress(0)
            for idx, review in enumerate(reviews):
                try:
                    r = run_prediction(review, model_choice_b)
                    if r:
                        results.append({
                            "Review":     review[:80] + ("..." if len(review) > 80 else ""),
                            "Sentiment":  r["sentiment"].upper(),
                            "Confidence": f"{r['confidence']:.1%}"
                        })
                except Exception as e:
                    results.append({"Review": review[:80], "Sentiment": "ERROR", "Confidence": str(e)})
                progress.progress((idx + 1) / len(reviews))

            import pandas as pd
            df = pd.DataFrame(results)

            def colour(val):
                c = {"POSITIVE": "background-color:#d4edda",
                     "NEUTRAL":  "background-color:#fff3cd",
                     "NEGATIVE": "background-color:#f8d7da"}
                return c.get(val, "")

            st.dataframe(df.style.applymap(colour, subset=["Sentiment"]),
                         use_container_width=True)

            # Pie chart
            counts = df["Sentiment"].value_counts()
            fig = go.Figure(go.Pie(
                labels=counts.index, values=counts.values, hole=0.4,
                marker=dict(colors=["#28a745", "#ffc107", "#dc3545"])
            ))
            fig.update_layout(title="Sentiment Distribution", height=350)
            st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: Model Comparison ───────────────────────────────────────
with tab3:
    st.subheader("Model Performance Comparison")

    comp_path = os.path.join(MODEL_DIR, "comparison_results.json")
    if not os.path.exists(comp_path):
        st.warning("comparison_results.json not found. Run: python src/save_comparison.py")
    else:
        with open(comp_path) as f:
            data = json.load(f)

        models  = ["Logistic Regression", "BiLSTM", "DistilBERT"]
        metrics = ["accuracy", "precision", "recall", "f1"]
        colors  = ["#4C72B0", "#DD8452", "#55A868"]

        # Table
        import pandas as pd
        rows = []
        for m in models:
            if m in data:
                rows.append({
                    "Model":     m,
                    "Accuracy":  data[m].get("accuracy",  0),
                    "Precision": data[m].get("precision", 0),
                    "Recall":    data[m].get("recall",    0),
                    "F1-Score":  data[m].get("f1",        0),
                })
        if rows:
            comp_df = pd.DataFrame(rows).set_index("Model")
            st.dataframe(comp_df.style.format("{:.4f}"), use_container_width=True)

            # Bar chart
            fig = go.Figure()
            for i, m in enumerate(models):
                if m in data:
                    fig.add_trace(go.Bar(
                        name=m,
                        x=["Accuracy", "Precision", "Recall", "F1-Score"],
                        y=[data[m].get(k, 0) for k in metrics],
                        marker_color=colors[i],
                        text=[f"{data[m].get(k,0):.3f}" for k in metrics],
                        textposition="outside"
                    ))
            fig.update_layout(
                barmode="group",
                title="All Models — Performance Metrics",
                yaxis=dict(range=[0, 1.15]),
                height=450,
                legend=dict(orientation="h", y=-0.2)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model results found in comparison_results.json")

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:gray; font-size:0.85rem'>
NLP Semester Project · Eeman Khalid · 22F-3173 · BAI-8A · FAST-NUCES
</div>
""", unsafe_allow_html=True)