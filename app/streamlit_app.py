"""
streamlit_app.py
----------------
Sentiment Classifier UI — supports all three models.
Run with: streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Classifier",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .positive  { background: #d4edda; color: #155724; border: 2px solid #28a745; }
    .neutral   { background: #fff3cd; color: #856404; border: 2px solid #ffc107; }
    .negative  { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ---------------------------------------------------------------------------
MODEL_DIR = "models/saved"

@st.cache_resource(show_spinner="Loading Logistic Regression model...")
def load_lr():
    import joblib
    path = os.path.join(MODEL_DIR, "lr_pipeline.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner="Loading LSTM model...")
def load_lstm():
    import torch, json
    from src.train_lstm import BiLSTMClassifier, CONFIG as LSTM_CFG

    vocab_path = os.path.join(MODEL_DIR, "lstm_vocab.json")
    model_path = os.path.join(MODEL_DIR, "lstm_model.pt")
    if not os.path.exists(vocab_path) or not os.path.exists(model_path):
        return None, None

    with open(vocab_path) as f:
        vocab = json.load(f)

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=LSTM_CFG["embed_dim"],
        hidden_dim=LSTM_CFG["hidden_dim"],
        num_layers=LSTM_CFG["num_layers"],
        num_classes=LSTM_CFG["num_classes"],
        dropout=LSTM_CFG["dropout"],
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, vocab


@st.cache_resource(show_spinner="Loading DistilBERT model...")
def load_bert():
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    bert_dir = os.path.join(MODEL_DIR, "bert")
    if not os.path.exists(bert_dir):
        return None, None
    tokenizer = DistilBertTokenizerFast.from_pretrained(bert_dir)
    model = DistilBertForSequenceClassification.from_pretrained(bert_dir)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_lr(text: str, pipeline) -> dict:
    label_map_inv = {0: "negative", 1: "neutral", 2: "positive"}
    proba = pipeline.predict_proba([text])[0]
    pred = int(pipeline.predict([text])[0])
    return {
        "sentiment": label_map_inv[pred],
        "confidence": float(np.max(proba)),
        "probabilities": {"negative": float(proba[0]), "neutral": float(proba[1]), "positive": float(proba[2])},
    }


def predict_lstm(text: str, model, vocab) -> dict:
    import torch
    from src.preprocess import clean_text
    from src.train_lstm import encode, CONFIG as LSTM_CFG

    label_map_inv = {0: "negative", 1: "neutral", 2: "positive"}
    cleaned = clean_text(text)
    ids = encode(cleaned, vocab, LSTM_CFG["max_seq_len"])
    tensor = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(tensor)
        proba = torch.softmax(logits, dim=1)[0].numpy()
        pred = int(np.argmax(proba))
    return {
        "sentiment": label_map_inv[pred],
        "confidence": float(np.max(proba)),
        "probabilities": {"negative": float(proba[0]), "neutral": float(proba[1]), "positive": float(proba[2])},
    }


def predict_bert(text: str, model, tokenizer) -> dict:
    import torch
    from src.train_bert import CONFIG as BERT_CFG

    label_map_inv = {0: "negative", 1: "neutral", 2: "positive"}
    encoding = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=BERT_CFG["max_seq_len"], return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**encoding)
        proba = torch.softmax(outputs.logits, dim=1)[0].numpy()
        pred = int(np.argmax(proba))
    return {
        "sentiment": label_map_inv[pred],
        "confidence": float(np.max(proba)),
        "probabilities": {"negative": float(proba[0]), "neutral": float(proba[1]), "positive": float(proba[2])},
    }


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1>🛍️ Product Review Sentiment Classifier</h1>
    <p>NLP Semester Project — Eeman Khalid (22F-3173) | BAI-8A</p>
</div>
""", unsafe_allow_html=True)

# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "Select Model",
        ["Logistic Regression", "BiLSTM", "DistilBERT"],
        help="Choose the model for classification."
    )

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    if model_choice == "Logistic Regression":
        st.info("**Baseline model.** Uses TF-IDF features (unigrams + bigrams). Fast, interpretable.")
    elif model_choice == "BiLSTM":
        st.info("**Deep learning model.** Captures sequential patterns using bidirectional LSTM layers.")
    else:
        st.info("**Transformer model.** DistilBERT fine-tuned for 3-class sentiment. Best performance.")

    st.markdown("---")
    st.markdown("### 🏷️ Labels")
    st.markdown("- 🟢 **Positive** — 5★ reviews")
    st.markdown("- 🟡 **Neutral** — 3★ or 4★ reviews")
    st.markdown("- 🔴 **Negative** — 1★ or 2★ reviews")

# ---- Main tabs -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📝 Single Review", "📋 Batch Reviews", "📊 Model Comparison"])

# ---- Tab 1: Single Review --------------------------------------------------
with tab1:
    st.subheader("Classify a Single Review")
    user_input = st.text_area(
        "Enter a product review:",
        placeholder="e.g. The quality is amazing, fits perfectly and arrived on time!",
        height=150,
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        predict_btn = st.button("🔍 Analyse", use_container_width=True, type="primary")

    if predict_btn and user_input.strip():
        with st.spinner(f"Running {model_choice}..."):
            try:
                if model_choice == "Logistic Regression":
                    pipeline = load_lr()
                    if pipeline is None:
                        st.error("Model not found. Please train it first: `python src/train_lr.py`")
                        st.stop()
                    result = predict_lr(user_input, pipeline)

                elif model_choice == "BiLSTM":
                    model, vocab = load_lstm()
                    if model is None:
                        st.error("LSTM model not found. Please train on Kaggle and place weights in models/saved/")
                        st.stop()
                    result = predict_lstm(user_input, model, vocab)

                else:
                    model, tokenizer = load_bert()
                    if model is None:
                        st.error("DistilBERT model not found. Please train on Kaggle and place weights in models/saved/bert/")
                        st.stop()
                    result = predict_bert(user_input, model, tokenizer)

                # ---- Display result ----------------------------------------
                sentiment = result["sentiment"]
                confidence = result["confidence"]
                probs = result["probabilities"]

                emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}[sentiment]
                st.markdown(
                    f'<div class="result-box {sentiment}">'
                    f'{emoji} Sentiment: {sentiment.upper()} &nbsp;|&nbsp; Confidence: {confidence:.1%}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Probability bar chart
                fig = go.Figure(go.Bar(
                    x=["Negative", "Neutral", "Positive"],
                    y=[probs["negative"], probs["neutral"], probs["positive"]],
                    marker_color=["#dc3545", "#ffc107", "#28a745"],
                    text=[f"{v:.1%}" for v in [probs["negative"], probs["neutral"], probs["positive"]]],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Prediction Probabilities",
                    yaxis=dict(range=[0, 1.1], tickformat=".0%"),
                    showlegend=False,
                    height=350,
                    margin=dict(t=50, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")

    elif predict_btn:
        st.warning("Please enter a review first.")

# ---- Tab 2: Batch Reviews --------------------------------------------------
with tab2:
    st.subheader("Classify Multiple Reviews")
    st.markdown("Enter one review per line:")
    batch_input = st.text_area("Reviews (one per line):", height=200, placeholder="Great product!\nTerrible quality, broke after a day.\nIt was okay, nothing special.")

    if st.button("🔍 Analyse All", type="primary"):
        reviews = [r.strip() for r in batch_input.strip().split("\n") if r.strip()]
        if not reviews:
            st.warning("Please enter at least one review.")
        else:
            with st.spinner("Processing..."):
                results = []
                for review in reviews:
                    try:
                        if model_choice == "Logistic Regression":
                            pipeline = load_lr()
                            r = predict_lr(review, pipeline)
                        elif model_choice == "BiLSTM":
                            model, vocab = load_lstm()
                            r = predict_lstm(review, model, vocab)
                        else:
                            model, tokenizer = load_bert()
                            r = predict_bert(review, model, tokenizer)
                        results.append({"Review": review[:80] + "..." if len(review) > 80 else review,
                                        "Sentiment": r["sentiment"].upper(),
                                        "Confidence": f"{r['confidence']:.1%}"})
                    except Exception as e:
                        results.append({"Review": review[:80], "Sentiment": "ERROR", "Confidence": str(e)})

                import pandas as pd
                result_df = pd.DataFrame(results)

                # Colour rows
                def color_sentiment(val):
                    colors = {"POSITIVE": "background-color: #d4edda", "NEUTRAL": "background-color: #fff3cd", "NEGATIVE": "background-color: #f8d7da"}
                    return colors.get(val, "")

                st.dataframe(
                    result_df.style.applymap(color_sentiment, subset=["Sentiment"]),
                    use_container_width=True,
                )

                # Distribution pie chart
                counts = result_df["Sentiment"].value_counts()
                fig = go.Figure(go.Pie(
                    labels=counts.index, values=counts.values,
                    marker=dict(colors=["#28a745", "#ffc107", "#dc3545"]),
                    hole=0.4,
                ))
                fig.update_layout(title="Sentiment Distribution", height=350)
                st.plotly_chart(fig, use_container_width=True)

# ---- Tab 3: Model Comparison -----------------------------------------------
with tab3:
    st.subheader("Model Performance Comparison")
    st.markdown("These results are loaded from saved evaluation outputs after training on the Amazon Reviews dataset.")

    # Static placeholder results — replace with actual values after training
    comparison_data = {
        "Model": ["Logistic Regression", "BiLSTM", "DistilBERT"],
        "Accuracy": [0.000, 0.000, 0.000],
        "Precision": [0.000, 0.000, 0.000],
        "Recall": [0.000, 0.000, 0.000],
        "F1-Score": [0.000, 0.000, 0.000],
    }

    results_path = os.path.join(MODEL_DIR, "comparison_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            saved = json.load(f)
        for i, model_name in enumerate(comparison_data["Model"]):
            if model_name in saved:
                for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
                    comparison_data[metric][i] = saved[model_name][metric.lower().replace("-", "")]
        st.success("✓ Loaded saved evaluation results.")
    else:
        st.info("Train all three models and save results to `models/saved/comparison_results.json` to see the chart.")

    import pandas as pd
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df.set_index("Model"), use_container_width=True)

    # Grouped bar chart
    fig = go.Figure()
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for i, model_name in enumerate(comparison_data["Model"]):
        fig.add_trace(go.Bar(
            name=model_name,
            x=metrics,
            y=[comparison_data[m][i] for m in metrics],
            marker_color=colors[i],
            text=[f"{comparison_data[m][i]:.3f}" for m in metrics],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        title="Model Comparison — All Metrics",
        yaxis=dict(range=[0, 1.15]),
        height=450,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:grey; font-size:0.85rem'>"
    "NLP Semester Project · Product Review Sentiment Classification · FAST-NUCES Lahore"
    "</div>",
    unsafe_allow_html=True,
)