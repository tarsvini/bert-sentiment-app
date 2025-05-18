import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re
import pandas as pd

# ✅ Page config must be set immediately after imports
st.set_page_config(page_title="Haircare Sentiment Analyzer", layout="centered")

# Clean input review text
def clean_text_for_bert(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load tokenizer and model only once
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("./bert_tokenizer", local_files_only=True)
    model = BertForSequenceClassification.from_pretrained("./bert_model", local_files_only=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Label mapping
label_map = {0: "Positive", 1: "Negative", 2: "Neutral"}

def predict_sentiment(text):
    cleaned = clean_text_for_bert(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return label_map[pred], confidence

st.title("💬 Haircare Product Review Sentiment Analysis")

# Choose input method
mode = st.radio("Choose input mode:", ("Single Review", "Batch Upload (CSV)"))

if mode == "Single Review":
    review = st.text_area("📝 Enter a product review below:")
    if st.button("🔍 Analyze Sentiment"):
        if not review.strip():
            st.warning("Please enter a review.")
        else:
            try:
                label, conf = predict_sentiment(review)
                st.success(f"**Predicted Sentiment:** {label}  \n**Confidence:** {conf:.2f}")
            except Exception as e:
                st.error(f"⚠️ Error during prediction: {str(e)}")

else:
    uploaded_file = st.file_uploader("Upload CSV file with a 'review' column", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "review" not in df.columns:
                st.error("CSV must contain a 'review' column.")
            else:
                results = []
                for text in df["review"]:
                    label, conf = predict_sentiment(str(text))
                    results.append({"review": text, "predicted_sentiment": label, "confidence": round(conf, 2)})
                result_df = pd.DataFrame(results)
                st.dataframe(result_df)
        except Exception as e:
            st.error(f"Error reading file or making predictions: {e}")
