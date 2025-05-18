import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re

# Clean input review text
def clean_text_for_bert(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load tokenizer and model only once using Streamlit cache
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("./bert_tokenizer", local_files_only=True)
    model = BertForSequenceClassification.from_pretrained("./bert_model", local_files_only=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Label mapping
label_map = {0: "Positive", 1: "Negative", 2: "Neutral"}

# App UI
st.set_page_config(page_title="Haircare Sentiment Analyzer", layout="centered")
st.title("💬 Haircare Product Review Sentiment Analysis")

review = st.text_area("📝 Enter a product review below:")

if st.button("🔍 Analyze Sentiment"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        try:
            cleaned_review = clean_text_for_bert(review)
            inputs = tokenizer(cleaned_review, return_tensors="pt", truncation=True, padding=True, max_length=128)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred].item()

            st.success(f"**Predicted Sentiment:** {label_map[pred]}  \n**Confidence:** {confidence:.2f}")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {str(e)}")
