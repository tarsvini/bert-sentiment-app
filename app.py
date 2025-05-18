import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re
import pandas as pd

# Page config
st.set_page_config(page_title="Haircare Sentiment Analyzer", layout="centered")

# Clean input text
def clean_text_for_bert(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load model and tokenizer (cached)
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("./bert_tokenizer", local_files_only=True)
    model = BertForSequenceClassification.from_pretrained("./bert_model", local_files_only=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

label_map = {0: "Positive", 1: "Negative", 2: "Neutral"}

# Title
st.title("💬 Haircare Product Review Sentiment Analysis")

# Tabs for single or batch input
tab1, tab2 = st.tabs(["Single Review", "Batch Upload"])

with tab1:
    review = st.text_area("📝 Enter a product review below:")
    if st.button("🔍 Analyze Sentiment", key="single"):
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

with tab2:
    st.write("Upload a CSV file with a column named `review` to analyze multiple reviews at once.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "review" not in df.columns:
                st.error("CSV must contain a 'review' column.")
            else:
                # Clean and predict batch
                def predict_sentiment(text):
                    cleaned = clean_text_for_bert(text)
                    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                        pred = torch.argmax(probs, dim=1).item()
                        confidence = probs[0][pred].item()
                    return label_map[pred], confidence

                with st.spinner("Analyzing batch..."):
                    results = df["review"].apply(predict_sentiment)
                    df["Predicted Sentiment"] = results.apply(lambda x: x[0])
                    df["Confidence"] = results.apply(lambda x: x[1])

                st.dataframe(df)
                st.success("Batch analysis complete!")
        except Exception as e:
            st.error(f"⚠️ Error processing file: {str(e)}")

# Confidence explanation expandable
with st.expander("ℹ️ What does Confidence mean?"):
    st.markdown("""
    **Confidence** is a number between 0 and 1 (or 0% to 100%) indicating how sure the model is about its prediction.
    
    - A higher confidence means the model is more certain about the predicted sentiment.
    - For example, a confidence of **0.90 (90%)** means the model believes there is a 90% chance the prediction is correct.
    - Low confidence scores suggest the review might be ambiguous or hard to classify accurately.
    
    Use the confidence score to understand how much trust to place in the prediction.
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by *Tarsvini Ravinther*")
