import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from langdetect import detect_langs
import pandas as pd
import os
import shutil
from transformers.utils.hub import TRANSFORMERS_CACHE

# Use /tmp for caching instead of ~/.cache/huggingface
cache_dir = "/tmp/huggingface"

# Ensure cache directory exists but DO NOT delete it
os.makedirs(cache_dir, exist_ok=True)

HUGGINGFACE_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN", "")

# Download required NLTK resources with caching
@st.cache_resource
def download_nltk_resources():
    nltk.download("punkt")
    nltk.download('averaged_perceptron_tagger')

download_nltk_resources()

# Caching models for performance
@st.cache_resource(show_spinner=True)
def load_sentiment_model():
    model_name = "Anirudh2857/sentiment_model"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN, force_download=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=HUGGINGFACE_TOKEN, force_download=True)
    return tokenizer, model

@st.cache_resource(show_spinner=True)
def load_qa_model():
    qa_model_name = "Anirudh2857/qa_model"
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name, token=HUGGINGFACE_TOKEN, force_download=True)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name, token=HUGGINGFACE_TOKEN, force_download=True)
    return qa_tokenizer, qa_model

# Load models dynamically
try:
    tokenizer, model = load_sentiment_model()
    st.write("✅ Sentiment model loaded successfully!")
except Exception as e:
    st.error(f"🚨 Error loading Sentiment model: {str(e)}")

try:
    qa_tokenizer, qa_model = load_qa_model()
    st.write("✅ QA model loaded successfully!")
except Exception as e:
    st.error(f"🚨 Error loading QA model: {str(e)}")

nlp = spacy.load("en_core_web_sm")

# Initialize pipelines
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", token=HUGGINGFACE_TOKEN, force_download=True)
    st.write("✅ Summarization model loaded successfully!")
except Exception as e:
    st.error(f"🚨 Error loading summarization model: {str(e)}")

try:
    emotion_classifier = pipeline('text-classification', model='bhadresh-savani/distilbert-base-uncased-emotion', token=HUGGINGFACE_TOKEN, force_download=True)
    st.write("✅ Emotion detection model loaded successfully!")
except Exception as e:
    st.error(f"🚨 Error loading emotion detection model: {str(e)}")

# Streamlit UI
st.title("📝 Interactive NLP Web App")
st.sidebar.title("🔧 Settings")
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")
color_scheme = st.sidebar.selectbox("🎨 Word Cloud Color", ["white", "black", "gray"])

if dark_mode:
    st.markdown(
        """
        <style>
            body { background-color: #121212; color: white; }
        </style>
        """, unsafe_allow_html=True
    )

option = st.sidebar.selectbox("📌 Choose an NLP Task", 
    ["Sentiment Analysis", "Question Answering", "Summarization", "Word Cloud", "NER", "POS Tagging", 
     "Language Detection", "Emotion Detection"])

# NLP Functionalities
if option == "Sentiment Analysis":
    user_input = st.text_area("✍️ Enter text for sentiment analysis")
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            labels = ["Negative", "Positive"]
            sentiment = labels[torch.argmax(scores)]
            confidence = scores.max().item()
            st.success(f"**Sentiment:** {sentiment} (Confidence: {confidence:.2f})")
        else:
            st.warning("Please enter some text to analyze.")

elif option == "Question Answering":
    context = st.text_area("📖 Enter context")
    question = st.text_input("❓ Enter question")
    if st.button("Get Answer"):
        if context.strip() and question.strip():
            inputs = qa_tokenizer(question, context, return_tensors="pt", truncation=True, padding=True)
            outputs = qa_model(**inputs)
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            answer = qa_tokenizer.convert_tokens_to_string(
                qa_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end])
            )
            st.info(f"**Answer:** {answer}")
        else:
            st.warning("Please enter both a context and a question.")
