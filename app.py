import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import pandas as pd
import os
import shutil
from transformers.utils.hub import TRANSFORMERS_CACHE

# Securely load Hugging Face token from Streamlit secrets
HUGGINGFACE_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN", "")

# Clear Hugging Face cache to prevent corrupted downloads
cache_dir = TRANSFORMERS_CACHE if TRANSFORMERS_CACHE else os.path.expanduser("~/.cache/huggingface")
shutil.rmtree(cache_dir, ignore_errors=True)
os.makedirs(cache_dir, exist_ok=True)

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
    ["Sentiment Analysis", "Question Answering", "Word Cloud", "NER", "POS Tagging"])

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

elif option == "Word Cloud":
    text = st.text_area("🌥️ Enter text for word cloud")
    if st.button("Generate Word Cloud"):
        if text.strip():
            text = text[:5000]  # Prevent huge inputs from crashing the app
            wordcloud = WordCloud(width=800, height=400, max_words=100, background_color=color_scheme).generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.warning("Please enter some text to generate a word cloud.")

elif option == "NER":
    text = st.text_area("📌 Enter text for Named Entity Recognition")
    if st.button("Analyze NER"):
        if text.strip():
            doc = nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            df = pd.DataFrame(entities, columns=["Entity", "Label"])
            st.dataframe(df)
        else:
            st.warning("Please enter some text for analysis.")

elif option == "POS Tagging":
    text = st.text_area("🔤 Enter text for POS Tagging")
    if st.button("Tag POS"):
        if text.strip():
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            df = pd.DataFrame(pos_tags, columns=["Word", "POS Tag"])
            st.dataframe(df)
        else:
            st.warning("Please enter some text for POS tagging.")
