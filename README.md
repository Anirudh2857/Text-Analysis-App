# NLP Web App

This is an interactive NLP web application built using Streamlit, Hugging Face transformers, and other NLP libraries. The app provides various natural language processing functionalities, including Sentiment Analysis, Question Answering, Named Entity Recognition (NER), Part-of-Speech (POS) Tagging, and Word Cloud generation.

## Features
- **Sentiment Analysis**: Determines whether a given text has a positive or negative sentiment.
- **Question Answering**: Extracts answers to questions based on a given context.
- **Named Entity Recognition (NER)**: Identifies entities such as names, locations, and organizations in text.
- **Part-of-Speech (POS) Tagging**: Analyzes words and their corresponding POS tags.
- **Word Cloud Generation**: Creates a visual representation of word frequency in a text.

## Deployment
You can access the deployed application at: [https://text-analysis-app-fletzht2ccefy97zus729q.streamlit.app](#) 

## Installation

To run the application locally, follow these steps:

### 1. Clone the repository
```bash
 git clone https://github.com/Anirudh2857/Text-Analysis-App
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up Hugging Face API Token
Create a Streamlit secrets file (`.streamlit/secrets.toml`) and add your Hugging Face token:
```toml
HUGGINGFACE_TOKEN = "your_huggingface_api_token"
```

### 5. Run the Streamlit app
```bash
streamlit run app.py
```

## Usage
After launching the app, you can select an NLP task from the sidebar:

- **Sentiment Analysis**: Enter text and analyze sentiment.
- **Question Answering**: Provide a context and ask a question to get an answer.
- **NER**: Detect named entities in a given text.
- **POS Tagging**: Analyze words and their grammatical roles.
- **Word Cloud**: Generate a word cloud visualization from text.

## Dependencies
This application uses the following key libraries:
- `streamlit` (for UI)
- `transformers` (for NLP models)
- `torch` (for deep learning inference)
- `spacy` (for NER)
- `nltk` (for tokenization and POS tagging)
- `matplotlib` and `wordcloud` (for visualization)
- `pandas` (for displaying results in tabular format)

## Notes
- The models are loaded from Hugging Face and require authentication via API token.
- The cache is cleared before loading models to prevent corrupted downloads.
- The app uses `@st.cache_resource` to optimize performance by caching model loads.

## License
This project is licensed under the MIT License.

## Author
Developed by **Anirudh Jeevan**.

