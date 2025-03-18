# Sentiment Analysis and Question Answering Models

This repository contains the implementation of a **Sentiment Analysis Model** and a **Question Answering Model**, both fine-tuned using **Hugging Face Transformers**.

## Models Overview

### 1. Sentiment Analysis Model
- Uses **DistilBERT** for binary classification (Positive/Negative sentiment).
- Trained on the **IMDB dataset**.
- Fine-tuned using the **Trainer API**.

### 2. Question Answering (QA) Model
- Uses **DistilBERT** fine-tuned on the **SQuAD dataset**.
- Supports extractive question answering.
- Implements **early stopping** and dropout adjustments to prevent overfitting.

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd <repository-folder>
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

## Training the Models

### Sentiment Analysis Model

1. Run the following command to train the sentiment model:
```bash
python train_sentiment.py
```
2. The trained model will be saved in the `./sentiment_model` directory.

### Question Answering Model

1. Run the following command to train the QA model:
```bash
python train_qa.py
```
2. The trained model will be saved in the `./qa_model` directory.

## Model Details

### Sentiment Analysis
- **Base Model**: `distilbert-base-uncased`
- **Dataset**: IMDB
- **Batch Size**: 8
- **Epochs**: 3
- **Optimizer**: AdamW
- **Evaluation Strategy**: Per epoch

### Question Answering
- **Base Model**: `distilbert-base-uncased`
- **Dataset**: SQuAD
- **Batch Size**: 16
- **Epochs**: 5 (with early stopping)
- **Dropout**: 0.3
- **Mixed Precision**: Enabled (`fp16=True`)
- **Evaluation Strategy**: Per epoch

## Saving and Loading Models

### Saving
Both models are automatically saved after training using:
```python
model.save_pretrained("./model_directory")
tokenizer.save_pretrained("./model_directory")
```

### Loading
To use the trained models:
```python
from transformers import AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoTokenizer

# Load Sentiment Model
tokenizer = AutoTokenizer.from_pretrained("./sentiment_model")
model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model")

# Load QA Model
qa_tokenizer = AutoTokenizer.from_pretrained("./qa_model")
qa_model = AutoModelForQuestionAnswering.from_pretrained("./qa_model")
```

## License
This project is licensed under the MIT License.

## Author
Developed by **Anirudh Jeevan**.
