# Auto-Tagging Support Tickets Using LLMs & ML

This project demonstrates how to automatically classify support tickets using two approaches:

- **LLMs (Gemini, BART)** for zero-shot and few-shot classification
- Traditional ML pipeline using TF-IDF + Logistic Regression

We used a real-world dataset with fields like `title`, `body`, and `category` to build a robust and reusable classification pipeline.

---

## Objective

Automatically predict ticket category using LLMs and ML pipeline, and compare:

- Zero-shot (LLMs)
- Few-shot (LLMs)
- Supervised learning (LogReg)

---

## Dataset

Dataset: `support-ticket.csv`

- Rows: \~48,000
- Columns: 9 (title, body, category, etc.)

---

## Libraries Used

```python
!pip install -q -U google-generativeai transformers scikit-learn pandas
```

```python
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
```

---

## Steps Implemented

### Step 1: Load Dataset

- Loaded and viewed shape, columns, and sample rows

### Step 2: Clean and Prepare Text

- Merged `title` and `body` into single `text`
- Dropped rows with missing `category`

### Step 3: Zero-Shot Classification (BART)

- Used `facebook/bart-large-mnli`
- Labeled top category via Hugging Face Transformers

### Step 4: Gemini Zero-shot Prediction

```python
model = genai.GenerativeModel("models/gemini-1.5-flash")
prompt = "You are a support ticket classifier. Text: ..."
response = model.generate_content(prompt)
```

### Step 5: Few-Shot Prompt

- Added 3 labeled examples to prompt
- Sent to Gemini to boost contextual learning

### Step 6: Top 3 Categories from Gemini

- Prompted Gemini to return ranked predictions

### Step 7: ML Model Pipeline

```python
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train)
```

- TF-IDF + Logistic Regression
- Accuracy: \~86%

### Step 8: Compare Gemini vs ML Model

- Sampled 5 tickets
- Predicted via both Gemini and ML pipeline
- Displayed side-by-side predictions

---

## Results

| Model               | Type          | Accuracy           | Pros                   | Cons                        |
| ------------------- | ------------- | ------------------ | ---------------------- | --------------------------- |
| Gemini              | Zero/Few-shot | No training needed | Smart, Reasoned        | Slow, Costly, Token-limited |
| Logistic Regression | Supervised    | \~86%              | Fast, Local, Efficient | Needs labeled data          |

---

## When to Use What?

- Use **LLMs** for few-shot / small data scenarios or interpretability
- Use **ML models** for scalable, offline, efficient inference



## Developed By

**Muhammad Aqeel Zafar**\
*AI Ticket Classification Challenge*

