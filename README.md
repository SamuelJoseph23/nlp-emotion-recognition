# BERT-based Emotion Recognition Pipeline

## Overview

This project provides a fully reproducible pipeline for text-based emotion detection using BERT and modern NLP preprocessing. Includes:
- Dataset merging
- Advanced preprocessing (including negation handling)
- Automatic negation-based data augmentation
- Model training and interactive/manual inference (CLI or web)

---

## Features

- **Unified data preparation script:** Combines, preprocesses, and augments emotion datasets.
- **Flexible CSV input:** Accepts and merges standard emotion datasets (e.g. `train.csv`, `test.csv`, `val.csv`).
- **Preprocessing:** Lemmatization, proper stopword handling, explicit negation marking.
- **Augmentation:** Adds robust negated phrase samples for each emotion class to boost performance on tricky text.
- **BERT workflow:** Scripted, interactive, and web app interfaces for batch or real-time emotion detection.

---

## Repository Structure

.
├── data/
│ ├── train.csv
│ ├── test.csv
│ ├── val.csv
│ ├── final_data.csv # Optional: raw/merged input
│ └── final_data_aug.csv # Output: run after prepare_data.py
├── prepare_data.py # Data pipeline: load, preprocess, augment
├── bert_emotion.py # Model training & CLI inference
├── requirements.txt
├── README.md
└── bert_emotion_model/
└── ... (saved model after training)


---

## Setup

### 1. Install requirements
pip install -r requirements.txt


### 2. Download NLTK Data (first time only)

import nltk

nltk.download('punkt')

nltk.download('stopwords')

nltk.download('wordnet')

nltk.download('omw-1.4')


---

## Data Preparation

Run the unified script to preprocess and augment data:
python prepare_data.py

- Output: `data/final_data_aug.csv` for model training.

---

## Model Training

Train the BERT emotion classifier interactively:
python bert_emotion.py

- Choose `1` for model training.
- The trained model and tokenizer are saved in `bert_emotion_model/`.

---

## Inference

### **Command-line emotion detection:**
python bert_emotion.py

- Choose `2` for interactive detection.
- Type sentences and press Enter for results.

### **(Optional) Web App:**
- If you have `app.py` (Streamlit), run:

streamlit run app.py


---

## Requirements

pandas
nltk
torch
transformers
scikit-learn
accelerate

Add `streamlit` if you demo as a web app.

---

## Tips

- Add or tune emotion labels in `prepare_data.py` as needed.
- Full GPU support if PyTorch w/ CUDA is installed.
- For other languages/datasets, adapt preprocessing and label mapping.

---

## License

MIT License (or specify your license)

---

_Questions? PRs and issues welcome!_
