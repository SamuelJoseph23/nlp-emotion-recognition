# Emotion Detection from Text Using Deep Learning

A deep learning and natural language processing (NLP) solution for classifying emotions from text messages, tweets, or sentences. The project features a complete training and prediction pipeline based on a Bidirectional LSTM model, custom text preprocessing, robust evaluation, and easy extension to web or CLI tools.

---

## Features

- **Dataset Handling:** Loads and combines train.csv, test.csv, and val.csv with auto column detection (supports standard emotion datasets in CSV format).
- **Smart Preprocessing:** Cleans text, performs lemmatization, and applies emotion-aware stopword filtering. Handles negation (e.g., "not happy" ≠ "happy") for realistic emotion modeling.
- **Tokenization & Splitting:** Keras tokenization, sequence padding, categorical label encoding, and stratified train/test splits.
- **Model Architecture:** Bidirectional LSTM (Keras/TensorFlow) with dropout, L2 regularization, batch normalization, and class weighting to reduce overfitting and improve generalization.
- **Training:** Aggressive early stopping, model checkpointing, and adaptive learning rate reduction.
- **Evaluation:** Metrics, classification report, and visualizations including training curves and confusion matrix.
- **Prediction Tool:** Interactive CLI to predict emotions from sample or custom sentences, with probabilistic outputs.
- **Deployment Ready:** Code is modular for easy conversion to a Streamlit or Flask web application.

---

## Folder Structure

```plaintext
emotion-detection-lstm/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── val.csv
│   ├── combined_emotions.csv
│   └── final_data.csv
├── models/
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│   ├── config.pkl
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   ├── emotion_model_untrained.keras
│   ├── emotion_model_final.keras
│   ├── best_model.keras
│   └── training_history.pkl
├── visualizations/
│   ├── emotion_distribution.png
│   ├── text_length.png
│   ├── word_count.png
│   ├── training_history.png
│   └── confusion_matrix.png
├── 01_load_data.py
├── 02_visualize_data.py
├── 03_preprocessing.py
├── 04_tokenization.py
├── 05_build_model.py
├── 06_train_model.py
├── 07_evaluate_model.py
├── 08_predict.py
├── setup_nltk.py
├── requirements.txt
├── README.md
└── .gitignore

```
---

## Quick Start

**1. Setup and Install Dependencies**
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate (Windows)
pip install --upgrade pip
pip install -r requirements.txt
python setup_nltk.py


**2. Prepare Data**
- Download or place train.csv, test.csv, val.csv in `data/`.
- Run:
python 01_load_data.py
python 02_visualize_data.py


**3. Preprocess and Train**
python 03_preprocessing.py
python 04_tokenization.py
python 05_build_model.py
python 06_train_model.py


**4. Evaluate Results**
python 07_evaluate_model.py


**5. Predict Emotions (CLI)**
python 08_predict.py


---

## Model & Approach

- **Preprocessing:** Cleans informal text, removes noise but preserves critical emotional and negation words. Implements custom negation handling (`NOT_` tags around negated words).
- **LSTM Model:** A bidirectional LSTM network is used for its ability to learn long-range dependencies and context within text. Dropout and L2 regularization prevent overfitting.
- **Class Weights:** Handles class imbalance for multi-emotion datasets.
- **Early Stopping:** Avoids overfitting by monitoring validation loss.
- **Performance:** Expected 85–90% accuracy (can vary based on dataset/domain).

---

## Deployment

- The repository is ready to be expanded into a web application using Streamlit or Flask (see comments in `08_predict.py`).
- For online demos, export as a HuggingFace Space or deploy with Streamlit Cloud.

---

## Requirements

- Python 3.7+
- TensorFlow, scikit-learn, NLTK, pandas, matplotlib, seaborn (see `requirements.txt`)

---

## Credits

- Dataset: [Parul Pandey's Emotion Dataset on Kaggle](https://www.kaggle.com/datasets/parulpandey/emotion-dataset)
- Model and code: Developed by [Samuel Joseph S, Jaiden Dennis, Sunith K]
