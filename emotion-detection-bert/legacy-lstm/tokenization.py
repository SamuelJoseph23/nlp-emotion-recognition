import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("="*60)
print("TOKENIZATION AND SEQUENCE PREPARATION")
print("="*60)

df = pd.read_csv('data/final_data.csv')

MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100

print(f"Total samples: {len(df)}")

# Tokenizer setup
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])

word_index = tokenizer.word_index
actual_vocab_size = len(word_index)
vocab_size = min(MAX_VOCAB_SIZE, actual_vocab_size) + 1  # +1 for padding index

print(f"Actual vocab size: {actual_vocab_size}, Model embedding vocab size: {vocab_size}")

sequences = tokenizer.texts_to_sequences(df['text'])
padded_sequences = pad_sequences(
    sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post'
)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(df['label'])
from tensorflow.keras.utils import to_categorical
categorical_labels = to_categorical(encoded_labels)

# Train-test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, categorical_labels, test_size=0.2,
    random_state=42, stratify=encoded_labels
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Save all components
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

np.save('models/X_train.npy', X_train)
np.save('models/X_test.npy', X_test)
np.save('models/y_train.npy', y_train)
np.save('models/y_test.npy', y_test)

config = {
    'max_vocab_size': MAX_VOCAB_SIZE,
    'max_sequence_length': MAX_SEQUENCE_LENGTH,
    'vocab_size': vocab_size,
    'num_emotions': len(label_encoder.classes_),
    'emotions': label_encoder.classes_.tolist(),
    'train_samples': len(X_train),
    'test_samples': len(X_test)
}
with open('models/config.pkl', 'wb') as f:
    pickle.dump(config, f)

print("✓ Saved tokenizer, label encoder, train/test splits, and config.")
