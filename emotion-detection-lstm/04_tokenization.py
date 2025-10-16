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

# Load preprocessed data
print("\nLoading preprocessed data...")
try:
    df = pd.read_csv('data/final_data.csv')
    print(f"✓ Loaded {len(df)} samples")
except Exception as e:
    print(f"✗ Error: {e}")
    print("Please run 03_preprocessing.py first!")
    exit(1)

print(f"Emotions: {sorted(df['label'].unique())}")

# ===== PARAMETERS =====
MAX_VOCAB_SIZE = 10000  # Maximum number of words to keep
MAX_SEQUENCE_LENGTH = 100  # Maximum length of each sequence

print("\n" + "="*60)
print("TOKENIZATION PARAMETERS")
print("="*60)
print(f"Max vocabulary size: {MAX_VOCAB_SIZE}")
print(f"Max sequence length: {MAX_SEQUENCE_LENGTH}")

# ===== TOKENIZATION =====
print("\n" + "="*60)
print("STEP 1: TOKENIZING TEXT")
print("="*60)

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])

# Get vocabulary info
word_index = tokenizer.word_index
vocab_size = len(word_index)

print(f"Total unique words: {vocab_size}")
print(f"Using vocabulary size: {min(MAX_VOCAB_SIZE, vocab_size)}")

# Show most common words
print("\nTop 20 most common words:")
sorted_words = sorted(word_index.items(), key=lambda x: x[1])[:20]
for word, idx in sorted_words:
    print(f"  {idx:3d}: {word}")

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(df['text'])

print(f"\nConverted {len(sequences)} texts to sequences")
print("\nExample conversion:")
print(f"Text     : {df['text'].iloc[0]}")
print(f"Sequence : {sequences[0][:20]}...")  # Show first 20 numbers

# ===== PADDING =====
print("\n" + "="*60)
print("STEP 2: PADDING SEQUENCES")
print("="*60)

# Analyze sequence lengths
seq_lengths = [len(seq) for seq in sequences]
print(f"\nSequence length statistics:")
print(f"  Min    : {min(seq_lengths)}")
print(f"  Max    : {max(seq_lengths)}")
print(f"  Mean   : {np.mean(seq_lengths):.2f}")
print(f"  Median : {np.median(seq_lengths):.2f}")

# Pad sequences
padded_sequences = pad_sequences(
    sequences,
    maxlen=MAX_SEQUENCE_LENGTH,
    padding='post',
    truncating='post'
)

print(f"\nPadded sequences shape: {padded_sequences.shape}")
print(f"Example padded sequence:")
print(padded_sequences[0])

# ===== ENCODE LABELS =====
print("\n" + "="*60)
print("STEP 3: ENCODING LABELS")
print("="*60)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(df['label'])

print(f"Number of emotion classes: {len(label_encoder.classes_)}")
print(f"Emotion classes: {label_encoder.classes_}")
print(f"\nLabel encoding:")
for i, emotion in enumerate(label_encoder.classes_):
    print(f"  {emotion:12s} → {i}")

# Convert to categorical (one-hot encoding)
from tensorflow.keras.utils import to_categorical
categorical_labels = to_categorical(encoded_labels)

print(f"\nCategorical labels shape: {categorical_labels.shape}")
print(f"Example encoding for '{df['label'].iloc[0]}':")
print(f"  Integer : {encoded_labels[0]}")
print(f"  One-hot : {categorical_labels[0]}")

# ===== TRAIN-TEST SPLIT =====
print("\n" + "="*60)
print("STEP 4: TRAIN-TEST SPLIT")
print("="*60)

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences,
    categorical_labels,
    test_size=0.2,
    random_state=42,
    stratify=encoded_labels  # Maintain emotion distribution
)

print(f"Training samples  : {len(X_train)}")
print(f"Testing samples   : {len(X_test)}")
print(f"Input shape       : {X_train.shape}")
print(f"Output shape      : {y_train.shape}")

# Check emotion distribution in splits
train_labels = np.argmax(y_train, axis=1)
test_labels = np.argmax(y_test, axis=1)

print("\nEmotion distribution in train set:")
for i, emotion in enumerate(label_encoder.classes_):
    count = np.sum(train_labels == i)
    percentage = (count / len(train_labels)) * 100
    print(f"  {emotion:12s}: {count:4d} ({percentage:5.2f}%)")

print("\nEmotion distribution in test set:")
for i, emotion in enumerate(label_encoder.classes_):
    count = np.sum(test_labels == i)
    percentage = (count / len(test_labels)) * 100
    print(f"  {emotion:12s}: {count:4d} ({percentage:5.2f}%)")

# ===== SAVE EVERYTHING =====
print("\n" + "="*60)
print("STEP 5: SAVING PROCESSED DATA")
print("="*60)

# Save tokenizer
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("✓ Saved tokenizer to 'models/tokenizer.pkl'")

# Save label encoder
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("✓ Saved label encoder to 'models/label_encoder.pkl'")

# Save train-test data
np.save('models/X_train.npy', X_train)
np.save('models/X_test.npy', X_test)
np.save('models/y_train.npy', y_train)
np.save('models/y_test.npy', y_test)
print("✓ Saved train-test data to 'models/' folder")

# Save configuration
config = {
    'max_vocab_size': MAX_VOCAB_SIZE,
    'max_sequence_length': MAX_SEQUENCE_LENGTH,
    'vocab_size': min(MAX_VOCAB_SIZE, vocab_size),
    'num_emotions': len(label_encoder.classes_),
    'emotions': label_encoder.classes_.tolist(),
    'train_samples': len(X_train),
    'test_samples': len(X_test)
}

with open('models/config.pkl', 'wb') as f:
    pickle.dump(config, f)
print("✓ Saved configuration to 'models/config.pkl'")

print("\n" + "="*60)
print("TOKENIZATION COMPLETE!")
print("="*60)
print(f"✓ Vocabulary size: {config['vocab_size']}")
print(f"✓ Sequence length: {MAX_SEQUENCE_LENGTH}")
print(f"✓ Number of emotions: {config['num_emotions']}")
print(f"✓ Training samples: {config['train_samples']}")
print(f"✓ Testing samples: {config['test_samples']}")
print("\n✓ Ready for model building!")
print("="*60)
