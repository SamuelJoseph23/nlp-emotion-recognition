import pandas as pd
import numpy as np
import os

pd.set_option('display.max_colwidth', 100)

print("="*60)
print("LOADING DATASET (CSV FORMAT)")
print("="*60)

# Find CSV file in data folder
csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]

if not csv_files:
    print("\n✗ No CSV file found in data folder!")
    print("Please download the dataset from Kaggle and place it in the data folder")
    exit(1)

csv_file = csv_files[0]
print(f"\nFound CSV file: {csv_file}")

# Load CSV dataset
try:
    df = pd.read_csv(f'data/{csv_file}')
    print(f"✓ Successfully loaded {csv_file}")
except Exception as e:
    print(f"✗ Error loading CSV: {e}")
    exit(1)

print(f"Total samples: {len(df)}")
print(f"\nOriginal columns: {df.columns.tolist()}")

# Display first few rows to understand structure
print("\n--- First 5 rows ---")
print(df.head())

# Standardize column names
column_mapping = {}

# Find text column
text_columns = ['text', 'content', 'tweet', 'message', 'sentence']
for col in df.columns:
    if col.lower() in text_columns:
        column_mapping[col] = 'text'
        break

# Find label column
label_columns = ['label', 'emotion', 'sentiment', 'emotions', 'feeling']
for col in df.columns:
    if col.lower() in label_columns:
        column_mapping[col] = 'label'
        break

# Rename columns
if column_mapping:
    df = df.rename(columns=column_mapping)
    print(f"\n✓ Renamed columns: {column_mapping}")

# Verify we have required columns
if 'text' not in df.columns or 'label' not in df.columns:
    print(f"\n✗ Error: Could not find text and label columns")
    print(f"Available columns: {df.columns.tolist()}")
    print("\nPlease manually check your CSV structure")
    exit(1)

# Keep only text and label columns
df = df[['text', 'label']].copy()

# Convert numeric labels to emotion names
label_mapping = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

# Check if labels are numeric
if df['label'].dtype in ['int64', 'int32', 'float64']:
    print("\n✓ Detected numeric labels. Converting to emotion names...")
    df['label'] = df['label'].map(label_mapping)
    print(f"Label mapping: {label_mapping}")

print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing)

# Remove missing values
original_len = len(df)
df = df.dropna()
print(f"\nRemoved {original_len - len(df)} rows with missing values")
print(f"Remaining samples: {len(df)}")

# Remove duplicates
original_len = len(df)
df = df.drop_duplicates()
print(f"Removed {original_len - len(df)} duplicate rows")
print(f"Final samples: {len(df)}")

print("\n" + "="*60)
print("EMOTION DISTRIBUTION")
print("="*60)
emotion_counts = df['label'].value_counts()
print(emotion_counts)

print("\nPercentage distribution:")
for emotion, count in emotion_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{str(emotion):12s}: {count:6d} samples ({percentage:5.2f}%)")

# Text length analysis
df['text_length'] = df['text'].astype(str).apply(len)
df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))

print("\n" + "="*60)
print("TEXT STATISTICS")
print("="*60)
print("\nCharacter length statistics:")
print(df['text_length'].describe())

print("\nWord count statistics:")
print(df['word_count'].describe())

print("\n" + "="*60)
print("SAMPLE TEXTS FOR EACH EMOTION")
print("="*60)
for emotion in sorted(df['label'].unique()):
    print(f"\n--- {str(emotion).upper()} ---")
    samples = df[df['label'] == emotion]['text'].head(3)
    for i, text in enumerate(samples, 1):
        print(f"{i}. {text}")

# Save combined dataset
output_file = 'data/combined_emotions.csv'
df.to_csv(output_file, index=False)

print("\n" + "="*60)
print("SUCCESS!")
print("="*60)
print(f"✓ Saved to '{output_file}'")
print(f"✓ Total emotions: {df['label'].nunique()}")
print(f"✓ Emotion categories: {sorted(df['label'].unique())}")
print(f"✓ Total samples: {len(df)}")
print("="*60)
