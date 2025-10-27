import pandas as pd
import os

pd.set_option('display.max_colwidth', 100)

print("="*60)
print("LOADING DATASET (CSV FORMAT)")
print("="*60)

all_files = os.listdir('data')
csv_files = [f for f in all_files if f.endswith('.csv')]

# Expected files
required_csv = ['train.csv', 'test.csv', 'val.csv']
dfs = []

print("\nFound files:", csv_files)

for filename in required_csv:
    if filename in csv_files:
        df = pd.read_csv(f"data/{filename}")
        print(f"  {filename}: {len(df)}")
        dfs.append(df)

if not dfs:
    print("No expected CSVs found. Please place train.csv, test.csv, val.csv in the data/ folder.")
    exit(1)

df = pd.concat(dfs, ignore_index=True)
print(f"\n✓ Combined total: {len(df)} samples")

# Try to standardize columns
column_mapping = {}
for col in df.columns:
    if col.lower() in ['text', 'content', 'sentence', 'tweet', 'message']:
        column_mapping[col] = 'text'
    if col.lower() in ['label', 'emotion', 'sentiment', 'class', 'category']:
        column_mapping[col] = 'label'

df = df.rename(columns=column_mapping)
df = df[['text', 'label']]

# If labels are numbers, map to emotion names (0-5 --> sadness, joy, love, anger, fear, surprise)
if df['label'].dtype in ['int64', 'int32', 'float64']:
    label_mapping = {0:'sadness',1:'joy',2:'love',3:'anger',4:'fear',5:'surprise'}
    df['label'] = df['label'].map(label_mapping)

# Clean
df = df.dropna()
df = df.drop_duplicates()

print("\nLabel distribution:")
print(df['label'].value_counts())

df.to_csv('data/combined_emotions.csv', index=False)
print("\n✓ Saved to data/combined_emotions.csv")
