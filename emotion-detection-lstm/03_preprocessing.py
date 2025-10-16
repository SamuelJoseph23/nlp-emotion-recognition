import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

print("="*60)
print("TEXT PREPROCESSING PIPELINE")
print("="*60)

# Load data
print("\nLoading dataset...")
try:
    df = pd.read_csv('data/combined_emotions.csv')
    print(f"✓ Loaded {len(df)} samples")
except Exception as e:
    print(f"✗ Error: {e}")
    print("Please run 01_load_data.py first!")
    exit(1)

# ===== STEP 1: TEXT CLEANING =====
print("\n" + "="*60)
print("STEP 1: TEXT CLEANING")
print("="*60)

def clean_text(text):
    """Clean text by removing URLs, mentions, punctuation, numbers"""
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions (@username) and hashtags
    text = re.sub(r'\@\w+|\#\w+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

print("Cleaning text...")
df['cleaned_text'] = df['text'].apply(clean_text)

# Show examples
print("\nCleaning Examples:")
for i in [0, 50, 100]:
    if i < len(df):
        print(f"\n--- Example {i+1} ---")
        print(f"Original : {df['text'].iloc[i][:80]}...")
        print(f"Cleaned  : {df['cleaned_text'].iloc[i][:80]}...")

# ===== STEP 2: TOKENIZATION & LEMMATIZATION =====
print("\n" + "="*60)
print("STEP 2: TOKENIZATION & LEMMATIZATION")
print("="*60)

# Initialize
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Keep emotional words that are typically stopwords
emotional_words = {'not', 'no', 'never', 'very', 'really', 'so', 'too', 'just', 'dont', 'didnt', 'doesnt'}
stop_words = stop_words - emotional_words

print(f"Using {len(stop_words)} stopwords")
print(f"Kept emotional words: {emotional_words}")

def preprocess_text(text):
    """Tokenize, remove stopwords, and lemmatize"""
    try:
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in stop_words and len(token) > 1
        ]
        
        return ' '.join(processed_tokens)
    except:
        return text

print("\nProcessing text...")
df['processed_text'] = df['cleaned_text'].apply(preprocess_text)

# Show examples
print("\nProcessing Examples:")
for i in [0, 50, 100]:
    if i < len(df):
        print(f"\n--- Example {i+1} ---")
        print(f"Cleaned   : {df['cleaned_text'].iloc[i][:70]}...")
        print(f"Processed : {df['processed_text'].iloc[i][:70]}...")

# ===== STEP 3: FILTERING =====
print("\n" + "="*60)
print("STEP 3: FILTERING")
print("="*60)

print(f"Before filtering: {len(df)} samples")

# Remove empty processed texts
df = df[df['processed_text'].str.strip() != '']
print(f"After removing empty texts: {len(df)} samples")

# Remove very short texts (less than 2 words)
df['word_count_processed'] = df['processed_text'].apply(lambda x: len(str(x).split()))
df = df[df['word_count_processed'] >= 2]
print(f"After removing short texts: {len(df)} samples")

# ===== STEP 4: STATISTICS =====
print("\n" + "="*60)
print("PREPROCESSING STATISTICS")
print("="*60)

df['processed_length'] = df['processed_text'].apply(len)

print("\nProcessed text length:")
print(df['processed_length'].describe())

print("\nProcessed word count:")
print(df['word_count_processed'].describe())

print("\nEmotion distribution after preprocessing:")
print(df['label'].value_counts())

# ===== STEP 5: SAVE =====
print("\n" + "="*60)
print("SAVING PREPROCESSED DATA")
print("="*60)

# Save full preprocessed dataset
df.to_csv('data/preprocessed_emotions.csv', index=False)
print("✓ Saved to 'data/preprocessed_emotions.csv'")

# Save only necessary columns for modeling
df_final = df[['processed_text', 'label']].copy()
df_final.columns = ['text', 'label']  # Rename for consistency
df_final.to_csv('data/final_data.csv', index=False)
print("✓ Saved final data to 'data/final_data.csv'")

print("\n" + "="*60)
print("PREPROCESSING COMPLETE!")
print("="*60)
print(f"✓ Final dataset: {len(df_final)} samples")
print(f"✓ Emotions: {sorted(df_final['label'].unique())}")
print(f"✓ Ready for model training!")
print("="*60)
