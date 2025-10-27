import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

print("="*60)
print("IMPROVED TEXT PREPROCESSING")
print("="*60)

df = pd.read_csv('data/combined_emotions.csv')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
emotional_words = {
    'not','no','never','neither','nor','none','nobody','nothing','nowhere',
    'dont','doesn','didn','won','wouldn','shouldn','couldn','cant','cannot',
    'ain','aren','isn','wasn','weren','haven','hasn','hadn','shan',
    'very','really','so','too','happy','glad','joy','joyful','sad','angry','love',
    'hate','scared','amazing','wonderful','great','good','fantastic','excellent',
    'lovely','awesome','perfect','delighted','pleased','cheerful','unhappy',
    'depressed','miserable','terrible','awful','bad','horrible','mad','furious',
    'annoyed','frustrated','upset','disappointed','hurt','fear','worried',
    'anxious','nervous','terrified','surprised','shock','shocked','unexpected','wow','omg'
}
stop_words = stop_words - emotional_words

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"\S+@\S+", '', text)
    text = re.sub(r"<.*?>", '', text)
    text = ' '.join(text.split())
    return text

def preprocess_text(text):
    # Basic cleaning
    text = clean_text(text)
    tokens = word_tokenize(text)
    processed = []
    for token in tokens:
        if len(token) < 2 and token != 'i': continue
        if token in stop_words: continue
        lemma = lemmatizer.lemmatize(token)
        processed.append(lemma)
    return ' '.join(processed)

def handle_negations(text):
    tokens = text.split()
    result = []
    neg_words = {'not', 'no', 'never', 'dont', 'doesn', 'didn', 'won', 'wouldn','shouldn','couldn','cant','cannot','ain','aren','isn','wasn','weren','haven','hasn','hadn','shan'}
    negated = False
    for token in tokens:
        if any(neg in token for neg in neg_words):
            result.append(token)
            negated = True
        elif negated:
            result.append(f'NOT_{token}')
            negated = False
        else:
            result.append(token)
    return ' '.join(result)

df['processed_text'] = df['text'].apply(preprocess_text).apply(handle_negations)
df = df[df['processed_text'].str.strip() != '']
df['word_count_processed'] = df['processed_text'].apply(lambda x: len(str(x).split()))
df = df[df['word_count_processed'] >= 1]

df[['processed_text','label']].rename(columns={'processed_text':'text'}).to_csv('data/final_data.csv',index=False)
print('✓ Saved cleaned and negation-handled data to data/final_data.csv')
