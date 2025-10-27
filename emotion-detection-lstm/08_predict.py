import numpy as np
import pickle
import re
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

print("="*60)
print("EMOTION DETECTION - PREDICTION TOOL")
print("="*60)

# Load all necessary components
model = load_model('models/best_model.keras')
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('models/config.pkl', 'rb') as f:
    config = pickle.load(f)

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

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"<.*?>", '', text)
    text = ' '.join(text.split())
    tokens = word_tokenize(text)
    processed = []
    for token in tokens:
        if len(token) < 2 and token != 'i': continue
        if token in stop_words: continue
        lemma = lemmatizer.lemmatize(token)
        processed.append(lemma)
    processed = ' '.join(processed)
    # Negation handling
    tokens = processed.split()
    result = []
    neg_words = {'not','no','never','dont','doesn','didn','won','wouldn','shouldn','couldn','cant','cannot','ain','aren','isn','wasn','weren','haven','hasn','hadn','shan'}
    negated = False
    for tok in tokens:
        if any(neg in tok for neg in neg_words):
            result.append(tok)
            negated = True
        elif negated:
            result.append(f'NOT_{tok}')
            negated = False
        else:
            result.append(tok)
    return ' '.join(result)

def predict_emotion(text):
    processed = preprocess_text(text)
    if len(processed.strip()) < 1: return None, None, None
    seq = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, maxlen=config['max_sequence_length'], padding='post')
    probs = model.predict(padded, verbose=0)[0]
    pred_idx = np.argmax(probs)
    pred_emo = label_encoder.classes_[pred_idx]
    confidence = probs[pred_idx]
    sorted_probs = sorted([(label_encoder.classes_[i], float(probs[i])) for i in range(len(probs))], key=lambda x: x[1], reverse=True)
    return pred_emo, confidence, sorted_probs

print("\nType text (or 'quit') to predict:")

while True:
    user_input = input("\nText: ").strip()
    if user_input.lower() in ['quit','exit','q','']: break
    pred, conf, all_probs = predict_emotion(user_input)
    if pred:
        print(f"\nEmotion: {pred.upper()}, Confidence: {conf*100:.1f}%")
        print("Probabilities:")
        for emo, prob in all_probs: print(f"  {emo:10s} : {prob*100:.1f}%")
    else:
        print("Could not process text or too short.")
