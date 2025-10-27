import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

df = pd.read_csv('data/combined_emotions.csv')
df['text_length'] = df['text'].astype(str).apply(len)
df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))

plt.figure(figsize=(12,6))
sns.countplot(x='label',data=df,order=df['label'].value_counts().index)
plt.title("Emotion Class Distribution")
plt.savefig('visualizations/emotion_distribution.png')
plt.close()

plt.figure(figsize=(12,6))
sns.boxplot(x='label',y='text_length',data=df,order=df['label'].value_counts().index)
plt.title("Text Length per Emotion")
plt.savefig('visualizations/text_length.png')
plt.close()

plt.figure(figsize=(12,6))
sns.boxplot(x='label',y='word_count',data=df,order=df['label'].value_counts().index)
plt.title("Word Count per Emotion")
plt.savefig('visualizations/word_count.png')
plt.close()

print("Saved visualizations in visualizations/")
