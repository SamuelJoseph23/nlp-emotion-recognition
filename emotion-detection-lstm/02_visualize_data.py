import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create visualizations folder if it doesn't exist
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)

print("Loading data...")
try:
    df = pd.read_csv('data/combined_emotions.csv')
    print(f"✓ Loaded {len(df)} samples")
except Exception as e:
    print(f"✗ Error: {e}")
    print("Please run 01_load_data.py first!")
    exit(1)

print("\nCreating visualizations...")

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

emotion_counts = df['label'].value_counts()
colors = sns.color_palette('viridis', len(emotion_counts))

# Plot 1: Bar chart
plt.subplot(2, 3, 1)
sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette=colors)
plt.title('Emotion Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Emotion', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Plot 2: Pie chart
plt.subplot(2, 3, 2)
plt.pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
plt.title('Emotion Percentage', fontsize=14, fontweight='bold')

# Plot 3: Text length distribution
plt.subplot(2, 3, 3)
for emotion in df['label'].unique():
    subset = df[df['label'] == emotion]['text_length']
    plt.hist(subset, alpha=0.5, label=str(emotion), bins=30)
plt.title('Text Length by Emotion', fontsize=14, fontweight='bold')
plt.xlabel('Character Length', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(loc='upper right', fontsize=8)

# Plot 4: Word count box plot
plt.subplot(2, 3, 4)
df.boxplot(column='word_count', by='label', ax=plt.gca())
plt.title('Word Count Distribution by Emotion', fontsize=14, fontweight='bold')
plt.suptitle('')
plt.xlabel('Emotion', fontsize=12)
plt.ylabel('Word Count', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Plot 5: Horizontal bar chart (sorted)
plt.subplot(2, 3, 5)
emotion_counts_sorted = emotion_counts.sort_values()
plt.barh(emotion_counts_sorted.index, emotion_counts_sorted.values, color=colors)
plt.title('Emotion Counts (Sorted)', fontsize=14, fontweight='bold')
plt.xlabel('Count', fontsize=12)
plt.ylabel('Emotion', fontsize=12)

# Plot 6: Average text length by emotion
plt.subplot(2, 3, 6)
avg_length = df.groupby('label')['text_length'].mean().sort_values()
plt.barh(avg_length.index, avg_length.values, color=colors)
plt.title('Average Text Length by Emotion', fontsize=14, fontweight='bold')
plt.xlabel('Average Character Length', fontsize=12)
plt.ylabel('Emotion', fontsize=12)

plt.tight_layout()
output_path = 'visualizations/data_overview.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to '{output_path}'")
plt.show()

print("\n" + "="*60)
print("VISUALIZATION COMPLETE!")
print("="*60)
