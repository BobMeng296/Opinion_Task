import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Load the dataset
df = pd.read_csv('./data/data_all.csv', encoding='utf-8', encoding_errors='replace')

# Function for text preprocessing (Sentiment-focused)
def preprocess_text_sentiment(text):
    # Convert to lowercase, but keep certain punctuation
    text = text.lower()
    
    # Remove '????' and similar patterns
    text = re.sub(r'\?{2,}', '', text)

    # Remove special characters except ! and ?
    text = re.sub(r'[^a-zA-Z\s!?]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords, but keep negation words
    stop_words = set(stopwords.words('english')) - {'no', 'not', 'nor', 'neither', 'never', 'hardly', 'scarcely'}
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Function for text preprocessing (NER-focused)
def preprocess_text_ner(text):
    # Keep original case
    # Remove possessive 's
    text = re.sub(r"'s\b", "", text)
    # Remove '????' and similar patterns
    text = re.sub(r'\?{2,}', '', text)
    # Remove special characters except hyphen and period
    text = re.sub(r'[^a-zA-Z0-9\s\-\.]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Apply preprocessing to the 'sentence' column
df['processed_text_sentiment'] = df['sentence'].apply(preprocess_text_sentiment)
df['processed_text_ner'] = df['sentence'].apply(preprocess_text_ner)

# Display the first few rows to verify preprocessing
print(df[['sentence', 'processed_text_sentiment', 'processed_text_ner']].head())

# Create NER labels (assuming 'product' and 'part' columns exist)
def create_ner_labels(row):
    text = row['processed_text_ner']
    tokens = word_tokenize(text)
    labels = ['O'] * len(tokens) # Initialize all as 'O' (Outside)
    
    entities = []
    if pd.notnull(row['product']):
        entities.append(('PRODUCT', preprocess_text_ner(row['product'])))
    if pd.notnull(row['part']):
        entities.append(('PART', preprocess_text_ner(row['part'])))
    
    for entity_type, entity in entities:
        entity_tokens = word_tokenize(entity)
        for i in range(len(tokens) - len(entity_tokens) + 1):
            if tokens[i:i+len(entity_tokens)] == entity_tokens:
                labels[i] = f'B-{entity_type}'
                for j in range(1, len(entity_tokens)):
                    labels[i+j] = f'I-{entity_type}'
    
    return ' '.join(labels)


df['ner_labels'] = df.apply(create_ner_labels, axis=1)

# Sentiment distribution
# Sentiment distribution
plt.figure(figsize=(10, 8))
sentiment_counts = df['class'].value_counts()
plt.pie(
    sentiment_counts.values, 
    labels=sentiment_counts.index, 
    autopct='%1.1f%%', 
    startangle=90,
    textprops={'fontsize': 18}  
)
# plt.title('Sentiment Distribution', fontsize=16) 
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()

# Word frequency analysis (for sentiment)
all_words = ' '.join(df['processed_text_sentiment']).split()
word_freq = Counter(all_words)
common_words = word_freq.most_common(20)

plt.figure(figsize=(12, 6))
sns.barplot(x=[word[0] for word in common_words], y=[word[1] for word in common_words])
plt.title('20 Most Common Words (Sentiment)')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# NER label distribution
ner_label_dist = Counter(' '.join(df['ner_labels']).split())
plt.figure(figsize=(10, 6))
sns.barplot(x=list(ner_label_dist.keys()), y=list(ner_label_dist.values()))
plt.title('NER Label Distribution')
plt.xlabel('NER Labels')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

ner_label_dist = Counter(' '.join(df['ner_labels']).split())
plt.figure(figsize=(12, 8))
plt.pie(
    ner_label_dist.values(), 
    labels=ner_label_dist.keys(), 
    autopct='%1.1f%%', 
    startangle=90,
    textprops={'fontsize': 18}  
)
# plt.title('NER Label Distribution', fontsize=16)  
plt.axis('equal')
plt.show()


# Save the processed data
df.to_csv('./data/data_all_processed.csv', index=False)

print("Enhanced data processing completed. Processed data saved to 'data_all_processed.csv'.")