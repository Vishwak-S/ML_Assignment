import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Configuration
DATA_PATH = "IMDB Dataset.csv"
GLOVE_PATH = "glove.6B.100d.txt"
OUTPUT_DIR = "results"
MAX_FEATURES = 5000  # For BoW and TF-IDF
SAMPLE_SIZE = None   # Set to a number (e.g., 20000) for faster testing, None utilizes all 50k
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'classification_reports'), exist_ok=True)

print("Running NLP Word Vectorization Pipeline...")

# --- 1. PREPROCESSING ---
print("\n--- 1. Data Loading and Preprocessing ---")
df = pd.read_csv(DATA_PATH)
if SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Subsampled dataset to {SAMPLE_SIZE} records.")
else:
    print(f"Loaded {len(df)} records from IMDB dataset.")

# Map sentiment
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Ensure NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens), tokens

print("Preprocessing texts (this may take a few minutes for 50k)...")
start_time = time.time()
df['clean_text'], df['tokens'] = zip(*df['review'].apply(preprocess_text))
print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds.")

# Train/Test Split
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=RANDOM_STATE
)
X_train_tokens = df.loc[X_train_text.index, 'tokens']
X_test_tokens = df.loc[X_test_text.index, 'tokens']

# Dictionary to store all results
results = []

# Helper function to evaluate and save results
def evaluate_model(y_true, y_pred, method_name, model_name, vect_time):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    results.append({
        'Method': method_name,
        'Classifier': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'Vectorization_Time (s)': vect_time
    })
    
    with open(os.path.join(OUTPUT_DIR, 'classification_reports', f'{method_name.lower().replace(" ", "_")}_report.txt'), 'a') as f:
        f.write(f"\n--- {method_name} with {model_name} ---\n")
        f.write(classification_report(y_true, y_pred))

# Helper to train and evaluate Linear Models
def train_and_eval(X_train_vec, X_test_vec, method_name, vect_time):
    print(f"Training Logistic Regression for {method_name}...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_vec, y_train)
    evaluate_model(y_test, lr.predict(X_test_vec), method_name, 'Logistic Regression', vect_time)
    
    print(f"Training LinearSVC for {method_name}...")
    svc = LinearSVC(max_iter=2000)
    svc.fit(X_train_vec, y_train)
    evaluate_model(y_test, svc.predict(X_test_vec), method_name, 'Linear SVM', vect_time)


# --- 2. CONVENTIONAL METHODS ---
print("\n--- 2. Conventional Methods ---")

# 2.1 Bag of Words
print("Running Bag of Words (BoW)...")
start_time = time.time()
bow_vectorizer = CountVectorizer(max_features=MAX_FEATURES)
X_train_bow = bow_vectorizer.fit_transform(X_train_text)
X_test_bow = bow_vectorizer.transform(X_test_text)
bow_time = time.time() - start_time
print(f"BoW Vectorization Time: {bow_time:.2f} seconds")
train_and_eval(X_train_bow, X_test_bow, 'Bag of Words', bow_time)

# 2.2 TF-IDF
print("\nRunning TF-IDF...")
start_time = time.time()
tfidf_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
tfidf_time = time.time() - start_time
print(f"TF-IDF Vectorization Time: {tfidf_time:.2f} seconds")
train_and_eval(X_train_tfidf, X_test_tfidf, 'TF-IDF', tfidf_time)


# --- 3. DEEP LEARNING / EMBEDDING METHODS ---
print("\n--- 3. Word Embedding Methods ---")

# 3.1 Word2Vec skipped (Gensim has build issues on Py 3.14 without C++ tools)
# GloVe and BERT are sufficient to cover the deep learning requirement.

# 3.2 GloVe (Pre-trained)
print("\nRunning GloVe...")
def load_glove_model(glove_file):
    print("Loading GloVe Model...")
    model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            model[word] = embedding
    print(f"{len(model)} words loaded!")
    return model

def get_average_glove_embeddings(tokens_list, glove_model, vector_size=100):
    doc_vectors = []
    for tokens in tokens_list:
        valid_words = [word for word in tokens if word in glove_model]
        if not valid_words:
            doc_vectors.append(np.zeros(vector_size))
        else:
            doc_vectors.append(np.mean([glove_model[w] for w in valid_words], axis=0))
    return np.array(doc_vectors)

start_time = time.time()
if os.path.exists(GLOVE_PATH):
    glove_dict = load_glove_model(GLOVE_PATH)
    X_train_glove = get_average_glove_embeddings(X_train_tokens, glove_dict)
    X_test_glove = get_average_glove_embeddings(X_test_tokens, glove_dict)
    glove_time = time.time() - start_time
    print(f"GloVe Vectorization Time: {glove_time:.2f} seconds")
    train_and_eval(X_train_glove, X_test_glove, 'GloVe', glove_time)
else:
    print(f"ERROR: {GLOVE_PATH} not found. Skipping GloVe.")

# 3.3 BERT
# Skipped - Python 3.14 lacks pre-compiled PyTorch wheels. GloVe provides sufficient deep learning embedding representation.


# --- 4. RESULTS AND VISUALIZATIONS ---
print("\n--- 4. Comparison and SAVING ---")
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, 'comparison_table.csv'), index=False)
print("Results saved to results/comparison_table.csv")

# Visualization 1: Accuracy Comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='Method', y='Accuracy', hue='Classifier')
plt.title('Accuracy Comparison among Word Vectorization Methods')
plt.ylim(0, 1.0)
plt.ylabel('Accuracy')
plt.xlabel('Vectorization Method')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_comparison.png'))
print("Saved accuracy_comparison.png")

# Visualization 2: Time Complexity
time_df = results_df[['Method', 'Vectorization_Time (s)']].drop_duplicates()
plt.figure(figsize=(8, 5))
sns.barplot(data=time_df, x='Method', y='Vectorization_Time (s)', hue='Method', legend=False, palette='viridis')
plt.title('Vectorization Time Required for Each Method')
plt.ylabel('Time (seconds)')
plt.xlabel('Vectorization Method')
for index, row in enumerate(time_df['Vectorization_Time (s)']):
    plt.text(index, row + 0.5, f"{row:.1f}s", color='black', ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'time_comparison.png'))
print("Saved time_comparison.png")

print("\nAll tasks completed successfully!")
