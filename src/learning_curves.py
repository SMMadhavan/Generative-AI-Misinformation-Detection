import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
import os

# Sklearn Imports
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

warnings.filterwarnings('ignore')

# Configuration
DB_PATH = 'neural_db_v2.sqlite'
OUTPUT_IMG_PATH = 'reports/learning_curve_svc.png'

# Feature engineering
def get_advanced_features(text):
    text = str(text).lower()
    words = text.split()
    if len(words) == 0: return [0.0] * 7

    sentences = [s for s in text.split('.') if s.strip()]
    sent_lengths = [len(s.split()) for s in sentences]
    if not sent_lengths: sent_lengths = [0]
    
    uniformity = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
    richness = len(set(words)) / len(words)
    
    buzzwords = ['pivotal', 'delve', 'comprehensive', 'resonate', 'unravel', 'provisionally', 'tapestry', 'synergistic', 'paradigm', 'underscores', 'multifaceted', 'nuance', 'robust', 'landscape']
    buzz_density = sum([1 for w in words if w in buzzwords]) / len(words)
    
    burstiness = np.var(sent_lengths) if len(sent_lengths) > 1 else 0
    punc_density = sum([1 for char in text if char in '.,!?;:']) / len(words)
    avg_sent_len = np.mean(sent_lengths)
    complexity_ratio = sum([1 for w in words if len(w) > 7]) / len(words)

    return [float(uniformity), float(richness), float(buzz_density), float(burstiness), float(complexity_ratio), float(punc_density), float(avg_sent_len)]

def generate_learning_curve():
    print("\n" + "="*80)
    print("📈 GENERATING LEARNING CURVES (LINEAR SVC)")
    print("="*80)

    # A. Load Data
    print("   [1/4] Loading Data...")
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM training_dataset", conn)
    except Exception as e:
        print(f"Error: {e}")
        return
    conn.close()

    if df.empty:
        print("   ❌ Error: No data found.")
        return

    # Clean
    df.dropna(subset=['text', 'label'], inplace=True)
    df['text'] = df['text'].astype(str)

    # Downsample if too huge (Learning curves take time)
    if len(df) > 5000:
        print("   [INFO] Downsampling to 5000 samples for curve generation...")
        df = df.sample(n=5000, random_state=42)

    # B. Features
    print("   [2/4] Preparing Features...")
    features_list = [get_advanced_features(t) for t in df['text']]
    X_custom = np.array(features_list)
    scaler = StandardScaler()
    X_custom_scaled = scaler.fit_transform(X_custom)

    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X_text = tfidf.fit_transform(df['text'])
    selector = SelectKBest(score_func=chi2, k=500)
    X_text_selected = selector.fit_transform(X_text, df['label'])

    X_final = hstack([X_text_selected, X_custom_scaled])
    y = df['label']

    # C. Calculate Curve
    print("   [3/4] Calculating Training vs Validation Scores...")
    # We use the LinearSVC configuration found in app.py (C=10)
    model = LinearSVC(C=10, max_iter=2000, random_state=42, dual='auto')

    train_sizes, train_scores, test_scores = learning_curve(
        model, X_final, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="accuracy"
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # D. Plotting
    print("   [4/4] Rendering Plot...")
    plt.figure(figsize=(10, 6))
    plt.title("Learning Curve: Linear SVC (Forensic Engine)")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy Score")
    plt.grid(True)

    # Plot Training
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")

    # Plot Validation
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-Validation Score")

    plt.legend(loc="best")
    
    # Ensure directory exists
    if not os.path.exists('reports'):
        os.makedirs('reports')
        
    plt.savefig(OUTPUT_IMG_PATH)
    print(f"\n   ✅ Success! Learning curve saved to: {OUTPUT_IMG_PATH}")
    print("="*80 + "\n")

if __name__ == "__main__":
    generate_learning_curve()