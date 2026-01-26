import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

warnings.filterwarnings('ignore')

# Configuration
DB_PATH = 'neural_db_v2.sqlite'
OUTPUT_IMG_PATH = 'reports/svm_feature_importance.png'

# Replicating feature engineering
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

def analyze_svm_weights():
    print("\n" + "="*80)
    print("🧠 SVM COEFFICIENT ANALYSIS: WHAT THE MODEL 'THINKS'")
    print("="*80)

    # 1. Load & Sample Data
    conn = sqlite3.connect(DB_PATH)
    try: df = pd.read_sql("SELECT * FROM training_dataset", conn)
    except: return
    conn.close()

    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42)

    df.dropna(subset=['text', 'label'], inplace=True)
    df['text'] = df['text'].astype(str)

    # 2. Vectorize & Prepare
    print("   [INFO] Vectorizing Text and calculating Linguistic DNA...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000) # Limit features for cleaner plot
    X_text = tfidf.fit_transform(df['text'])
    
    features_list = [get_advanced_features(t) for t in df['text']]
    feature_names_dna = ['DNA_Uniformity', 'DNA_Richness', 'DNA_Buzzwords', 'DNA_Burstiness', 'DNA_Complexity', 'DNA_Punc', 'DNA_Length']
    
    X_custom = np.array(features_list)
    scaler = StandardScaler()
    X_custom_scaled = scaler.fit_transform(X_custom)

    X_final = hstack([X_text, X_custom_scaled])
    
    # 3. Train Linear SVC
    print("   [INFO] Training LinearSVC to extract coefficients...")
    svm = LinearSVC(C=10, random_state=42, dual='auto', max_iter=2000)
    svm.fit(X_final, df['label'])

    # 4. Extract Top Coefficients
    # Get all feature names: TFIDF words + DNA names
    all_feature_names = tfidf.get_feature_names_out().tolist() + feature_names_dna
    
    # Get weights (coefficients)
    coefs = svm.coef_.ravel()
    
    # Sort them
    # Positive Coefs = Strong indicators of Class 1 (AI)
    # Negative Coefs = Strong indicators of Class 0 (Human)
    top_positive_indices = np.argsort(coefs)[-15:]
    top_negative_indices = np.argsort(coefs)[:15]
    
    top_indices = np.hstack([top_negative_indices, top_positive_indices])
    top_features = [all_feature_names[i] for i in top_indices]
    top_coefs = coefs[top_indices]

    # 5. Plot
    print("   [INFO] Plotting Feature Importance...")
    plt.figure(figsize=(10, 8))
    colors = ['green' if c < 0 else 'red' for c in top_coefs]
    
    plt.barh(range(len(top_indices)), top_coefs, color=colors)
    plt.yticks(range(len(top_indices)), top_features)
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title("Top Predictors: Human (Green) vs. AI (Red)")
    plt.xlabel("SVM Coefficient Magnitude")
    
    if not os.path.exists('reports'): os.makedirs('reports')
    plt.savefig(OUTPUT_IMG_PATH)
    print(f"   ✅ Saved Coefficient Chart to: {OUTPUT_IMG_PATH}")
    print("="*80 + "\n")

if __name__ == "__main__":
    analyze_svm_weights()