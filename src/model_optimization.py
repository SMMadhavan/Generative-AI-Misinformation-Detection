import sqlite3
import pandas as pd
import numpy as np
import joblib
import warnings
import time

# Sklearn Imports
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore')

# Configuration
DB_PATH = 'neural_db_v2.sqlite'
MODEL_DIR = 'models/'

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

# Optimization pipeline
def run_optimization():
    print("\n" + "="*80)
    print("🧬 HYPERPARAMETER TUNING: LINEAR SVC OPTIMIZATION")
    print("="*80)

    # A. Load Data
    print("   [1/5] Loading Data from SQLite...")
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM training_dataset", conn)
    except Exception as e:
        print(f"Error reading DB: {e}")
        return
    conn.close()

    if df.empty:
        print("   ❌ Error: Database is empty.")
        return

    # Clean Data
    df.dropna(subset=['text', 'label'], inplace=True)
    df['text'] = df['text'].astype(str)
    
    # Downsample for speed during GridSearch (Optimization doesn't need all data)
    if len(df) > 10000:
        print("   [INFO] Downsampling to 10k records for faster GridSearch...")
        df = df.sample(n=10000, random_state=42)

    # B. Feature Extraction
    print("   [2/5] Engineering Features (TF-IDF + Linguistic DNA)...")
    
    # Custom Features
    features_list = [get_advanced_features(t) for t in df['text']]
    X_custom = np.array(features_list)
    scaler = StandardScaler()
    X_custom_scaled = scaler.fit_transform(X_custom)

    # Text Features
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X_text = tfidf.fit_transform(df['text'])
    
    # Select Best Text Features
    selector = SelectKBest(score_func=chi2, k=500)
    X_text_selected = selector.fit_transform(X_text, df['label'])

    # Combine
    X_final = hstack([X_text_selected, X_custom_scaled])
    y = df['label']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    # C. Define The Grid
    # We are tuning 'C' (Strictness) and 'max_iter' (Convergence)
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'loss': ['squared_hinge'],
        'max_iter': [1000, 2000]
    }

    print(f"   [3/5] Starting GridSearch on {len(X_train.toarray())} samples...")
    print(f"         Grid: {param_grid}")

    # D. Run GridSearch
    svc = LinearSVC(random_state=42, dual='auto')
    grid_search = GridSearchCV(svc, param_grid, cv=3, verbose=2, n_jobs=-1, scoring='accuracy')
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start_time

    # E. Results
    print("\n" + "="*80)
    print("🏆 OPTIMIZATION RESULTS")
    print("="*80)
    print(f"   ✅ Best Parameters: {grid_search.best_params_}")
    print(f"   ✅ Best CV Score:   {round(grid_search.best_score_ * 100, 2)}%")
    print(f"   ⏱️  Time Taken:     {round(elapsed, 2)} seconds")

    # Validate on Test Set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"\n   🧪 Test Set Accuracy: {round(test_acc * 100, 2)}%")
    print("\n   [Classification Report]")
    print(classification_report(y_test, y_pred))

    # Note: We do NOT overwrite the model files here to respect app.py logic.
    # This script is purely for evidence/analysis.

if __name__ == "__main__":
    run_optimization()