import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# Configuration
DB_PATH = 'neural_db_v2.sqlite'
OUTPUT_IMG_PATH = 'reports/feature_distribution.png'

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

def run_eda_analysis():
    print("\n" + "="*80)
    print("📊 EXPLORATORY DATA ANALYSIS (EDA): FEATURE SEPARATION")
    print("="*80)

    # 1. Load Data
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM training_dataset", conn)
    except:
        print("❌ Error: Database not found.")
        return
    conn.close()

    if df.empty: return
    
    # Downsample for clearer plots
    if len(df) > 2000:
        df = df.groupby('label').sample(n=1000, random_state=42)

    # 2. Extract Features
    print("   [INFO] Extracting Linguistic DNA for comparison...")
    feature_names = ['Uniformity', 'Richness', 'Buzz Density', 'Burstiness', 'Complexity', 'Punc Density', 'Avg Sent Len']
    
    features = []
    for txt in df['text']:
        features.append(get_advanced_features(txt))
    
    feat_df = pd.DataFrame(features, columns=feature_names)
    feat_df['Label'] = df['label'].map({0: 'Human', 1: 'AI-Generated'})

    # 3. Visualization (Violin Plots)
    print("   [INFO] Generating Distribution Plots...")
    plt.figure(figsize=(14, 8))
    
    # Plot top 4 most important features
    top_features = ['Burstiness', 'Richness', 'Buzz Density', 'Uniformity']
    
    for i, col in enumerate(top_features):
        plt.subplot(2, 2, i+1)
        sns.violinplot(x='Label', y=col, data=feat_df, palette={'Human': '#28a745', 'AI-Generated': '#dc3545'}, split=True)
        plt.title(f"Distribution: {col}")
        plt.xlabel("")
        plt.ylabel("Score")

    plt.tight_layout()
    
    if not os.path.exists('reports'): os.makedirs('reports')
    plt.savefig(OUTPUT_IMG_PATH)
    print(f"   ✅ Saved EDA Chart to: {OUTPUT_IMG_PATH}")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_eda_analysis()