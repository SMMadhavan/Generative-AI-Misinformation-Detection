import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.sparse import hstack

# --- 1. SENSES: Advanced AI Fingerprinting ---
def get_advanced_features(text):
    text = str(text).lower()
    words = text.split()
    if len(words) == 0: return [0, 0, 0, 0, 0]
    
    sentences = [s for s in text.split('.') if s.strip()]
    sent_lengths = [len(s.split()) for s in sentences]
    
    uniformity = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
    richness = len(set(words)) / len(words)
    
    buzzwords = ['pivotal', 'delve', 'comprehensive', 'resonate', 'unravel', 'provisionally']
    buzz_density = sum([1 for w in buzzwords if w in text]) / len(words)
    
    burstiness = np.var(sent_lengths) if len(sent_lengths) > 1 else 0
    
    # NEW SENSE: Complexity Ratio (Long words vs. Short words)
    long_words = sum([1 for w in words if len(w) > 7])
    complexity_ratio = long_words / len(words)
        
    return [uniformity, richness, buzz_density, burstiness, complexity_ratio]

# --- NEW: Cleaning Pipeline (Meets Evaluation Criteria) ---
class DataCleaner:
    def __init__(self, df):
        self.df_raw = df.copy()
        self.df = df
        
    def remove_duplicates(self): #
        self.df = self.df.drop_duplicates()
        return self
        
    def handle_missing_values(self): #
        self.df = self.df.dropna()
        return self

    def fix_data_types(self): #
        self.df['text'] = self.df['text'].astype(str)
        return self

    def get_report(self): #
        print("\n📊 DATA CLEANING COMPARISON TABLE")
        print(f"{'Metric':<20} | {'Before':<15} | {'After':<15}")
        print("-" * 55)
        print(f"{'Total Rows':<20} | {len(self.df_raw):<15} | {len(self.df):<15}")
        print(f"{'Missing Values':<20} | {self.df_raw.isnull().sum().sum():<15} | {0:<15}")
        print(f"{'Duplicate Rows':<20} | {self.df_raw.duplicated().sum():<15} | {0:<15}")
        return self.df
    

# --- 2. THE ENGINE: Data Loading ---
def load_and_merge():
    print("📂 Loading datasets...")
    true_df = pd.read_csv('data/True.csv')
    fake_df = pd.read_csv('data/Fake.csv')
    true_df['label'] = 1
    fake_df['label'] = 0
    isot = pd.concat([true_df, fake_df])[['text', 'label']]
    
    welfake = pd.read_csv('data/WELFake_Dataset.csv')
    welfake = welfake.rename(columns={'label': 'label'})[['text', 'label']]
    
    genai = pd.read_csv('data/generative_ai_misinformation_dataset.csv')
    genai = genai.rename(columns={'is_misinformation': 'label'})
    genai['label'] = genai['label'].apply(lambda x: 0 if x == 1 else 1) 
    genai = genai[['text', 'label']]
    
    return pd.concat([isot, welfake, genai], ignore_index=True).dropna()

# --- NEW: EDA & Visualization Dashboard --- [cite: 81]
def run_eda_dashboard(df_features):
    print("\n🔍 Generating EDA Visuals (Days 3-4 Tasks)...") 
    
    # Chart 1: Distribution with KDE [cite: 84]
    plt.figure(figsize=(10, 5))
    sns.histplot(df_features['burstiness'], kde=True, color='teal')
    plt.title("Statistical Insight: Burstiness Distribution") 
    plt.savefig('eda_distribution.png') # Save Artifact [cite: 113, 114]
    
    # Chart 2: Correlation Heatmap [cite: 69, 87]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_features.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Interaction Analysis") 
    plt.savefig('eda_correlation.png')
    
    print("✅ EDA Artifacts saved to folder.") 


if __name__ == "__main__":
    df_raw = load_and_merge()
    
    # 1. RUN CLEANING PIPELINE [cite: 44]
    cleaner = DataCleaner(df_raw)
    df = cleaner.remove_duplicates().handle_missing_values().fix_data_types().get_report()
    
    print("⚖️ Balancing the Scales (Strict Stratified Sampling)...")
    df_sample = df.groupby('label').apply(lambda x: x.sample(n=5000, random_state=42)).reset_index(drop=True)

    print("📏 Extracting Advanced AI Fingerprints...")
    feature_names = ['uniformity', 'richness', 'buzz_density', 'burstiness', 'complexity']
    custom_features = np.array([get_advanced_features(t) for t in df_sample['text']])
    
    # 2. RUN EDA DASHBOARD [cite: 81]
    import seaborn as sns # Ensure seaborn is imported at the top
    run_eda_dashboard(pd.DataFrame(custom_features, columns=feature_names))
    
    # ... Continue with your Vectorization and Training ...

    print("🔢 Vectorizing (Increasing Vocabulary Memory)...")
    # Increasing to 8,000 features to capture more 'AI-specific' word pairs
    tfidf = TfidfVectorizer(stop_words='english', max_features=8000, ngram_range=(1,2))
    X_text = tfidf.fit_transform(df_sample['text'].astype(str))
    X_final = hstack([X_text, custom_features])

    # Stratify=y ensures the 80/20 split keeps the 50/50 balance of labels
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, df_sample['label'], test_size=0.2, random_state=42, stratify=df_sample['label']
    )

    print("🌲 Training Elite Random Forest (The Final 60% Push)...")
    # n_estimators=250: More trees to 'average out' the noise
    # min_samples_leaf=2: Prevents trees from memorizing single 'noisy' articles
    rf_model = RandomForestClassifier(
        n_estimators=250, 
        max_depth=40,        # Giving it even more room to think
        min_samples_leaf=2,   # Stabilization tweak
        random_state=42, 
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)

    # Final Result
    y_pred = rf_model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"🎯 FINAL TUNED ACCURACY: {round(score*100, 2)}%")

    # --- 3. DIAGNOSTICS: The Confusion Matrix ---
    print("📊 Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake/AI', 'Real'])
    disp.plot(cmap=plt.cm.Blues)
    plt.show() # This is your "X-ray" to see which news is tricking the AI

    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(rf_model, 'models/misinfo_detector_model.joblib', compress=3)
    joblib.dump(tfidf, 'models/tfidf_vectorizer.joblib', compress=3)
    print("🎉 Upgraded model saved and ready!")