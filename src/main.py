import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# --- 1. SENSES: Advanced AI Fingerprinting (7 Senses Version) ---
def get_advanced_features(text):
    text = str(text).lower()
    words = text.split()
    
    if len(words) == 0: return [0.0] * 7
    
    sentences = [s for s in text.split('.') if s.strip()]
    sent_lengths = [len(s.split()) for s in sentences]
    if not sent_lengths: sent_lengths = [0]
    
    # Structural DNA
    uniformity = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
    richness = len(set(words)) / len(words)
    
    # Keyword Signal
    buzzwords = ['pivotal', 'delve', 'comprehensive', 'resonate', 'unravel', 'provisionally']
    buzz_density = sum([1 for w in words if w in buzzwords]) / len(words)
    
    burstiness = np.var(sent_lengths) if len(sent_lengths) > 1 else 0
    
    # Linguistic Patterns
    punc_count = sum([1 for char in text if char in '.,!?;:'])
    punc_density = punc_count / len(words)
    
    avg_sent_len = np.mean(sent_lengths)
    
    long_words = sum([1 for w in words if len(w) > 7])
    complexity_ratio = long_words / len(words)

    return [float(uniformity), float(richness), float(buzz_density), 
            float(burstiness), float(complexity_ratio), 
            float(punc_density), float(avg_sent_len)]


def explain_misinformation(model, vectorizer, selector, text):
    # 1. Transform the dynamic user input
    X_raw = vectorizer.transform([text])
    X_selected = selector.transform(X_raw)
    
    # 2. Extract weights from the 'Calibrated' wrapper
    # We look inside the calibration to find the real SVM coefficients
    base_model = model.calibrated_classifiers_[0].base_estimator
    weights = base_model.coef_[0]

    # 3. Map weights to the 500 high-impact features
    feature_names = vectorizer.get_feature_names_out()
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    # 4. Dynamically find which words in the CURRENT text are triggers
    found_triggers = []
    words_in_text = text.lower().split()
    
    for word in words_in_text:
        if word in selected_features:
            idx = selected_features.index(word)
            # If the weight is positive, it contributes to the 'Fake' prediction
            if weights[idx] > 0.05: 
                found_triggers.append(word)
    
    return list(set(found_triggers))

# --- 2. CLEANING PIPELINE ---
class DataCleaner:
    def __init__(self, df):
        self.df_raw = df.copy()
        self.df = df
        
    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()
        return self
        
    def handle_missing_values(self):
        self.df = self.df.dropna()
        return self

    def fix_data_types(self):
        self.df['text'] = self.df['text'].astype(str)
        return self

    def get_report(self):
        print("\n📊 DATA CLEANING COMPARISON TABLE")
        print(f"{'Metric':<20} | {'Before':<15} | {'After':<15}")
        print("-" * 55)
        print(f"{'Total Rows':<20} | {len(self.df_raw):<15} | {len(self.df):<15}")
        print(f"{'Missing Values':<20} | {self.df_raw.isnull().sum().sum():<15} | {0:<15}")
        print(f"{'Duplicate Rows':<20} | {self.df_raw.duplicated().sum():<15} | {0:<15}")
        return self.df

def load_and_merge():
    print("📂 Loading datasets...")
    # Load and label individual datasets
    t_df = pd.read_csv('data/True.csv'); t_df['label'] = 1
    f_df = pd.read_csv('data/Fake.csv'); f_df['label'] = 0
    isot = pd.concat([t_df, f_df])[['text', 'label']]
    
    welfake = pd.read_csv('data/WELFake_Dataset.csv')[['text', 'label']]
    
    genai = pd.read_csv('data/generative_ai_misinformation_dataset.csv')
    genai = genai.rename(columns={'is_misinformation': 'label'})
    genai['label'] = genai['label'].apply(lambda x: 0 if x == 1 else 1)
    genai = genai[['text', 'label']]
    
    return pd.concat([isot, welfake, genai], ignore_index=True).dropna()

def run_eda_dashboard(df_features):
    print("\n🔍 Generating EDA Visuals (Days 3-4 Tasks)...") 
    plt.figure(figsize=(10, 5))
    sns.histplot(df_features['burstiness'], kde=True, color='teal')
    plt.title("Statistical Insight: Burstiness Distribution") 
    plt.savefig('eda_distribution.png') 
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_features.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Interaction Analysis") 
    plt.savefig('eda_correlation.png')
    print("✅ EDA Artifacts saved to folder.") 

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    df_raw = load_and_merge()
    
    # Step 1: Cleaning
    cleaner = DataCleaner(df_raw)
    df = cleaner.remove_duplicates().handle_missing_values().fix_data_types().get_report()
    
    # Step 2: Sampling (30k rows for better pattern detection)
    print("\n⚖️ Balancing the Scales (Expanding to 30,000 samples)...")
    df_sample = df.groupby('label').apply(lambda x: x.sample(n=15000, random_state=42)).reset_index(drop=True)

    # Step 3: DNA Extraction
    print("📏 Extracting Advanced AI Fingerprints...")
    feature_names = ['uniformity', 'richness', 'buzz_density', 'burstiness', 'complexity', 'punc_density', 'avg_sent_len']
    features_list = [get_advanced_features(t) for t in df_sample['text']]
    custom_features = np.array(features_list, dtype=np.float64)
    
    run_eda_dashboard(pd.DataFrame(custom_features, columns=feature_names))
    
    # Step 4: Text Vectorization (N-Grams 2-3 for phrases)
    print("🔢 Vectorizing (Phrase Pattern Extraction)...")
    print("🔢 Vectorizing (Memory-Safe Unigrams)...")
    # Using 1,1 avoids the MemoryError while max_features=5000 keeps accuracy high
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,1))
    X_text = tfidf.fit_transform(df_sample['text'].astype(str))

    # Step 5: Feature Selection & Scaling
    print("🎯 Selecting Top High-Impact Features...")
    selector = SelectKBest(score_func=chi2, k=500)
    X_text_selected = selector.fit_transform(X_text, df_sample['label'])
    
    scaler = StandardScaler()
    custom_features_scaled = scaler.fit_transform(custom_features)
    
    X_final = hstack([X_text_selected, custom_features_scaled])

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, df_sample['label'], test_size=0.2, random_state=42, stratify=df_sample['label']
    )

    # Step 6: Model Training (LinearSVC for accuracy rescue)
    print("🌲 Training Linear Support Vector Machine (Optimized)...")
    # C=10 makes the model much more aggressive at finding the 70-80% accuracy boundary
    base_svm = LinearSVC(C=10, random_state=42, max_iter=3000)
    final_model = CalibratedClassifierCV(base_svm, cv=3)
    final_model.fit(X_train, y_train)

    # Step 7: Evaluation
    y_pred = final_model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"\n🎯 FINAL TUNED ACCURACY: {round(score*100, 2)}%")

    print("📊 Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake/AI', 'Real'])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('confusion_matrix.png')

    # Step 8: Final Saving
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(final_model, 'models/misinfo_detector_model.joblib', compress=3)
    joblib.dump(tfidf, 'models/tfidf_vectorizer.joblib', compress=3)
    joblib.dump(scaler, 'models/dna_scaler.joblib', compress=3)
    joblib.dump(selector, 'models/feature_selector.joblib', compress=3)
    print("🎉 Upgraded model and components saved and ready!")