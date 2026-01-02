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
    if len(words) == 0: return [0, 0, 0]
    
    sentences = text.split('.')
    sent_lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]
    uniformity = np.std(sent_lengths) if len(sent_lengths) > 1 else 0 # AI writes very consistently
    
    richness = len(set(words)) / len(words) # Humans use more diverse words
    
    buzzwords = ['pivotal', 'delve', 'comprehensive', 'resonate', 'paving', 'unravel']
    buzz_density = sum([1 for w in buzzwords if w in text]) / len(words)
    
    return [uniformity, richness, buzz_density]

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

if __name__ == "__main__":
    df = load_and_merge() # This defines 'df' so you don't get a NameError anymore!
    
    print("⚖️ Balancing the Scales (Strict Stratified Sampling)...")
    # We take exactly 5,000 from each group to ensure 0 bias
    df_sample = df.groupby('label').apply(lambda x: x.sample(n=5000, random_state=42)).reset_index(drop=True)

    print("📏 Extracting Advanced AI Fingerprints...")
    custom_features = np.array([get_advanced_features(t) for t in df_sample['text']])

    print("🔢 Vectorizing (Increasing Vocabulary Memory)...")
    # Increasing to 8,000 features to capture more 'AI-specific' word pairs
    tfidf = TfidfVectorizer(stop_words='english', max_features=8000, ngram_range=(1,2))
    X_text = tfidf.fit_transform(df_sample['text'].astype(str))
    X_final = hstack([X_text, custom_features])

    # Stratify=y ensures the 80/20 split keeps the 50/50 balance of labels
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, df_sample['label'], test_size=0.2, random_state=42, stratify=df_sample['label']
    )

    print("🌲 Training Final Random Forest (The 60% Push)...")
    # n_estimators=200: Adding more voters to the committee for a more stable result
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=35, 
        min_samples_split=4,
        random_state=42, 
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Final Accuracy Check
    y_pred = rf_model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"🎯 FINAL TARGET ACCURACY: {round(score*100, 2)}%")

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