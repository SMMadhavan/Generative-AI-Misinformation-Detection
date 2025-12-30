import pandas as pd
import os
import re
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# --- STEP 1: DATA LOADING & CLEANING ---

def load_and_merge():
    # 1. Load ISOT (True=1, Fake=0)
    true_df = pd.read_csv('data/True.csv')
    fake_df = pd.read_csv('data/Fake.csv')
    true_df['label'] = 1
    fake_df['label'] = 0
    isot = pd.concat([true_df, fake_df])[['text', 'label']]

    # 2. Load WELFake
    welfake = pd.read_csv('data/WELFake_Dataset.csv')
    welfake = welfake.rename(columns={'label': 'label'})[['text', 'label']]

    # 3. Load GenAI
    genai = pd.read_csv('data/generative_ai_misinformation_dataset.csv')
    genai = genai.rename(columns={'is_misinformation': 'label'})
    genai['label'] = genai['label'].apply(lambda x: 0 if x == 1 else 1) 
    genai = genai[['text', 'label']]

    # 4. Final Merge
    master_df = pd.concat([isot, welfake, genai], ignore_index=True)
    master_df = master_df.dropna()
    print(f"✅ Success! Master Dataset created with {len(master_df)} rows.")
    return master_df

def clean_text(text):
    text = str(text).lower() 
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) 
    text = re.sub(r'\n', '', text) 
    text = re.sub(r'\w*\d\w*', '', text) 
    return text

# --- STEP 2: CUSTOM AI SIGNATURE FUNCTION ---

def get_features(text):
    """Extracts structural patterns that often distinguish AI from humans."""
    text = str(text).strip()
    if not text:
        return [0, 0]
        
    sentences = [s for s in text.split('.') if s.strip()]
    if not sentences:
        return [0, 0]
        
    # Average Sentence Length (AI is often very consistent)
    avg_sent_len = np.mean([len(s.split()) for s in sentences])
    # Punctuation Density (AI uses punctuation very predictably)
    punc_count = sum([1 for char in text if char in '?!,;:']) / (len(text) + 1)
    
    return [avg_sent_len, punc_count]

# --- STEP 3: MAIN EXECUTION ---

if __name__ == "__main__":
    # 1. Load and Clean
    df = load_and_merge()
    print("🧹 Cleaning text... please wait.")
    df['text'] = df['text'].apply(clean_text)

    # 2. Save Cleaned Data
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/master_cleaned.csv'
    df.to_csv(output_path, index=False)
    print(f"💾 DONE! Dataset saved at: {output_path}")

    # 3. Visualization: Class Distribution
    print("📊 Generating Class Distribution Chart...")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=df, palette='viridis')
    plt.title('Distribution of Real (1) vs. Fake (0) News')
    os.makedirs('dashboard', exist_ok=True)
    plt.savefig('dashboard/class_distribution.png')
    plt.close()

    # 4. Visualization: Word Clouds
    print("☁️ Generating Word Clouds...")
    fake_sample = df[df['label'] == 0]['text'].sample(n=5000, random_state=42).astype(str)
    real_sample = df[df['label'] == 1]['text'].sample(n=5000, random_state=42).astype(str)

    def save_cloud(text_series, title, filename):
        full_text = " ".join(text_series)
        wc = WordCloud(width=800, height=400, background_color='white', max_words=50).generate(full_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.savefig(f'dashboard/{filename}')
        plt.close()

    save_cloud(fake_sample, 'Common Words in Fake News', 'wordcloud_fake.png')
    save_cloud(real_sample, 'Common Words in Real News', 'wordcloud_real.png')

    # 5. Machine Learning Setup (Balanced 30k Sample)
    print("⚖️ Balancing Dataset for Training...")
    df_sample = df.groupby('label').apply(lambda x: x.sample(n=15000, random_state=42)).reset_index(drop=True)

    print("📏 Extracting AI Signatures...")
    custom_features = np.array([get_features(t) for t in df_sample['text']])

    print("✂️ Splitting data...")
    X_train_text, X_test_text, y_train, y_test, X_train_custom, X_test_custom = train_test_split(
        df_sample['text'], df_sample['label'], custom_features, test_size=0.2, random_state=42
    )

    # 6. Vectorization & Model Training
    print("🔢 Vectorizing Text & Combining Features...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train_text.astype(str))
    X_test_tfidf = tfidf.transform(X_test_text.astype(str))

    # Join the TF-IDF word math with our Custom AI Signatures
    X_train_final = hstack([X_train_tfidf, X_train_custom])
    X_test_final = hstack([X_test_tfidf, X_test_custom])

    print("🧠 Training Passive Aggressive Classifier...")
    pac = PassiveAggressiveClassifier(max_iter=1000, tol=1e-3, random_state=42)
    pac.fit(X_train_final, y_train)

    # 7. Final Results
    y_pred = pac.predict(X_test_final)
    score = accuracy_score(y_test, y_pred)
    print(f"🎯 FINAL SUCCESS! ENHANCED ACCURACY: {round(score*100, 2)}%")