import joblib
import numpy as np
import re
from scipy.sparse import hstack

# 1. Load the Brain
print("🧠 Loading the Elite Truth-Checker...")
model = joblib.load('models/misinfo_detector_model.joblib')
tfidf = joblib.load('models/tfidf_vectorizer.joblib')

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

def check_news(news_text):
    # Prepare the input for the Forest
    text_vectorized = tfidf.transform([news_text])
    fingerprints = np.array([get_advanced_features(news_text)])
    final_input = hstack([text_vectorized, fingerprints])
    
    # Get prediction and confidence
    prediction = model.predict(final_input)
    probability = model.predict_proba(final_input)[0]
    
    label = "✅ REAL NEWS" if prediction[0] == 1 else "🚨 FAKE / AI GENERATED"
    confidence = max(probability) * 100
    
    return f"\nAI Analysis: {label} ({confidence:.1f}% confidence)"

if __name__ == "__main__":
    print("\n--- Welcome to the AI News Guard (v2.0) ---")
    while True:
        user_input = input("\n📰 Paste news text here (or type 'exit'): ")
        if user_input.lower() == 'exit': break
        print(check_news(user_input))