import joblib
import numpy as np
import os
from scipy.sparse import hstack

# --- 1. NARRATIVE AUDIT ENGINE (Synchronized with Pipeline) ---
def explain_misinformation(model, vectorizer, selector, text):
    # 1. Transform text
    X_raw = vectorizer.transform([text])
    X_selected = selector.transform(X_raw)
    
    # 2. Extract weights from the first calibrated classifier
    # This is where the 'learning' lives for LinearSVC
    base_model = model.calibrated_classifiers_[0].base_estimator
    weights = base_model.coef_[0]

    # 3. Get the names of the 500 words we selected
    feature_names = vectorizer.get_feature_names_out()
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    # 4. Find matches in the current text
    found_triggers = []
    # We clean the text to match the vectorizer (lowercase, no punctuation)
    clean_words = text.lower().replace('.', '').replace(',', '').split()
    
    for word in clean_words:
        if word in selected_features:
            idx = selected_features.index(word)
            # CHANGE: We look for ANY positive weight (above 0.0)
            # Positive weight = Word is associated with Misinformation/AI
            if weights[idx] > 0.0: 
                found_triggers.append(word)
    
    return list(set(found_triggers))

# --- 2. DYNAMIC PREDICTION RUNNER ---
def dynamic_predict():
    print("🛡️ --- DYNAMIC AI NEWS AUDITOR ---")
    
    # A. Load Pipeline Components
    try:
        model = joblib.load('models/misinfo_detector_model.joblib')
        tfidf = joblib.load('models/tfidf_vectorizer.joblib')
        scaler = joblib.load('models/dna_scaler.joblib')
        selector = joblib.load('models/feature_selector.joblib')
        print("✅ Status: All model components loaded successfully.")
    except Exception as e:
        print(f"❌ Error: Could not load model files. Run master_pipeline.py first!")
        return

    # B. Get Dynamic Input
    print("\n✍️ PASTE THE ARTICLE TEXT BELOW (Press Enter to analyze):")
    user_text = input("> ")

    if not user_text.strip():
        print("⚠️ Warning: No text entered. Exiting...")
        return

    # C. Sequential Processing Pipeline
    # 1. Vectorize text
    X_text_raw = tfidf.transform([user_text])
    X_text_selected = selector.transform(X_text_raw)
    
    # 2. Add Scaled DNA baseline
    # We use neutral values (0.5) for the structural senses in this quick test
    dna_baseline = scaler.transform([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
    
    # 3. Combine
    X_final = hstack([X_text_selected, dna_baseline])

    # D. Output Prediction & Audit
    prediction = model.predict(X_final)[0]
    prob = model.predict_proba(X_final)[0]

    print("\n📊 --- ANALYSIS RESULTS ---")
    if prediction == 1:
        print(f"🚩 PREDICTION: LIKELY AI-GENERATED MISINFORMATION")
        print(f"📈 Confidence: {prob[1]*100:.2f}%")
        
        # Run Narrative Audit to explain 'WHY'
        triggers = explain_misinformation(model, tfidf, selector, user_text)
        if triggers:
            print(f"🔍 Audit Found Triggers: {', '.join(triggers)}")
            print("💡 These words are linked to historical misinfo patterns in our 2024-2025 data.")
    else:
        print(f"✅ PREDICTION: LIKELY HUMAN-WRITTEN NEWS")
        print(f"📉 Confidence: {prob[0]*100:.2f}%")

if __name__ == "__main__":
    dynamic_predict()