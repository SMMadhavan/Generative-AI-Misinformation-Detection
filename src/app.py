import os
import time
import sqlite3
import joblib
import pandas as pd
import numpy as np
import warnings

import shutil
if os.path.exists('cachedir'):
    shutil.rmtree('cachedir')

from flask import Flask, render_template, request, make_response
from scipy.sparse import hstack
from fpdf import FPDF  

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

warnings.filterwarnings('ignore')

app = Flask(__name__)
DB_PATH = 'neural_db_v2.sqlite'
MODEL_DIR = 'models/'
CSV_SOURCE_PATH = 'data/processed/master_cleaned.csv'

# Domain Sensitivity Configuration
DOMAIN_CONFIG = {
    "politics": { "label": "Politics", "threshold": 0.10 },
    "finance": { "label": "Finance", "threshold": 0.75 },
    "news": { "label": "General News", "threshold": 0.65 },
    "education": { "label": "Education", "threshold": 0.65 },
    "healthcare": { "label": "Healthcare", "threshold": 0.80 }
}

# --- 1. DATABASE LAYER ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # FORCE RESET THE TABLE TO FIX THE COLUMN ISSUE
    c.execute("DROP TABLE IF EXISTS training_dataset") 
    c.execute('''CREATE TABLE training_dataset (id INTEGER PRIMARY KEY, text TEXT, label INTEGER)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS audit_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, input_text TEXT, domain TEXT, verdict TEXT, confidence REAL)''')
    conn.commit()
    return conn

def migrate_csv_to_db():
    conn = init_db()
    if conn.execute("SELECT COUNT(*) FROM training_dataset").fetchone()[0] == 0:
        print("\n" + "="*60)
        print("🗄️  DATABASE MIGRATION PROTOCOL (CSV -> SQL)")
        print("="*60)
        if os.path.exists(CSV_SOURCE_PATH):
            try:
                print(f"   [INFO] Reading Raw Data: {CSV_SOURCE_PATH}")
                df = pd.read_csv(CSV_SOURCE_PATH).dropna().drop_duplicates()
                print(f"   [INFO] Data Shape: {df.shape}")
                print("   [INFO] Inserting into SQLite Database...")
                df[['text', 'label']].to_sql('training_dataset', conn, if_exists='append', index=False)
                print(f"   ✅ SUCCESS: Migrated {len(df)} records.")
            except Exception as e: print(f"   ❌ ERROR: {e}")
        else:
            print(f"   ❌ ERROR: CSV Not Found at {CSV_SOURCE_PATH}")
    conn.close()

def log_audit(text, domain, verdict, conf):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO audit_logs (input_text, domain, verdict, confidence) VALUES (?, ?, ?, ?)",
                 (text[:10000], domain, verdict, conf)) # Store more text
    conn.commit()
    conn.close()

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    stats = {
        "train_count": conn.execute("SELECT COUNT(*) FROM training_dataset").fetchone()[0],
        "total": conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0],
        "threats": conn.execute("SELECT COUNT(*) FROM audit_logs WHERE verdict LIKE '%AI%'").fetchone()[0],
        "history": pd.read_sql("SELECT * FROM audit_logs ORDER BY id DESC LIMIT 5", conn).to_dict('records')
    }
    conn.close()
    return stats

# --- 2. FORENSIC FEATURE ENGINEERING ---
def get_advanced_features(text):
    text = str(text).lower()
    words = text.split()
    if len(words) == 0: return [0.0] * 7, 0
    
    sentences = [s for s in text.split('.') if s.strip()]
    sent_lengths = [len(s.split()) for s in sentences]
    if not sent_lengths: sent_lengths = [0]
    
    uniformity = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
    richness = len(set(words)) / len(words)
    buzzwords = ['pivotal', 'delve', 'comprehensive', 'resonate', 'unravel', 'provisionally', 'tapestry', 'synergistic', 'paradigm', 'underscores', 'multifaceted', 'nuance', 'robust', 'landscape']
    buzz_count = sum([1 for w in words if w in buzzwords])
    buzz_density = buzz_count / len(words)
    burstiness = np.var(sent_lengths) if len(sent_lengths) > 1 else 0
    punc_count = sum([1 for char in text if char in '.,!?;:'])
    punc_density = punc_count / len(words)
    avg_sent_len = np.mean(sent_lengths)
    long_words = sum([1 for w in words if len(w) > 7])
    complexity_ratio = long_words / len(words)

    return [float(uniformity), float(richness), float(buzz_density), float(burstiness), float(complexity_ratio), float(punc_density), float(avg_sent_len)], buzz_count

def normalize_dna_for_ui(raw_dna):
    """Safely scales features 0-1 for the Radar Chart (Visuals only)."""
    try:
        def clean(val, scale, cap=1.0):
            if np.isnan(val) or np.isinf(val): return 0.0
            return min(cap, max(0.0, float(val) / scale)) # Ensure float conversion
            
        result = [
            clean(raw_dna[0], 5.0),    # Uniformity
            clean(raw_dna[1], 1.0),    # Richness
            clean(raw_dna[2], 0.05),   # Buzz
            clean(raw_dna[3], 50.0),   # Burstiness
            clean(raw_dna[4], 1.0),    # Complexity
            clean(raw_dna[5], 0.2),    # Punct
            clean(raw_dna[6], 30.0)    # Length
        ]
        return result
    except: return [0.5] * 7

def interpret_dna(dna_values):
    explanations = []
    if dna_values[3] < 20: explanations.append(("Low Burstiness", "Robotic, uniform sentence structure."))
    else: explanations.append(("High Burstiness", "Natural, varied human writing."))
    if dna_values[1] < 0.4: explanations.append(("Low Richness", "Vocabulary is repetitive."))
    if dna_values[2] > 0.0: explanations.append(("AI Artifacts", "Contains specific AI hallucinations."))
    return explanations

def extract_misinfo_triggers(text):
    triggers = []
    buzzwords = ['pivotal', 'delve', 'comprehensive', 'resonate', 'unravel', 'provisionally', 'tapestry', 'synergistic', 'paradigm', 'underscores', 'multifaceted', 'nuance', 'robust', 'landscape']
    for word in text.lower().split():
        if word in buzzwords: triggers.append((word, "High Probability AI Artifact"))
    if not triggers: triggers.append(("Structure", "Low Variance / Robotic Pattern"))
    return sorted(list(set(triggers)), key=lambda x: x[0])[:6]

# --- 3. TRAINING PIPELINE (REQUIRED FOR TERMINAL OUTPUT) ---
def train_and_evaluate():
    print("\n" + "="*80)
    print("🚀 NEURAL AUDITOR: INITIALIZING DATA SCIENCE PIPELINE")
    print("="*80)
    
    #migrate_csv_to_db()
    conn = sqlite3.connect(DB_PATH)
    try: 
        print("   [PIPELINE] Fetching Data from SQLite DB...")
        df = pd.read_sql("SELECT * FROM training_dataset", conn)
    except: df = pd.DataFrame()
    conn.close()
    
    if df.empty:
        print("❌ FATAL: No training data found.")
        return

    print(f"   [DATA] Total Records Loaded: {len(df)}")
    if len(df) > 20000:
        print("   [OPTIMIZATION] Downsampling to 20,000 records for performance...")
        # SAFEST METHOD: This samples 10k items per group without deleting columns
        try:
            df = df.groupby('label').sample(n=10000, random_state=42)
        except ValueError:
            # Fallback if one group is too small: just take the top 20k
            df = df.sample(n=20000, random_state=42)

    # ================= CRITICAL DATA SANITIZATION =================
    print(f"   [CLEANING] Removing empty rows... (Before: {len(df)})")
    # 1. Drop rows where text is missing/empty
    df.dropna(subset=['text', 'label'], inplace=True)
    # 2. Ensure all text is actually string format
    df['text'] = df['text'].astype(str)
    print(f"   [CLEANING] Data clean. (After: {len(df)})")
    # ==============================================================

    print("\n⚙️  FEATURE ENGINEERING:")
    print("   - Extracting Linguistic DNA...")
    features_list = [get_advanced_features(t)[0] for t in df['text']]
    custom_features = np.array(features_list, dtype=np.float64)
    
    print("   - Vectorizing Text (TF-IDF)...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,1))
    X_text = tfidf.fit_transform(df['text'].astype(str))
    
    selector = SelectKBest(score_func=chi2, k=500)
    
    # ================= EMERGENCY FIX START =================
    # 1. Force all columns to lowercase and remove spaces
    df.columns = df.columns.str.strip().str.lower()
    
    # 2. Rename common variations if 'label' is still missing
    rename_map = {
        'class': 'label',
        'target': 'label',
        'fake': 'label', 
        'label ': 'label'
    }
    df.rename(columns=rename_map, inplace=True)

    # 3. DEBUG PRINT: Check if it exists now
    print(f"DEBUG: Final Columns available: {df.columns.tolist()}")
    
    if 'label' not in df.columns:
        raise ValueError(f"CRITICAL ERROR: 'label' column is MISSING. Found only: {df.columns.tolist()}")
    # ================= EMERGENCY FIX END =================
    
    X_text_selected = selector.fit_transform(X_text, df['label'])
    
    scaler = StandardScaler()
    custom_features_scaled = scaler.fit_transform(custom_features)
    X_final = hstack([X_text_selected, custom_features_scaled])

    X_train, X_test, y_train, y_test = train_test_split(X_final, df['label'], test_size=0.2, random_state=42)

    print("\n🏆 MODEL COMPARISON TABLE:")
    print(f"   {'Algorithm':<25} | {'Accuracy':<10} | {'ROC-AUC':<10} | {'Fit Time'}")
    print("   " + "-"*65)
    
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("Random Forest", RandomForestClassifier(n_estimators=50, max_depth=10)),
        ("Linear SVC (Selected)", LinearSVC(C=10, max_iter=2000))
    ]
    
    for name, clf in models:
        start_time = time.time()
        clf.fit(X_train, y_train)
        end_time = time.time()
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        try:
            if hasattr(clf, "decision_function"): probs = clf.decision_function(X_test)
            else: probs = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
        except: auc = 0.0
        print(f"   {name:<25} | {round(acc*100, 2)}%      | {round(auc, 3):<10} | {round(end_time - start_time, 3)}s")

    print("\n🚀  TRAINING FINAL ENSEMBLE (LinearSVC + Calibration)...")
    base_svm = LinearSVC(C=10, random_state=42, max_iter=3000)
    final_model = CalibratedClassifierCV(base_svm, cv=3)
    final_model.fit(X_train, y_train)
    
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    joblib.dump(final_model, MODEL_DIR + 'misinfo_detector_model.joblib')
    joblib.dump(tfidf, MODEL_DIR + 'tfidf_vectorizer.joblib')
    joblib.dump(scaler, MODEL_DIR + 'dna_scaler.joblib')
    joblib.dump(selector, MODEL_DIR + 'feature_selector.joblib')
    print("🎉 SYSTEM READY. LAUNCHING FLASK INTERFACE...\n")

# --- 4. FLASK INTEGRATION ---

@app.route('/download_report')
def download_report():
    # 1. Fetch latest log
    conn = sqlite3.connect(DB_PATH)
    try: data = pd.read_sql("SELECT * FROM audit_logs ORDER BY id DESC LIMIT 1", conn).to_dict('records')[0]
    except: return "No data found."
    conn.close()

    # 2. Setup PDF
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 10)
            self.set_text_color(100)
            self.cell(0, 10, 'NEURAL AUDITOR | ENTERPRISE FORENSICS', 0, 1, 'R')
            self.set_draw_color(200)
            self.line(10, 20, 200, 20)
            self.ln(15)
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(150)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    # Helper for special characters
    def safe_text(text):
        return text.encode('latin-1', 'replace').decode('latin-1')

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", 'B', 22)
    pdf.set_text_color(33)
    pdf.cell(0, 10, "Forensic Analysis Report", ln=True)
    pdf.ln(5)

    # Executive Summary Box
    pdf.set_fill_color(245, 245, 245)
    pdf.rect(10, pdf.get_y(), 190, 35, 'F')
    pdf.set_y(pdf.get_y() + 5)

    # Metadata
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(35, 8, "  Timestamp:", ln=0)
    pdf.set_font("Arial", '', 11)
    pdf.cell(60, 8, str(data['timestamp']), ln=1)

    pdf.set_font("Arial", 'B', 11)
    pdf.cell(35, 8, "  Context:", ln=0)
    pdf.set_font("Arial", '', 11)
    pdf.cell(60, 8, safe_text(data['domain'].upper()), ln=1)

    pdf.set_font("Arial", 'B', 11)
    pdf.cell(35, 8, "  Verdict:", ln=0)
    
    if "AI" in data['verdict'] or "SUSPICIOUS" in data['verdict']:
        pdf.set_text_color(220, 53, 69) # Red
    else:
        pdf.set_text_color(25, 135, 84) # Green
        
    pdf.cell(60, 8, safe_text(data['verdict']), ln=1)
    
    pdf.set_text_color(33)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(35, 8, "  Confidence:", ln=0)
    pdf.set_font("Arial", '', 11)
    pdf.cell(60, 8, f"{round(data['confidence'], 2)}%", ln=1)
    pdf.ln(15)

    # Full Narrative
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Analyzed Content", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(50)
    # This captures infinite text wrapping
    pdf.multi_cell(0, 7, txt=safe_text(data['input_text']))

    response = make_response(pdf.output(dest='S').encode('latin-1'))
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=Neural_Forensic_Report.pdf'
    return response

@app.route('/', methods=['GET', 'POST'])
def index():
    analysis = None
    domain_selected = "politics"
    stats = get_stats()

    if request.method == 'POST':
        user_text = request.form.get('text_input', '')
        domain_selected = request.form.get('domain_select', 'politics')

        if user_text.strip():
            try:
                model = joblib.load(MODEL_DIR + 'misinfo_detector_model.joblib')
                tfidf = joblib.load(MODEL_DIR + 'tfidf_vectorizer.joblib')
                scaler = joblib.load(MODEL_DIR + 'dna_scaler.joblib')
                selector = joblib.load(MODEL_DIR + 'feature_selector.joblib')
            except: return "Error: Models not ready."

            X_text_raw = tfidf.transform([user_text])
            X_text_selected = selector.transform(X_text_raw)
            dna_raw, buzz_count = get_advanced_features(user_text)
            dna_scaled = scaler.transform([dna_raw])
            X_final = hstack([X_text_selected, dna_scaled])
            
            # Prediction
            probs = model.predict_proba(X_final)[0]
            prob_human = probs[1]
            penalty = buzz_count * 0.02 # Strict but fair penalty
            adjusted_score = prob_human - penalty
            threshold = DOMAIN_CONFIG[domain_selected]['threshold']

            if adjusted_score >= threshold: verdict = "VERIFIED_HUMAN"
            elif adjusted_score < threshold and adjusted_score > 0.45: verdict = "SUSPICIOUS"
            else: verdict = "AI_GENERATED"

            triggers = []
            if verdict != "VERIFIED_HUMAN":
                triggers = extract_misinfo_triggers(user_text)

            analysis = {
                "verdict_code": verdict,
                "score": round(adjusted_score * 100, 2),
                "triggers": triggers,
                "dna_visual": normalize_dna_for_ui(dna_raw), # THIS IS NOW FIXED
                "dna_explained": interpret_dna(dna_raw),
                "threshold": threshold
            }
            log_audit(user_text, domain_selected, verdict, analysis['score'])
            stats = get_stats()

    return render_template('index.html', analysis=analysis, domains=DOMAIN_CONFIG, selected_domain=domain_selected, stats=stats)

if __name__ == '__main__':
    train_and_evaluate() # Run Pipeline on Start
    app.run(debug=True, use_reloader=False)