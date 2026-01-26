# 🛡️ Neural Auditor: AI-Driven Misinformation Detection & Trend Analytics

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0-green?style=for-the-badge&logo=flask)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)

## 🚀 Project Overview
**Neural Auditor** is an enterprise-grade forensic tool designed to detect AI-generated misinformation and synthetic text patterns. By leveraging advanced Natural Language Processing (NLP) and a hybrid machine learning pipeline, the system analyzes "Linguistic DNA"—features like burstiness, perplexity, and semantic richness—to distinguish between human-written journalism and machine-generated hallucinations.

Unlike simple classifiers, this project provides a **Forensic Dashboard** that offers real-time visualization, domain-specific sensitivity tuning (Politics, Healthcare, etc.), and automated PDF reporting for content moderation teams.

---

## ⚡ Key Features

### 🧠 **Forensic AI Analysis**
* **Hybrid Detection Engine:** Combines TF-IDF vectorization with "Linguistic DNA" features (sentence complexity, buzzword density) to detect robotic writing patterns.
* **Domain Context Awareness:** Adjusts sensitivity thresholds dynamically based on the topic (e.g., stricter rules for *Healthcare* vs. *Entertainment*).

### 📊 **Interactive Dashboard**
* **Real-Time Visualization:** Radar charts displaying linguistic signatures (Uniformity vs. Richness).
* **Live Threat Logs:** SQLite-backed audit trail of all scanned content.
* **Visual Indicators:** Clear "Green/Yellow/Red" verdict system for Verified Human, Suspicious, or AI-Generated content.

### 📑 **Automated Reporting**
* **PDF Generation:** One-click generation of professional forensic reports containing the analyzed text, verdict metadata, and confidence scores.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Core Logic** | Python 3.x |
| **Web Framework** | Flask (Server-side rendering) |
| **Machine Learning** | Scikit-learn (LinearSVC, RandomForest, TF-IDF) |
| **Data Processing** | Pandas, NumPy, NLTK |
| **Database** | SQLite (Lightweight audit logging) |
| **Frontend** | HTML5, Bootstrap 5, Chart.js |
| **Reporting** | FPDF (PDF Generation) |

---

## ✅ Development Milestones

### **Phase 1: Data Engineering**
* Successfully merged and cleaned a massive dataset of **117,493 rows** from ISOT, WELFake, and GenAI sources.
* Achieved a perfect **50/50 class balance** (~60k real vs. ~60k fake articles) to prevent model bias.

### **Phase 2: Exploratory Data Analysis (EDA)**
* Analyzed high-frequency terms using Word Clouds to identify "hallucination triggers."
* Visualized class distribution to ensure data integrity.

### **Phase 3: Modeling & Optimization**
* **Baseline:** Established initial accuracy of **51.3%** using memory-stable sampling.
* **Current Production Model:** Migrated to a **Calibrated LinearSVC Ensemble** with custom feature engineering, significantly improving detection rates on complex synthetic text.

---

## 💻 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/ai-misinfo-detection.git](https://github.com/yourusername/ai-misinfo-detection.git)
    cd ai-misinfo-detection
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**
    ```bash
    python app.py
    ```
    *Access the dashboard at `http://127.0.0.1:5000`*

---

## 📁 Repository Structure

```text
├── data/
│   └── processed/          # Master cleaned dataset (CSV)
├── models/                 # Pre-trained .joblib models & vectorizers
├── src/
│   ├── static/             # CSS, Images, JS assets
│   ├── templates/          # HTML templates for Flask
│   └── app.py              # Main application entry point
├── reports/                # Saved EDA visualizations
├── neural_db_v2.sqlite     # Live audit database
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```
## 📌 Future Roadmap

* **N-gram Optimization:** Fine-tuning the vectorizer to catch multi-word AI phrases.
* **Deep Learning Integration:** Exploring LSTM/BERT for deeper semantic analysis.
* **API Deployment:** Dockerizing the Flask app for cloud deployment.