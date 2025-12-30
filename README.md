# AI-Driven Generative-AI Misinformation Detection & Trend Analytics

## 🚀 Project Overview
This project focuses on detecting AI-generated misinformation using machine learning and natural language processing techniques. 
The system analyzes textual features to distinguish between human-written news and synthetic misinformation, providing analytical insights through data visualization.

## ✅ Current Milestones Achieved
- **Data Engineering:** Successfully merged and cleaned a massive dataset of **117,493 rows** from ISOT, WELFake, and GenAI sources.
- **Exploratory Data Analysis (EDA):** - Confirmed a perfect **50/50 class balance** (~60k real vs. ~60k fake articles).
    - Generated Word Clouds to visualize high-frequency terms in misinformation.
- **Baseline Modeling:** - Implemented a **Passive Aggressive Classifier**.
    - Established an initial baseline accuracy of **51.3%** using a memory-stable 50,000-row sample.

## 📊 Visual Dashboard
The following insights are currently saved in the `/dashboard` folder:
- `class_distribution.png`: Visual proof of dataset balance.
- `wordcloud_fake.png`: Key vocabulary in fake news.
- `wordcloud_real.png`: Key vocabulary in real news.

## 🛠 Tech Stack
- **Language:** Python
- **Libraries:** Pandas, Scikit-learn, Matplotlib, Seaborn, WordCloud
- **ML Model:** Passive Aggressive Classifier (Baseline)

## 📁 Repository Structure
- `data/processed/`: Master cleaned dataset.
- `dashboard/`: Visual assets and charts.
- `src/`: Source code for data processing.
- `main.py`: Main execution script.
- `venv/`: Virtual environment.

## 📌 Status: PAUSED
The project is currently paused after establishing the data pipeline and baseline model. Improving the model prediction accuracy is currently being focused on and the Next steps will involve **Hyperparameter Tuning** and **N-gram optimization** to increase model accuracy.