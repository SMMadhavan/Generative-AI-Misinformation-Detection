import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

def run_baseline_comparison():
    print("🚀 Starting Baseline Model Comparison...")
    
    # 1. Load Data
    df = pd.read_csv('data/processed/master_cleaned.csv')
    
    # 2. Prepare Features (X) and Target (y)
    # This automatically selects ONLY the columns with numbers (your linguistic features)
    X = df.select_dtypes(include=[np.number]).drop(columns=['label'], errors='ignore')
    y = df['label']
    
    # Safety Check: If X is still empty, the model cannot run
    if X.empty:
        print("❌ Error: No numerical features found! Ensure your linguistic features are in the CSV.")
        return

    print(f"📊 Features being used: {list(X.columns)}")
    
    # 3. Stratified Train-Test Split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Define models for comparison [cite: 2, 22]
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000), # [cite: 23]
        "Decision Tree": DecisionTreeClassifier(random_state=42), # [cite: 24]
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42) # [cite: 34, 39]
    }
    
    results = []

    for name, model in models.items():
        print(f"--- Training {name} ---")
        start_time = time.time()
        
        # Fit Model [cite: 16]
        model.fit(X_train, y_train)
        fit_time = time.time() - start_time # [cite: 30]
        
        # Predict Probabilities for ROC-AUC [cite: 14]
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        # Calculate Metrics [cite: 11]
        acc = accuracy_score(y_test, y_pred) # [cite: 12]
        auc = roc_auc_score(y_test, y_proba) # [cite: 14]
        
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "ROC-AUC": round(auc, 4),
            "Fit Time": f"{round(fit_time, 4)}s"
        })

    # 4. Display Comparison Table [cite: 28, 29]
    comparison_df = pd.DataFrame(results)
    print("\n📊 BASELINE MODEL PERFORMANCE TABLE")
    print(comparison_df.to_string(index=False))
    
    # 5. Feature Importance Visualization 
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_importances.values, y=feat_importances.index, palette='viridis')
    plt.title("Random Forest - Feature Importance (Linguistic DNA)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png')
    print("\n✅ Saved Feature Importance chart to reports/feature_importance.png")

if __name__ == "__main__":
    run_baseline_comparison()