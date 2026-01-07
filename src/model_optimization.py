import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split

def tune_random_forest():
    print("🎯 Starting Hyperparameter Optimization (Step 4)...")
    
    # 1. Load Data
    df = pd.read_csv('data/processed/master_cleaned.csv')
    
    # 2. Select numerical features (the ones we saved earlier)
    X = df.select_dtypes(include=[np.number]).drop(columns=['label'], errors='ignore')
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3. Define the Hyperparameter Grid (The "Dials" to turn)
    # This satisfies Review 2: Step 1 (Hyperparameter Tuning)
    param_dist = {
        'n_estimators': [50, 100, 200], # Number of trees
        'max_depth': [None, 10, 20],     # Depth of trees
        'min_samples_split': [2, 5],     # Minimum samples to split a node
        'bootstrap': [True, False]       # Method for sampling data
    }

    # 4. Initialize Randomized Search
    # This uses 3-fold Cross-Validation as required
    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
    estimator=rf, 
    param_distributions=param_dist, 
    n_iter=5, 
    cv=3,     
    verbose=2, 
    random_state=42, 
    n_jobs=1  # <--- Change this from -1 to 1
)

    # 5. Execute Tuning
    print("⏳ Searching for optimal settings... (This may take a moment)")
    random_search.fit(X_train, y_train)

    # 6. Output Results
    print("\n✅ HYPERPARAMETER TUNING RESULTS")
    print("-" * 30)
    print(f"Best Parameters found: {random_search.best_params_}")
    print(f"Best Cross-Validation Score: {round(random_search.best_score_, 4)}")
    
    return random_search.best_params_

if __name__ == "__main__":
    tune_random_forest()