import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, train_test_split

def plot_learning_curves():
    print("📈 Generating Learning Curves (Review 2: Expected Output)...")
    
    # 1. Load Data
    df = pd.read_csv('data/processed/master_cleaned.csv')
    X = df.select_dtypes(include=[np.number]).drop(columns=['label'], errors='ignore')
    y = df['label']

    # 2. Define the Model (using the best parameters we found in Step 4)
    model = RandomForestClassifier(n_estimators=200, max_depth=20, bootstrap=False, random_state=42)

    # 3. Calculate Learning Curve Data
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=3, scoring='accuracy', n_jobs=1, 
        train_sizes=np.linspace(0.1, 1.0, 5)
    )

    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")

    plt.title("Learning Curves (Diagnosing Over/Underfitting)", fontsize=14)
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.grid(True)
    
    # Save for documentation
    plt.savefig('reports/learning_curves.png')
    print("✅ Saved Learning Curve to reports/learning_curves.png")
    plt.show()

if __name__ == "__main__":
    plot_learning_curves()