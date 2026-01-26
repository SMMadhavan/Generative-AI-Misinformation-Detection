import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def run_viva_report():
    print("\n🏢 THE MASTER DATA POOL: FINAL ARCHITECTURE")
    print("-" * 65)
    
    # Paths to your combined files
    # Note: Use '../data/...' if running from inside 'src' folder
    # Use 'data/...' if running from the project root
    train_path = 'data/train_data.csv'
    test_path = 'data/test_data.csv'
    
    # Attempt to adjust path if script is run from inside src
    if not os.path.exists(train_path):
        train_path = '../' + train_path
        test_path = '../' + test_path

    paths = [("Training Set (Textbook)", train_path), ("Testing Set (Final Exam)", test_path)]
    
    report_data = []

    for name, path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            total = len(df)
            real = (df['label'] == 1).sum()
            ai = (df['label'] == 0).sum()
            print(f"{name:<25} | Total: {total:<5} | Real: {real:<5} | AI: {ai:<5}")
            
            # Store data for the chart [cite: 51]
            report_data.append({'Set': name, 'Type': 'Human (Real)', 'Count': real})
            report_data.append({'Set': name, 'Type': 'Machine (AI)', 'Count': ai})
        else:
            print(f"⚠️ Warning: {path} not found. Ensure main.py has run successfully!")

    # Visualization block
    if report_data:
        viz_df = pd.DataFrame(report_data)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=viz_df, x='Set', y='Count', hue='Type', palette='viridis')
        plt.title("Data Quality Dimensions: Class Balance Verification")
        plt.ylabel("Number of Samples")
        plt.xlabel("Dataset Split")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save as artifact for Day 9 portfolio [cite: 113, 114]
        if not os.path.exists('reports'): os.makedirs('reports')
        plt.savefig('reports/data_balance_check.png')
        print(f"\n✅ Visual Insight saved to: reports/data_balance_check.png")
        plt.show()

if __name__ == "__main__":
    run_viva_report()