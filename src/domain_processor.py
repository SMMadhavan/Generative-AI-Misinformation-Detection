import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def apply_domain_tags():
    """Step 1: Tagging the data and showing terminal stats"""
    path = 'data/processed/master_cleaned.csv'
    if not os.path.exists(path):
        print("❌ Error: 'data/processed/master_cleaned.csv' not found.")
        print("Please ensure your cleaning pipeline has been run first!")
        return None
    
    print("🔍 Loading data for Domain Audit...")
    df = pd.read_csv(path)

    # The 'Sorting Hat' logic to categorize text
    def get_domain(text):
        text = str(text).lower()
        if any(w in text for w in ['market', 'stock', 'bank', 'finance', 'economy', 'business']):
            return 'Finance'
        elif any(w in text for w in ['school', 'education', 'university', 'student', 'teacher']):
            return 'Education'
        elif any(w in text for w in ['government', 'election', 'minister', 'policy', 'trump', 'biden']):
            return 'Politics'
        elif any(w in text for w in ['covid', 'doctor', 'hospital', 'science', 'health', 'medical']):
            return 'Healthcare'
        return 'General News'

    print("🏷️ Applying Domain Tags (this may take a moment)...")
    df['domain'] = df['text'].apply(get_domain)

    # Save the new version
    output_path = 'data/processed/master_with_domains.csv'
    df.to_csv(output_path, index=False)
    print(f"✅ Domain tagging complete. File saved to: {output_path}")
    
    print("\n📊 DOMAIN DISTRIBUTION (%)")
    stats = df['domain'].value_counts(normalize=True) * 100
    # Rounded to 2 decimal places with a % sign added for clarity
    print(stats.round(2).astype(str) + '%')
    # ---------------------------
    
    return df

def plot_domains(df):
    """Step 2: Creating the visualization for the reports folder"""
    if df is None: return

    print("\n🎨 Generating Visualization Chart...")
    
    # Set the visual style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    
    # Create the plot - sorted from highest to lowest frequency
    order = df['domain'].value_counts().index
    ax = sns.countplot(data=df, x='domain', palette='viridis', order=order)
    
    # Add titles and labels
    plt.title("Project Health: Dataset Diversity by Domain", fontsize=15, fontweight='bold', pad=20)
    plt.xlabel("Topic Domain", fontsize=12)
    plt.ylabel("Number of Articles", fontsize=12)

    # Add count labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height()):,}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=10, fontweight='bold')

    # Ensure reports folder exists for Checklist 3
    if not os.path.exists('reports'):
        os.makedirs('reports')

    # Save the chart and display it
    plt.tight_layout()
    plt.savefig('reports/domain_diversity.png')
    print("📂 Chart saved to: reports/domain_diversity.png")
    plt.show()

if __name__ == "__main__":
    # Execute the tagging and capture the dataframe
    processed_df = apply_domain_tags()
    
    # Generate the visual chart if data loading was successful
    if processed_df is not None:
        plot_domains(processed_df)
    
    print("\n✨ Domain Analysis Finished Successfully!")