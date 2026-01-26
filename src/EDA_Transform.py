import matplotlib.pyplot as plt
import pandas as pd
import os

def visualize_feature_transformation():
    print("📊 Visualizing Feature Transformation (The 'Funnel' Effect)...")

    # 1. Define the Raw Features from each source
    # These represent the 'Before' state from your datasets [cite: 51]
    raw_features = {
        'ISOT Dataset': ['title', 'text', 'subject', 'date'],
        'WELFake Dataset': ['index', 'title', 'text', 'label'],
        'GenAI Dataset': ['id', 'post_id', 'platform', 'timestamp', 'date', 'time', 'text', 'readability', 'label']
    }

    # 2. Define the Target Features (The 'After' state)
    target_features = ['text', 'label']

    # 3. Create the Visualization
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Colors for the sources
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    y_pos = [2, 1, 0] # Positions for the three bars
    
    for i, (source, features) in enumerate(raw_features.items()):
        # Draw the "Raw" box
        ax.barh(y_pos[i], len(features), color=colors[i], alpha=0.6, label=f"Raw {source}")
        ax.text(0.1, y_pos[i], f"{source}: {', '.join(features)}", va='center', fontweight='bold')
        
        # Draw the arrow showing the 'Merge' [cite: 65]
        ax.annotate('', xy=(len(features), y_pos[i]), xytext=(12, 1),
                    arrowprops=dict(arrowstyle="->", lw=2, color='gray', alpha=0.3))

    # 4. Draw the "Final Destination" box (Master_Cleaned.csv)
    ax.barh(1, len(target_features), left=12, color='gold', alpha=0.8)
    ax.text(12.2, 1, f"MASTER_CLEANED: {', '.join(target_features)}", va='center', fontweight='bold', color='black')

    # Formatting
    plt.title("Feature Evolution: How 'load_and_merge' Standardized the Data\n(Transforming Disparate Columns into a Unified Format)", fontsize=14)
    plt.xlabel("Number of Features (Columns)")
    ax.set_yticks([]) # Hide Y axis labels
    ax.set_xlim(0, 16)
    
    # 5. Save the Artifact for Day 9 Portfolio [cite: 113]
    if not os.path.exists('reports'): os.makedirs('reports')
    plt.savefig('reports/feature_transformation_flow.png')
    
    print("✅ Visualizing the transformation of features into [text, label].")
    print("💾 Chart saved to reports/feature_transformation_flow.png")
    plt.show()

if __name__ == "__main__":
    visualize_feature_transformation()