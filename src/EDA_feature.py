import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_data_merging():
    print("🔄 Simulating 'load_and_merge' process...")

    # 1. Define the "Ingredients" (Simulating the row counts from your terminal output)
    # These numbers match the ~117k raw rows seen in your terminal screenshot
    data_sources = {
        'ISOT (True/Fake)': 44898,
        'WELFake': 72134,
        'GenAI Misinfo': 466  # The specific GenAI dataset size
    }

    # 2. Create the Visualization
    plt.figure(figsize=(10, 7))
    colors = ['#ff9999','#66b3ff','#99ff99']
    
    # Create a Donut Chart
    plt.pie(data_sources.values(), labels=data_sources.keys(), 
            autopct='%1.1f%%', startangle=140, colors=colors, 
            pctdistance=0.85, explode=(0.05, 0.05, 0.05))

    # Draw a circle in the center to make it a donut
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.title("Master_Cleaned.csv: Data Source Contribution\n(How 'load_and_merge' Standardized the Features)", fontsize=14)
    
    # Add a legend that explains the "Standardization"
    plt.legend(title="Standardized to: [text, label]", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # 3. Save the Artifact for Day 9 Portfolio
    if not os.path.exists('dashboard'): os.makedirs('dashboard')
    plt.savefig('dashboard/data_unification_donut.png')
    
    print("✅ Insight: Even though sources vary in size, they are now unified by 'text' and 'label' columns.")
    print("💾 Chart saved to dashboard/data_unification_donut.png")
    plt.show()

if __name__ == "__main__":
    visualize_data_merging()