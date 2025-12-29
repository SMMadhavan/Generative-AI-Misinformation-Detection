import pandas as pd
import os

def load_and_merge():
    # 1. Load ISOT (True=1, Fake=0)
    true_df = pd.read_csv('data/True.csv')
    fake_df = pd.read_csv('data/Fake.csv')
    true_df['label'] = 1
    fake_df['label'] = 0
    isot = pd.concat([true_df, fake_df])[['text', 'label']]

    # 2. Load WELFake (Assuming label 1=Real, 0=Fake)
    welfake = pd.read_csv('data/WELFake_Dataset.csv')
    welfake = welfake.rename(columns={'label': 'label'})[['text', 'label']]

    # 3. Load GenAI (Standardize to match)
    genai = pd.read_csv('data/generative_ai_misinformation_dataset.csv')
    # Rename their target column to 'label'
    genai = genai.rename(columns={'is_misinformation': 'label'})
    # IMPORTANT: If GenAI uses 1 for 'Misinfo', flip it to 0 to match our 0=Fake standard
    genai['label'] = genai['label'].apply(lambda x: 0 if x == 1 else 1) 
    genai = genai[['text', 'label']]

    # 4. The Final Merge
    master_df = pd.concat([isot, welfake, genai], ignore_index=True)
    master_df = master_df.dropna() # Remove any empty rows
    
    print(f"✅ Success! Master Dataset created with {len(master_df)} rows.")
    return master_df

if __name__ == "__main__":
    df = load_and_merge()