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
#---------------------------------------------------

import re
import string

def clean_text(text):
    text = str(text).lower() # Case standardization
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
    text = re.sub(r'\n', '', text) # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text) # Remove words containing numbers
    return text
print("Done 1")
# Apply it to your data
# df['text'] = df['text'].apply(clean_text)