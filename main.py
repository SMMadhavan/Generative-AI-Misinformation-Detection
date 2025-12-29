import pandas as pd
import os
import re
import string

def load_and_merge():
    # 1. Load ISOT (True=1, Fake=0)
    true_df = pd.read_csv('data/True.csv')
    fake_df = pd.read_csv('data/Fake.csv')
    true_df['label'] = 1
    fake_df['label'] = 0
    isot = pd.concat([true_df, fake_df])[['text', 'label']]

    # 2. Load WELFake
    welfake = pd.read_csv('data/WELFake_Dataset.csv')
    welfake = welfake.rename(columns={'label': 'label'})[['text', 'label']]

    # 3. Load GenAI
    genai = pd.read_csv('data/generative_ai_misinformation_dataset.csv')
    genai = genai.rename(columns={'is_misinformation': 'label'})
    genai['label'] = genai['label'].apply(lambda x: 0 if x == 1 else 1) 
    genai = genai[['text', 'label']]

    # 4. Final Merge
    master_df = pd.concat([isot, welfake, genai], ignore_index=True)
    master_df = master_df.dropna()
    print(f"✅ Success! Master Dataset created with {len(master_df)} rows.")
    return master_df

def clean_text(text):
    text = str(text).lower() 
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) 
    text = re.sub(r'\n', '', text) 
    text = re.sub(r'\w*\d\w*', '', text) 
    return text

# --- STEP 2: RUN THE BOSS (Execution) AT THE VERY BOTTOM ---

if __name__ == "__main__":
    # 1. Run your merge function
    df = load_and_merge()

    # 2. Clean the text (This takes a few minutes for 117k rows!)
    print("🧹 Cleaning 117,493 rows of text... please wait.")
    df['text'] = df['text'].apply(clean_text)

    # 3. Create a 'processed' folder inside 'data'
    os.makedirs('data/processed', exist_ok=True)

    # 4. Save your final 'Master' file
    output_path = 'data/processed/master_cleaned.csv'
    df.to_csv(output_path, index=False)

    print(f"💾 DONE! Your clean dataset is saved at: {output_path}")

    #---------------------------------------------------------
    # 5. Exploratory Data Analysis (EDA) - Class Distribution
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("📊 Generating Class Distribution Chart...")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=df, palette='viridis')
    plt.title('Distribution of Real (1) vs. Fake (0) News')
    plt.xlabel('Label (0: Fake, 1: Real)')
    plt.ylabel('Number of Articles')
    
    # Save the chart to your dashboard folder
    os.makedirs('dashboard', exist_ok=True)
    plt.savefig('dashboard/class_distribution.png')
    print("📈 Chart saved to dashboard/class_distribution.png")
    plt.show()

#-----------------------------------------

# ... (Your previous code for cleaning and saving CSV)

    # --- UPDATED WORD CLOUD SECTION (Memory Optimized) ---
    from wordcloud import WordCloud
    from collections import Counter

    print("☁️ Generating Word Clouds (Optimized for Memory)...")
    
    # 1. Take a safe sample (5,000 articles is plenty for a visual)
    fake_sample = df[df['label'] == 0]['text'].sample(n=5000, random_state=42).astype(str)
    real_sample = df[df['label'] == 1]['text'].sample(n=5000, random_state=42).astype(str)

    # 2. Function to generate cloud safely
    def save_cloud(text_series, title, filename):
        full_text = " ".join(text_series)
        wc = WordCloud(width=800, height=400, background_color='white', max_words=50).generate(full_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.savefig(f'dashboard/{filename}')
        plt.close() # Frees memory
        del full_text # Erases the big string

    save_cloud(fake_sample, 'Common Words in Fake News', 'wordcloud_fake.png')
    save_cloud(real_sample, 'Common Words in Real News', 'wordcloud_real.png')

    print("✨ Word clouds saved successfully!")

    # --- KEEP YOUR SPLITTING CODE BELOW THIS ---
    from sklearn.model_selection import train_test_split

    print("✂️ Splitting data into Training (80%) and Testing (20%) sets...")
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"✅ Training samples: {len(X_train)}")
    print(f"✅ Testing samples: {len(X_test)}")