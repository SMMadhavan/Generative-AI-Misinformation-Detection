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

    from wordcloud import WordCloud

    print("☁️ Generating Word Clouds (using a safe sample to avoid MemoryError)...")
    
    # We take a sample of 10,000 rows for each to save memory
    fake_sample = df[df['label'] == 0].sample(n=min(10000, len(df[df['label'] == 0])))
    real_sample = df[df['label'] == 1].sample(n=min(10000, len(df[df['label'] == 1])))

    # Join the sampled text
    fake_text = " ".join(fake_sample['text'].astype(str))
    real_text = " ".join(real_sample['text'].astype(str))

    # Create and save Fake News Word Cloud
    wordcloud_fake = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(fake_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_fake, interpolation='bilinear')
    plt.axis('off')
    plt.title('Common Words in Fake News (Sampled)')
    plt.savefig('dashboard/wordcloud_fake.png')

    # Create and save Real News Word Cloud
    wordcloud_real = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(real_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_real, interpolation='bilinear')
    plt.axis('off')
    plt.title('Common Words in Real News (Sampled)')
    plt.savefig('dashboard/wordcloud_real.png')
    
    # Clean up memory immediately
    del fake_text, real_text, fake_sample, real_sample
    
    print("✨ Word clouds saved! Check your 'dashboard' folder.")
    plt.show()