import sqlite3
import pandas as pd
import os

# CONFIG
DB_NAME = 'neural_db_v2.sqlite'
CSV_PATH = 'data/processed/master_cleaned.csv'

def build():
    print(f"🔨 STARTING MANUAL BUILD for {DB_NAME}...")
    
    # 1. Connect and Clean Slate
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME) # Delete old file to be 100% sure
        print("🗑️  Deleted old database file.")
        
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # 2. Create Table EXPLICITLY with 'label'
    print("📋 Creating table schema...")
    c.execute('''CREATE TABLE training_dataset (
                    id INTEGER PRIMARY KEY, 
                    text TEXT, 
                    label INTEGER
                )''')
    conn.commit()
    
    # 3. Load CSV
    print(f"📖 Reading CSV from {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
        # Force column names just in case
        df.columns = df.columns.str.strip().str.lower()
        df.rename(columns={'class': 'label', 'target': 'label'}, inplace=True)
        
        if 'label' not in df.columns:
            print(f"❌ ERROR: CSV columns are: {df.columns.tolist()}")
            return
            
        print(f"   - Found {len(df)} records.")
        print(f"   - Columns: {df.columns.tolist()}")
        
        # 4. Insert Data
        print("💾 Inserting data into SQL...")
        df[['text', 'label']].to_sql('training_dataset', conn, if_exists='append', index=False)
        print("✅ Insert Success.")
        
        # 5. VERIFY IMMEDIATELY
        check_df = pd.read_sql("SELECT * FROM training_dataset LIMIT 1", conn)
        print(f"\n🕵️ VERIFICATION CHECK: Columns in DB are: {check_df.columns.tolist()}")
        
        if 'label' in check_df.columns:
            print("🎉 SUCCESS! Database is ready and correct.")
        else:
            print("💀 FAIL: Label is still missing in DB.")
            
    except Exception as e:
        print(f"❌ CRITICAL FAILURE: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    build()