import pandas as pd
import os

csv_path = 'data/processed/master_cleaned.csv'
db_path = 'neural_db.sqlite'

try:
    print(f"🧹 Reading {csv_path} with BOM support...")
    
    # 1. Read with 'utf-8-sig' to remove invisible characters like \ufeff
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # 2. Force clean all whitespace from column headers
    df.columns = df.columns.str.strip()

    # 3. Rename 'Label' to 'label' if needed (handling Capitalization)
    df.rename(columns={'Label': 'label', 'TARGET': 'label', 'Class': 'label'}, inplace=True)

    print(f"👀 Columns detected: {df.columns.tolist()}")

    if 'label' in df.columns:
        # 4. Save back as standard utf-8 (No BOM)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print("✅ Success! CSV saved cleanly.")
        
        # 5. Delete the old database so it rebuilds fresh
        if os.path.exists(db_path):
            os.remove(db_path)
            print("🗑️  Old database deleted. It will rebuild on next run.")
    else:
        print("❌ ERROR: Still cannot find 'label' column. Please check the CSV content manually.")

except Exception as e:
    print(f"❌ Critical Error: {e}")