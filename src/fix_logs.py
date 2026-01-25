import sqlite3

# Connect to your NEW database (v2)
DB_PATH = 'neural_db_v2.sqlite'

print(f"🔧 Fixing database: {DB_PATH}...")

try:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Create the missing table
    c.execute('''CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
                    input_text TEXT, 
                    domain TEXT, 
                    verdict TEXT, 
                    confidence REAL
                )''')
    
    conn.commit()
    conn.close()
    print("✅ SUCCESS: 'audit_logs' table created successfully.")

except Exception as e:
    print(f"❌ Error: {e}")