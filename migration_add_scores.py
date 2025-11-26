import sqlite3
import os

DB_PATH = os.path.join("data", "prisma.db")

def migrate():
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if columns exist
        cursor.execute("PRAGMA table_info(articles)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if "relevance_score" not in columns:
            print("Adding relevance_score column...")
            cursor.execute("ALTER TABLE articles ADD COLUMN relevance_score FLOAT")
        else:
            print("relevance_score column already exists.")

        if "ia_metadata" not in columns:
            print("Adding ia_metadata column...")
            cursor.execute("ALTER TABLE articles ADD COLUMN ia_metadata TEXT")
        else:
            print("ia_metadata column already exists.")
            
        conn.commit()
        print("Migration successful.")
    except Exception as e:
        print(f"Migration failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
