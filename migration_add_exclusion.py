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
        # 1. Add suggested_reason column to articles
        cursor.execute("PRAGMA table_info(articles)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if "suggested_reason" not in columns:
            print("Adding suggested_reason column...")
            cursor.execute("ALTER TABLE articles ADD COLUMN suggested_reason TEXT")
        else:
            print("suggested_reason column already exists.")

        # 2. Create exclusion_criteria table
        print("Creating exclusion_criteria table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exclusion_criteria (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label VARCHAR NOT NULL,
                description VARCHAR NOT NULL,
                active INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 3. Add default criteria if empty
        cursor.execute("SELECT count(*) FROM exclusion_criteria")
        if cursor.fetchone()[0] == 0:
            print("Adding default exclusion criteria...")
            defaults = [
                ("Animal Study", "This study is conducted on animals (rats, mice, in vivo) and not on humans."),
                ("Review Article", "This is a literature review, systematic review, or meta-analysis, not an original research article."),
                ("Wrong Language", "The full text is not in English."),
                ("Conference Abstract", "This is only a conference abstract or poster without full text.")
            ]
            cursor.executemany("INSERT INTO exclusion_criteria (label, description) VALUES (?, ?)", defaults)
            
        conn.commit()
        print("Migration successful.")
    except Exception as e:
        print(f"Migration failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
