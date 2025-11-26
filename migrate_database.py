"""
Migration script to add new columns to existing database.
Adds: full_text, text_extraction_status, extraction_method
"""

import sqlite3
import os

DB_PATH = "data/prisma.db"

if os.path.exists(DB_PATH):
    print(f"Migrating database: {DB_PATH}")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if columns already exist
    cursor.execute("PRAGMA table_info(articles)")
    columns = [col[1] for col in cursor.fetchall()]
    
    print(f"Existing columns: {columns}")
    
    # Add new columns if they don't exist
    if 'full_text' not in columns:
        print("Adding column: full_text")
        cursor.execute("ALTER TABLE articles ADD COLUMN full_text TEXT")
    
    if 'text_extraction_status' not in columns:
        print("Adding column: text_extraction_status")
        cursor.execute("ALTER TABLE articles ADD COLUMN text_extraction_status TEXT DEFAULT 'NOT_ATTEMPTED'")
    
    if 'extraction_method' not in columns:
        print("Adding column: extraction_method")
        cursor.execute("ALTER TABLE articles ADD COLUMN extraction_method TEXT")
    
    conn.commit()
    conn.close()
    
    print("✅ Migration completed successfully!")
else:
    print(f"Database not found at {DB_PATH}. Will be created on first use.")
