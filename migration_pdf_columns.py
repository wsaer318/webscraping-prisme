"""
Migration : Ajout colonnes pdf_url et arxiv_id pour synchronisation
"""
import sqlite3

conn = sqlite3.connect('data/prisma.db')
cursor = conn.cursor()

print("Migration : Ajout colonnes manquantes...")

# Colonnes à ajouter
columns_to_add = [
    ("pdf_url", "VARCHAR(500)"),
    ("arxiv_id", "VARCHAR(100)")
]

for col_name, col_type in columns_to_add:
    try:
        cursor.execute(f"ALTER TABLE articles ADD COLUMN {col_name} {col_type}")
        print(f"✓ Colonne {col_name} ajoutée")
    except sqlite3.OperationalError as e:
        if "duplicate column" in str(e).lower():
            print(f"⚠️ Colonne {col_name} existe déjà")
        else:
            print(f"❌ Erreur {col_name}: {e}")

conn.commit()
conn.close()

print("\n✅ Migration terminée ! Rechargez la page Streamlit.")
