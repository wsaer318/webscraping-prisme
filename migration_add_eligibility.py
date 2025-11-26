"""
Migration : Ajout de la table EligibilityCriteria et colonnes pour phase Éligibilité
"""
from src.database import engine, Base, Article
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
import sqlite3

# Connexion directe SQLite pour migration
conn = sqlite3.connect('data/prisma.db')
cursor = conn.cursor()

print("Migration : Ajout table EligibilityCriteria + colonnes Article")

# 1. Créer table EligibilityCriteria si elle n'existe pas
try:
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS eligibility_criteria (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label VARCHAR(255) NOT NULL,
        description TEXT NOT NULL,
        type VARCHAR(20) NOT NULL,
        active INTEGER DEFAULT 1,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    print("✓ Table eligibility_criteria créée")
except Exception as e:
    print(f"⚠️ Table eligibility_criteria : {e}")

# 2. Ajouter colonnes à Article
columns_to_add = [
    ("eligibility_notes", "TEXT"),
    ("reviewed_at", "DATETIME"),
    ("reviewer", "VARCHAR(100)")
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

# 3. Ajouter critères par défaut (PICO template)
default_criteria = [
    ("Population adéquate", "Study conducted on appropriate target population (humans, correct age group, etc.)", "INCLUSION"),
    ("Intervention pertinente", "Study examines the intervention or exposure of interest", "INCLUSION"),
    ("Outcomes mesurés", "Study reports relevant outcomes or measurements", "INCLUSION"),
    ("Méthodologie insuffisante", "Study lacks clear methodology or has major methodological flaws", "EXCLUSION"),
    ("Texte complet indisponible", "Full text not available or inaccessible", "EXCLUSION")
]

try:
    # Vérifier si critères existent déjà
    cursor.execute("SELECT COUNT(*) FROM eligibility_criteria")
    count = cursor.fetchone()[0]
    
    if count == 0:
        for label, desc, ctype in default_criteria:
            cursor.execute("""
            INSERT INTO eligibility_criteria (label, description, type, active)
            VALUES (?, ?, ?, 1)
            """, (label, desc, ctype))
        print(f"✓ {len(default_criteria)} critères par défaut ajoutés")
    else:
        print(f"⚠️ {count} critères existent déjà, pas d'ajout")
        
except Exception as e:
    print(f"❌ Erreur ajout critères: {e}")

conn.commit()
conn.close()

print("\n✅ Migration terminée !")
