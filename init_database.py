"""
Script d'initialisation complète de la base de données PRISMA
Recrée la BDD avec tous les champs nécessaires si elle est supprimée
"""
import os
import sys
sys.path.append('.')

from src.database import Base, engine, init_db

DB_PATH = "data/prisma.db"

def init_complete_database():
    """
    Initialise ou réinitialise complètement la base de données
    avec tous les champs requis
    """
    print("=" * 60)
    print("INITIALISATION COMPLETE DE LA BASE DE DONNEES PRISMA")
    print("=" * 60)
    
    # Vérifier si la BDD existe déjà
    db_exists = os.path.exists(DB_PATH)
    
    if db_exists:
        print(f"\nBase de données existante détectée: {DB_PATH}")
        response = input("Voulez-vous la supprimer et recréer ? (o/n): ")
        
        if response.lower() == 'o':
            print(f"Suppression de {DB_PATH}...")
            os.remove(DB_PATH)
            print("Base de données supprimée.")
        else:
            print("Opération annulée. Mise à jour de la structure...")
    
    # Créer/Mettre à jour la structure
    print("\nCréation de la structure de la base de données...")
    print("\nTables à créer:")
    print("  - search_sessions (sessions de recherche)")
    print("  - articles (articles scientifiques)")
    print("\nChamps de la table 'articles':")
    print("  - id (clé primaire)")
    print("  - source, title, authors, year, link, doi")
    print("  - abstract (résumé)")
    print("  - pdf_path (chemin du PDF)")
    print("  - full_text (texte complet extrait)")
    print("  - text_extraction_status, extraction_method")
    print("  - status, exclusion_reason, notes")
    print("  - search_session_id (lien vers session)")
    print("  - created_at, updated_at")
    
    # Créer toutes les tables
    Base.metadata.create_all(bind=engine)
    
    print("\n✓ Structure de base de données créée avec succès!")
    print(f"📁 Fichier: {os.path.abspath(DB_PATH)}")
    
    # Vérifier les tables créées
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"\n✓ Tables créées: {', '.join([t[0] for t in tables])}")
    
    # Vérifier les colonnes de la table articles
    cursor.execute("PRAGMA table_info(articles)")
    columns = cursor.fetchall()
    
    print(f"\n✓ Colonnes de 'articles' ({len(columns)} colonnes):")
    for col in columns:
        col_id, name, type_, notnull, default, pk = col
        print(f"    {name}: {type_}")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("INITIALISATION TERMINEE AVEC SUCCES")
    print("=" * 60)

if __name__ == "__main__":
    init_complete_database()
