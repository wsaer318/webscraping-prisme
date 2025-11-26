"""
Script d'initialisation complète de la base de données PRISMA
Reflète l'état actuel avec toutes les tables et colonnes
"""
import os
import sys
sys.path.append('.')

from src.database import Base, engine, init_db, ExclusionCriteria, EligibilityCriteria, get_db

DB_PATH = "data/prisma.db"

def init_complete_database():
    """
    Initialise ou réinitialise complètement la base de données
    avec tous les champs requis et les données par défaut
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
            try:
                os.remove(DB_PATH)
                print("Base de données supprimée.")
            except PermissionError:
                print("ERREUR: Impossible de supprimer la base de données. Fermez l'application d'abord.")
                return
        else:
            print("Opération annulée. Mise à jour de la structure...")
    
    # Créer/Mettre à jour la structure
    print("\nCréation de la structure de la base de données...")
    print("\nTables gérées:")
    print("  - search_sessions (sessions de recherche)")
    print("  - articles (articles scientifiques avec colonnes Phase 1-3)")
    print("  - article_history (historique des décisions)")
    print("  - exclusion_criteria (critères IA pour Screening)")
    print("  - eligibility_criteria (critères Phase Éligibilité)")
    
    # Créer toutes les tables
    Base.metadata.create_all(bind=engine)
    
    print("\n✓ Structure de base de données créée/mise à jour avec succès!")
    
    # Peupler les données par défaut
    print("\nVérification des données par défaut...")
    db = next(get_db())
    try:
        # Critères d'exclusion (Screening - Phase 2)
        exclusion_count = db.query(ExclusionCriteria).count()
        if exclusion_count == 0:
            print("  > Ajout des critères d'exclusion standards (Screening)...")
            exclusion_defaults = [
                ("Hors sujet", "Article that does not discuss the research topic or is irrelevant."),
                ("Mauvaise population", "Study conducted on animals (rats, mice) or incorrect target population (not humans)."),
                ("Mauvaise intervention", "The intervention or method studied is not the one of interest."),
                ("Mauvais type d'étude", "Literature review, editorial, conference abstract, or book chapter without empirical data."),
                ("Langue incorrecte", "The full text is not in English or French."),
                ("Pas de données", "Theoretical article without empirical results or data.")
            ]
            
            for label, desc in exclusion_defaults:
                db.add(ExclusionCriteria(label=label, description=desc, active=1))
            db.commit()
            print(f"  ✓ {len(exclusion_defaults)} critères d'exclusion ajoutés.")
        else:
            print(f"  ✓ {exclusion_count} critères d'exclusion déjà présents.")
        
        # Critères d'éligibilité (Phase 3)
        eligibility_count = db.query(EligibilityCriteria).count()
        if eligibility_count == 0:
            print("  > Ajout des critères d'éligibilité standards (PICO)...")
            eligibility_defaults = [
                ("Population adéquate", "Study conducted on appropriate target population (humans, correct age group, etc.)", "INCLUSION"),
                ("Intervention pertinente", "Study examines the intervention or exposure of interest", "INCLUSION"),
                ("Outcomes mesurés", "Study reports relevant outcomes or measurements", "INCLUSION"),
                ("Méthodologie insuffisante", "Study lacks clear methodology or has major methodological flaws", "EXCLUSION"),
                ("Texte complet indisponible", "Full text not available or inaccessible", "EXCLUSION")
            ]
            
            for label, desc, ctype in eligibility_defaults:
                db.add(EligibilityCriteria(label=label, description=desc, type=ctype, active=1))
            db.commit()
            print(f"  ✓ {len(eligibility_defaults)} critères d'éligibilité ajoutés.")
        else:
            print(f"  ✓ {eligibility_count} critères d'éligibilité déjà présents.")
            
    except Exception as e:
        print(f"Erreur lors du peuplement des données: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()
    
    print("\n" + "=" * 60)
    print("INITIALISATION TERMINEE AVEC SUCCES")
    print("=" * 60)
    print("\nSchema Article inclut maintenant:")
    print("  - Colonnes Phase 1 (Recherche): title, authors, doi, abstract, pdf_path, full_text...")
    print("  - Colonnes Phase 2 (Screening): status, exclusion_reason, notes, relevance_score...")
    print("  - Colonnes Phase 3 (Éligibilité): eligibility_notes, reviewed_at, reviewer")
    print("  - Métadonnées PDFs: pdf_url, arxiv_id")

if __name__ == "__main__":
    init_complete_database()
