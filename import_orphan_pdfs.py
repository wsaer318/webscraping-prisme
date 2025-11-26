# -*- coding: utf-8 -*-
"""
Import des PDFs orphelins (dans le dossier mais pas en BDD)
"""
import os
import sys
sys.path.append('.')

from src.database import get_db, Article, SearchSession
from src.pdf_utils import extract_text_from_pdf
from pathlib import Path

db = next(get_db())

# Créer une session pour les imports
session = SearchSession(
    query="Import PDFs orphelins",
    num_results=0,
    successful_downloads=0,
    status='ACTIVE'
)
db.add(session)
db.commit()
db.refresh(session)

# Parcourir tous les PDFs
pdf_dir = Path("data/0_raw/pdfs/arxiv")
pdf_files = list(pdf_dir.glob("*.pdf"))

print(f"📁 {len(pdf_files)} PDFs trouvés dans {pdf_dir}")

# PDFs déjà en BDD
existing_paths = set(a.pdf_path for a in db.query(Article.pdf_path).filter(Article.pdf_path.isnot(None)).all())
print(f"✓ {len(existing_paths)} PDFs déjà en BDD")

# Importer les orphelins
imported = 0
for pdf_path in pdf_files:
    pdf_path_str = str(pdf_path)
    
    if pdf_path_str not in existing_paths:
        try:
            # Extraire titre du nom de fichier
            title = pdf_path.stem.replace('_', ' ')[:200]
            
            # Extraire texte
            full_text, status, method = extract_text_from_pdf(pdf_path_str)
            
            # Créer article
            article = Article(
                title=title,
                source="arXiv",
                pdf_path=pdf_path_str,
                full_text=full_text,
                text_extraction_status=status,
                extraction_method=method,
                status="IDENTIFIED",
                search_session_id=session.id
            )
            
            db.add(article)
            imported += 1
            
            if imported % 50 == 0:
                db.commit()
                print(f"  ... {imported} importés")
                
        except Exception as e:
            print(f"❌ Erreur {pdf_path.name}: {e}")

db.commit()
session.num_results = imported
db.commit()

print(f"\n✅ Import terminé : {imported} PDFs orphelins ajoutés à la BDD")
print(f"📊 Total articles en BDD : {db.query(Article).count()}")

db.close()
