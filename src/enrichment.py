# -*- coding: utf-8 -*-
"""
Module d'enrichissement des métadonnées (Citations, Impact)
Utilise l'API Semantic Scholar
"""
import requests
import time
from typing import List, Dict, Optional
from src.database import get_db, Article

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper"

def get_citations_for_doi(doi: str) -> Optional[int]:
    """
    Récupère le nombre de citations pour un DOI donné.
    Retourne None en cas d'erreur ou si non trouvé.
    """
    if not doi:
        return None
        
    try:
        # Nettoyage du DOI
        clean_doi = doi.strip()
        
        # Appel API
        url = f"{SEMANTIC_SCHOLAR_API_URL}/DOI:{clean_doi}"
        params = {"fields": "citationCount"}
        
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("citationCount", 0)
        elif response.status_code == 404:
            print(f"DOI non trouvé sur Semantic Scholar: {clean_doi}")
            return None
        elif response.status_code == 429:
            print("Rate limit Semantic Scholar atteint. Pause...")
            time.sleep(2) # Petit backoff
            return None
        else:
            print(f"Erreur API S2 ({response.status_code}): {response.text}")
            return None
            
    except Exception as e:
        print(f"Exception lors de la récupération des citations pour {doi}: {e}")
        return None

def get_citations_batch(dois: List[str]) -> Dict[str, int]:
    """
    Récupère les citations pour une liste de DOIs (Batch).
    Note: L'API Batch de S2 est plus complexe (POST), ici on itère simplement pour la version simple.
    Pour une version pro, utiliser l'endpoint /batch.
    """
    results = {}
    for doi in dois:
        count = get_citations_for_doi(doi)
        if count is not None:
            results[doi] = count
        time.sleep(0.5)
    return results

def enrich_session_articles(session_id: int):
    """
    Enrichit tous les articles d'une session avec le nombre de citations.
    Met à jour la base de données.
    Respecte le rate limit: 100 requêtes / 5 minutes (1 req / 3 sec)
    """
    db = next(get_db())
    try:
        articles = db.query(Article).filter(
            Article.search_session_id == session_id
        ).all()
        
        print(f"Enrichissement de {len(articles)} articles pour la session {session_id}...")
        
        updated_count = 0
        for article in articles:
            # Identifier : DOI ou ArXiv ID
            paper_id = None
            if article.doi:
                paper_id = f"DOI:{article.doi}"
            elif article.arxiv_id:
                paper_id = f"ARXIV:{article.arxiv_id}"
            
            if not paper_id:
                continue
                
            # Appel API
            try:
                url = f"{SEMANTIC_SCHOLAR_API_URL}/{paper_id}"
                params = {"fields": "citationCount"}
                response = requests.get(url, params=params, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    count = data.get("citationCount", 0)
                    article.citation_count = count
                    updated_count += 1
                elif response.status_code == 429:
                    time.sleep(2)
            except Exception:
                pass
                
            # Commit intermédiaire tous les 10 articles
            if updated_count % 10 == 0:
                db.commit()
                print(f"  ... {updated_count} articles mis à jour")
            
            # Rate limit global: 100 req / 5 min = 1 req / 3 sec
            time.sleep(3)
        
        db.commit()
        print(f"Terminé ! {updated_count} articles enrichis avec citations.")
        
    except Exception as e:
        print(f"Erreur lors de l'enrichissement : {e}")
    finally:
        db.close()

if __name__ == "__main__":
    # Test manuel
    test_doi = "10.1038/nature14539" # AlphaGo
    print(f"Test citations pour {test_doi}: {get_citations_for_doi(test_doi)}")
