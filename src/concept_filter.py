# -*- coding: utf-8 -*-
"""
Filtrage d'articles basé sur la présence de concepts
Supporte recherche dans titre, abstract et full text (avec chunking)
"""
import re
from typing import List, Dict, Tuple


def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """
    Découpe un texte en chunks de taille fixe
    
    Args:
        text: Texte à découper
        chunk_size: Nombre de mots par chunk
        
    Returns:
        Liste de chunks
    """
    if not text:
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks


def find_concept_in_text(text: str, concept: str) -> Tuple[bool, str]:
    """
    Cherche un concept dans un texte (insensible à la casse)
    
    Args:
        text: Texte à analyser
        concept: Concept à chercher
        
    Returns:
        (trouvé: bool, localisation: str)
    """
    if not text or not concept:
        return False, ""
    
    # Recherche insensible à la casse
    text_lower = text.lower()
    concept_lower = concept.lower()
    
    # Chercher le concept (mot entier ou phrase)
    pattern = r'\b' + re.escape(concept_lower) + r'\b'
    
    if re.search(pattern, text_lower):
        return True, text_lower.find(concept_lower)
    
    return False, ""


def filter_articles_by_concepts(
    articles: List[Dict],
    concepts: List[str],
    mode: str = "AND",
    search_in_fulltext: bool = True
) -> List[Dict]:
    """
    Filtre les articles selon la présence de concepts
    
    Args:
        articles: Liste d'articles (dict avec 'title', 'abstract', 'full_text')
        concepts: Concepts requis
        mode: "AND" (tous) ou "OR" (au moins un)
        search_in_fulltext: Chercher dans le full text (avec chunks)
        
    Returns:
        Articles filtrés avec métadonnées de matching
    """
    
    if not concepts:
        print("⚠️ Aucun concept fourni, pas de filtrage")
        return articles
    
    filtered = []
    
    print(f"\n🔍 Filtrage par concepts (mode {mode}):")
    print(f"   Concepts : {concepts}")
    print(f"   Full text : {'✓' if search_in_fulltext else '✗'}")
    
    for article in articles:
        # Préparer les zones de recherche
        search_zones = {
            'title': article.get('title', ''),
            'abstract': article.get('abstract', '')
        }
        
        if search_in_fulltext and article.get('full_text'):
            # Chunking pour performance
            chunks = chunk_text(article['full_text'], chunk_size=500)
            search_zones['full_text_chunks'] = chunks
        
        # Chercher chaque concept
        matched_concepts = {}
        
        for concept in concepts:
            found = False
            locations = []
            
            # Titre
            if find_concept_in_text(search_zones['title'], concept)[0]:
                found = True
                locations.append('title')
            
            # Abstract
            if find_concept_in_text(search_zones['abstract'], concept)[0]:
                found = True
                locations.append('abstract')
            
            # Full text (chunks)
            if search_in_fulltext and 'full_text_chunks' in search_zones:
                for chunk_idx, chunk in enumerate(search_zones['full_text_chunks']):
                    if find_concept_in_text(chunk, concept)[0]:
                        found = True
                        locations.append(f'fulltext_chunk_{chunk_idx}')
                        break  # Un seul chunk suffit
            
            if found:
                matched_concepts[concept] = locations
        
        # Décision selon mode
        if mode == "AND":
            # TOUS les concepts doivent être présents
            if len(matched_concepts) == len(concepts):
                article['matched_concepts'] = matched_concepts
                article['concept_match_count'] = len(matched_concepts)
                filtered.append(article)
        
        elif mode == "OR":
            # AU MOINS UN concept doit être présent
            if len(matched_concepts) > 0:
                article['matched_concepts'] = matched_concepts
                article['concept_match_count'] = len(matched_concepts)
                filtered.append(article)
    
    print(f"✓ Filtrage terminé : {len(filtered)}/{len(articles)} articles retenus")
    
    return filtered


def get_concept_coverage_stats(articles: List[Dict], concepts: List[str]) -> Dict:
    """
    Analyse la couverture des concepts dans les articles
    
    Returns:
        Statistiques de couverture
    """
    stats = {
        'total_articles': len(articles),
        'concepts': {},
        'avg_concepts_per_article': 0
    }
    
    total_matches = 0
    
    for concept in concepts:
        count = sum(
            1 for art in articles 
            if art.get('matched_concepts', {}).get(concept)
        )
        stats['concepts'][concept] = {
            'count': count,
            'percentage': (count / len(articles) * 100) if articles else 0
        }
    
    # Moyenne
    for art in articles:
        total_matches += len(art.get('matched_concepts', {}))
    
    stats['avg_concepts_per_article'] = total_matches / len(articles) if articles else 0
    
    return stats


if __name__ == "__main__":
    # Test
    test_articles = [
        {
            'title': 'Machine Learning in Healthcare',
            'abstract': 'This study applies machine learning to healthcare data',
            'full_text': 'Introduction... machine learning... healthcare applications...'
        },
        {
            'title': 'Deep Learning for Medical Imaging',
            'abstract': 'Deep learning techniques for medical imaging',
            'full_text': 'We use deep learning for medical diagnosis...'
        },
        {
            'title': 'Healthcare Data Analysis',
            'abstract': 'Analysis of healthcare data using statistics',
            'full_text': 'Statistical methods for healthcare...'
        }
    ]
    
    concepts = ['machine learning', 'healthcare']
    
    print("Test mode AND:")
    filtered_and = filter_articles_by_concepts(test_articles, concepts, mode="AND")
    print(f"Résultats: {len(filtered_and)} articles")
    
    print("\nTest mode OR:")
    filtered_or = filter_articles_by_concepts(test_articles, concepts, mode="OR")
    print(f"Résultats: {len(filtered_or)} articles")
