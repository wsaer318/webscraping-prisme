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


    return filtered


def chunk_text(text: str, chunk_size: int = 400, stride: int = 200, max_chunks: int = 30) -> List[str]:
    """
    Découpe le texte en morceaux (chunks) avec chevauchement.
    """
    if not text:
        return []
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
        
    chunks = []
    for i in range(0, len(words), stride):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
        if len(chunks) >= max_chunks:
            break
    return chunks


def filter_articles_semantically(
    articles: List[Dict],
    concepts: List[str],
    threshold: float = 0.6,
    mode: str = "AND"
) -> List[Dict]:
    """
    Filtre les articles sémantiquement (Embeddings) sur le Full Text
    
    Args:
        articles: Liste d'articles
        concepts: Concepts requis
        threshold: Seuil de similarité (0.0 à 1.0)
        mode: "AND" ou "OR"
        
    Returns:
        Articles filtrés
    """
    try:
        from sentence_transformers import SentenceTransformer, util
        import torch
        
        # Chargement du modèle
        model_name = "paraphrase-MiniLM-L3-v2"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading Semantic Model: {model_name} on {device}...")
        model = SentenceTransformer(model_name, device=device)
        
        filtered = []
        print(f"\n🧠 Filtrage Sémantique Full-Text (Seuil {threshold}, Mode {mode}):")
        
        # Encoder les concepts
        concept_embeddings = model.encode(concepts, convert_to_tensor=True)
        
        for article in articles:
            # 1. Préparer le texte (Full Text prioritaire, sinon Abstract)
            full_text = article.get('full_text', '')
            abstract = article.get('abstract', '')
            title = article.get('title', '')
            
            # Combiner intelligemment
            if full_text and len(full_text) > 500:
                # Si full text dispo, on l'utilise avec le titre
                text_source = f"{title}\n\n{full_text}"
            else:
                # Sinon fallback sur abstract
                text_source = f"{title}\n\n{abstract}"
            
            if not text_source.strip():
                continue
                
            # 2. Chunking
            chunks = chunk_text(text_source)
            if not chunks:
                continue
                
            # 3. Encoder tous les chunks de l'article
            # Cela crée une matrice [n_chunks, embedding_dim]
            chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
            
            # 4. Calculer similarité (Chunks vs Concepts)
            # Résultat: [n_chunks, n_concepts]
            # On veut savoir: Pour chaque concept, quel est le MEILLEUR chunk ?
            # Donc on prend le MAX sur l'axe des chunks (dim 0)
            
            # cos_sim(a, b) -> matrice [len(a), len(b)]
            chunk_concept_sims = util.cos_sim(chunk_embeddings, concept_embeddings)
            
            # Max par colonne (pour chaque concept, le meilleur score parmi tous les chunks)
            # values est un tenseur [n_concepts]
            best_scores_per_concept, _ = torch.max(chunk_concept_sims, dim=0)
            
            # 5. Vérifier les seuils
            matched_concepts = {}
            for idx, score in enumerate(best_scores_per_concept):
                score_val = float(score)
                if score_val >= threshold:
                    matched_concepts[concepts[idx]] = score_val
            
            # 6. Décision
            keep = False
            if mode == "AND":
                if len(matched_concepts) == len(concepts):
                    keep = True
            elif mode == "OR":
                if len(matched_concepts) > 0:
                    keep = True
            
            if keep:
                article['matched_concepts'] = matched_concepts
                article['semantic_score'] = float(torch.mean(best_scores_per_concept)) # Score moyen global
                filtered.append(article)
                
        print(f"✓ Filtrage Sémantique terminé : {len(filtered)}/{len(articles)} articles retenus")
        return filtered
        
    except ImportError:
        print("❌ Erreur: sentence_transformers non installé.")
        return []
    except Exception as e:
        print(f"❌ Erreur Filtrage Sémantique: {e}")
        return []


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
