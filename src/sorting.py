"""
Module de tri et de scoring des articles inspiré de process_improved.py
"""
import re
import unicodedata
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict

def clean_text(text: str) -> str:
    """Nettoyage basique du texte"""
    if not text:
        return ""
    # Normalisation Unicode
    text = unicodedata.normalize('NFKC', text)
    # Suppression caractères non-imprimables
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    # Espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_relevance_scores(articles: List[Dict], query: str) -> List[float]:
    """
    Calcule un score de pertinence pour chaque article par rapport à la requête
    Utilise BM25 (rapide et efficace) au lieu de BERT pour l'interactivité
    """
    if not articles or not query:
        return [0.0] * len(articles)
    
    # Préparation du corpus (Titre + Abstract)
    corpus = []
    for art in articles:
        title = clean_text(art.get('title', ''))
        abstract = clean_text(art.get('abstract', ''))
        # On donne plus de poids au titre
        text = f"{title} {title} {abstract}"
        corpus.append(text.lower().split())
    
    tokenized_query = clean_text(query).lower().split()
    
    # BM25
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenized_query)
    
    # Normalisation [0, 1]
    if scores.max() > 0:
        scores = scores / scores.max()
    
    return scores.tolist()
