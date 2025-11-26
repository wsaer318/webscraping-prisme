# -*- coding: utf-8 -*-
"""
Extraction de concepts clés à partir d'une requête de recherche
Utilise le LLM HuggingFace (Llama-3.2-3B) déjà configuré
"""
import os
import re
from typing import List
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


def extract_concepts(query: str, max_concepts: int = 5) -> List[str]:
    """
    Extrait les concepts clés d'une requête via LLM
    
    Args:
        query: Requête de recherche (ex: "l'effet de la lumière sur la matière")
        max_concepts: Nombre max de concepts à extraire
        
    Returns:
        Liste de concepts (ex: ["lumière", "matière", "effet"])
    """
    
    # Récupérer le token HF
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("⚠️ HF_TOKEN non trouvé - Utilisation extraction simple")
        return _extract_concepts_simple(query)
    
    try:
        # Initialiser le client
        client = InferenceClient(token=hf_token)
        
        # Prompt pour extraction de concepts
        prompt = f"""You are a research assistant analyzing search queries for systematic reviews.

TASK: Extract {max_concepts} key concepts from this research query.

QUERY: "{query}"

RULES:
1. Return ONLY the most important concepts (nouns, domains, entities)
2. Remove stop words (the, of, on, in, etc.)
3. Keep multi-word concepts together (e.g., "machine learning" not ["machine", "learning"])
4. Output format: one concept per line, no numbering, no explanation

EXAMPLE:
Query: "the effect of light on matter in physics"
Output:
light
matter
effect
physics

Now extract concepts from: "{query}" """

        # Appel API
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages=messages,
            model="meta-llama/Llama-3.2-3B-Instruct",
            max_tokens=200,
            temperature=0.2  # Faible pour cohérence
        )
        
        # Parser la réponse
        response_text = response.choices[0].message.content.strip()
        
        # Extraire les concepts (une ligne = un concept)
        concepts = [
            line.strip().lower() 
            for line in response_text.split('\n') 
            if line.strip() and not line.strip().startswith(('Output:', 'Query:', '-', '*'))
        ]
        
        # Nettoyer et limiter
        concepts = [c for c in concepts if len(c) > 2 and len(c) < 50][:max_concepts]
        
        if concepts:
            print(f"✓ Concepts extraits via LLM : {concepts}")
            return concepts
        else:
            # Fallback
            return _extract_concepts_simple(query)
        
    except Exception as e:
        print(f"❌ Erreur LLM : {e}")
        return _extract_concepts_simple(query)


def _extract_concepts_simple(query: str) -> List[str]:
    """Extraction simple de concepts sans LLM (fallback)"""
    
    # Mots vides en français et anglais
    stop_words = {
        'the', 'of', 'in', 'on', 'at', 'to', 'a', 'an', 'and', 'or', 'but', 'is', 'are',
        'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'ou', 'dans', 'sur', 'pour'
    }
    
    # Nettoyer et tokeniser
    query_clean = re.sub(r'[^\w\s]', ' ', query.lower())
    words = query_clean.split()
    
    # Filtrer stop words et mots courts
    concepts = [w for w in words if w not in stop_words and len(w) > 3]
    
    # Dédupliquer tout en gardant l'ordre
    seen = set()
    concepts_unique = []
    for c in concepts:
        if c not in seen:
            seen.add(c)
            concepts_unique.append(c)
    
    print(f"✓ Concepts extraits (simple) : {concepts_unique[:5]}")
    return concepts_unique[:5]


if __name__ == "__main__":
    # Tests
    test_queries = [
        "machine learning in healthcare",
        "l'effet de la lumière sur la matière",
        "climate change impact on agriculture in Africa",
        "CRISPR gene editing ethics"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        concepts = extract_concepts(query)
        print(f"Concepts: {concepts}")
