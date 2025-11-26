# -*- coding: utf-8 -*-
"""
Génération de critères d'exclusion via LLM (Hugging Face API)
Architecture: Critères universels + Critères contextuels générés par IA
"""
import json
import os
import re
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


def generate_exclusion_criteria(query: str) -> list:
    """
    Génère des critères d'exclusion pour une requête PRISMA.
    Combine critères universels + critères contextuels générés par IA.
    
    Args:
        query: La requête de recherche (ex: "machine learning")
        
    Returns:
        Liste de dictionnaires {label, description}
    """
    
    # Critères de base UNIVERSELS (toujours présents)
    base_criteria = [
        {
            "label": "Langue",
            "description": "Full text not available in English or French"
        },
        {
            "label": "Type publication",
            "description": "Editorial, opinion piece, book chapter, conference abstract, or non-peer-reviewed content"
        }
    ]
    
    # Récupérer le token HF
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("⚠️ HF_TOKEN non trouvé dans .env - Utilisation de critères par défaut")
        return base_criteria + _get_contextual_defaults(query)
    
    try:
        # Initialiser le client Hugging Face
        client = InferenceClient(token=hf_token)
        
        # Prompt LIBRE pour générer des critères contextuels
        prompt = f"""You are a research assistant analyzing a PRISMA systematic review query.

QUERY: "{query}"

Task: Generate 2-3 CONTEXTUAL exclusion criteria specific to this research topic.
Do NOT generate generic criteria (language, publication type - already handled).

Focus on:
- Topic-specific scope boundaries (what IS and ISN'T in scope)
- Domain-specific methodological requirements  
- Population/setting/context relevance (if applicable)

Be creative and analyze the query deeply. Think about what makes an article truly relevant vs off-topic for THIS specific research.

Output format (JSON only, no markdown):
[
  {{"label": "French label (2-4 words)", "description": "Detailed English description for AI classification"}},
  {{"label": "...", "description": "..."}},
  {{"label": "...", "description": "..."}}
]

Example for "machine learning healthcare":
[
  {{"label": "Pas d'application ML", "description": "Article discusses healthcare but does not apply or develop machine learning methods"}},
  {{"label": "ML théorique uniquement", "description": "Pure theoretical ML without healthcare application or validation"}},
  {{"label": "Données non-santé", "description": "Machine learning applied to non-healthcare domains (finance, robotics, etc.)"}}
]

Now generate for: "{query}" """

        print(f"🤖 Génération de critères contextuels pour : '{query}'...")
        
        # Appel API
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages=messages,
            model="meta-llama/Llama-3.2-3B-Instruct",
            max_tokens=500,
            temperature=0.4  # Un peu de créativité
        )
        
        response_text = response.choices[0].message.content
        
        # Parser avec robustesse
        contextual_criteria = _parse_llm_response(response_text)
        
        # Combiner base + contextuels
        all_criteria = base_criteria + contextual_criteria
        print(f"✓ {len(all_criteria)} critères générés ({len(base_criteria)} base + {len(contextual_criteria)} contextuels)")
        
        return all_criteria
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération LLM : {e}")
        print(f"→ Utilisation de critères par défaut")
        return base_criteria + _get_contextual_defaults(query)


def _parse_llm_response(response_text: str) -> list:
    """Parse la réponse LLM avec plusieurs stratégies de nettoyage"""
    
    response_clean = response_text.strip()
    
    # Stratégie 1: Retirer markdown
    if "```json" in response_clean:
        response_clean = response_clean.split("```json")[1].split("```")[0].strip()
    elif "```" in response_clean:
        response_clean = response_clean.split("```")[1].split("```")[0].strip()
    
    # Stratégie 2: Extraire le JSON
    json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_clean, re.DOTALL)
    if json_match:
        response_clean = json_match.group(0)
    
    try:
        criteria = json.loads(response_clean)
    except json.JSONDecodeError as e:
        # Nettoyer les guillemets problématiques
        response_clean = response_clean.replace('""', '"')
        try:
            criteria = json.loads(response_clean)
        except:
            raise ValueError(f"Impossible de parser: {e}\nContenu: {response_clean[:200]}")
    
    # Valider
    if not isinstance(criteria, list):
        raise ValueError("Réponse LLM n'est pas une liste")
    
    for c in criteria:
        if not isinstance(c, dict) or "label" not in c or "description" not in c:
            raise ValueError(f"Critère invalide: {c}")
    
    return criteria


def _get_contextual_defaults(query: str) -> list:
    """Critères contextuels par défaut si l'API échoue"""
    return [
        {
            "label": f"Hors sujet",
            "description": f"Article not primarily about {query} or significantly deviates from the core research topic"
        },
        {
            "label": "Revue/Synthèse",
            "description": "Literature review, systematic review, meta-analysis, or synthesis without original empirical data"
        },
        {
            "label": "Méthodologie",
            "description": "Purely theoretical article without empirical validation, experimental results, or data analysis"
        }
    ]


if __name__ == "__main__":
    # Test
    test_query = "machine learning"
    criteria = generate_exclusion_criteria(test_query)
    
    print("\n" + "="*60)
    print(f"Critères générés pour : {test_query}")
    print("="*60)
    for i, c in enumerate(criteria, 1):
        print(f"\n{i}. {c['label']}")
        print(f"   → {c['description']}")
