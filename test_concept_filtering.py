# -*- coding: utf-8 -*-
"""
Test manuel du filtrage multi-concepts
"""
import sys
sys.path.append('.')

from src.database import get_db, Article
from src.concept_filter import filter_articles_by_concepts

# Récupérer quelques articles
db = next(get_db())
articles = db.query(Article).filter(Article.full_text.isnot(None)).limit(20).all()

print(f"📊 Test sur {len(articles)} articles")
print("=" * 60)

# Convertir en format dict pour le filtre
articles_dict = []
for a in articles:
    articles_dict.append({
        'title': a.title,
        'abstract': a.abstract or '',
        'full_text': a.full_text or ''
    })

# Test 1 : Les 5 concepts (mode AND)
print("\n🧪 TEST 1 : 5 concepts (mode AND)")
concepts_5 = ['molecular structure', 'dna sequence', 'statistical learning', 'inference', 'models']
print(f"Concepts : {concepts_5}")

filtered_5 = filter_articles_by_concepts(articles_dict, concepts_5, mode="AND", search_in_fulltext=True)
print(f"Résultat : {len(filtered_5)}/{len(articles_dict)} articles")

if filtered_5:
    print("\nArticles retenus :")
    for art in filtered_5:
        print(f"  - {art['title'][:80]}")
        print(f"    Concepts trouvés : {list(art['matched_concepts'].keys())}")

# Test 2 : 2 concepts (mode AND)
print("\n" + "=" * 60)
print("🧪 TEST 2 : 2 concepts (mode AND)")
concepts_2 = ['molecular', 'learning']
print(f"Concepts : {concepts_2}")

filtered_2 = filter_articles_by_concepts(articles_dict, concepts_2, mode="AND", search_in_fulltext=True)
print(f"Résultat : {len(filtered_2)}/{len(articles_dict)} articles")

if filtered_2:
    print("\nArticles retenus :")
    for art in filtered_2[:5]:  # Limiter à 5
        print(f"  - {art['title'][:80]}")

# Test 3 : Mode OR avec 5 concepts
print("\n" + "=" * 60)
print("🧪 TEST 3 : 5 concepts (mode OR)")
print(f"Concepts : {concepts_5}")

filtered_or = filter_articles_by_concepts(articles_dict, concepts_5, mode="OR", search_in_fulltext=True)
print(f"Résultat : {len(filtered_or)}/{len(articles_dict)} articles")

# Test 4 : Vérification manuelle d'un article
print("\n" + "=" * 60)
print("🔍 ANALYSE DÉTAILLÉE d'un article")
if articles_dict:
    test_article = articles_dict[0]
    print(f"Titre : {test_article['title'][:100]}")
    
    # Chercher chaque concept
    for concept in concepts_5:
        in_title = concept.lower() in test_article['title'].lower()
        in_abstract = concept.lower() in test_article['abstract'].lower() if test_article['abstract'] else False
        in_fulltext = concept.lower() in test_article['full_text'].lower()[:5000] if test_article['full_text'] else False
        
        found = in_title or in_abstract or in_fulltext
        symbol = "✓" if found else "✗"
        
        location = []
        if in_title:
            location.append("titre")
        if in_abstract:
            location.append("abstract")
        if in_fulltext:
            location.append("fulltext")
        
        print(f"  {symbol} '{concept}' : {', '.join(location) if location else 'NON TROUVÉ'}")

db.close()

print("\n" + "=" * 60)
print("CONCLUSION :")
print(f"- Mode AND avec 5 concepts trop strict → {len(filtered_5)} retenus")
print(f"- Mode AND avec 2 concepts → {len(filtered_2)} retenus")
print(f"- Mode OR avec 5 concepts → {len(filtered_or)} retenus")
print("\nRecommandation : Utiliser 2-3 concepts MAX en mode AND, ou passer en mode OR")
