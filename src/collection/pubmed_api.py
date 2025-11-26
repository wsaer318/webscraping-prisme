"""
Scraper pour PubMed E-utilities API
Couvre: Médecine, Biologie, Sciences de la vie
Limite: 3 req/s sans clé API
"""
import urllib.request
import urllib.parse
import json
import time
from typing import List, Dict

class PubMedScraper:
    def __init__(self):
        self.search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        self.results_per_page = 100
    
    def search(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Recherche sur PubMed
        
        Args:
            query: Requête de recherche
            max_results: Nombre maximum de résultats
            
        Returns:
            Liste de dictionnaires contenant les métadonnées
        """
        results = []
        start = 0
        
        while start < max_results:
            batch_size = min(self.results_per_page, max_results - start)
            
            # Étape 1: Recherche pour obtenir les IDs
            encoded_query = urllib.parse.quote_plus(query)
            # Filtrer pour articles de journaux : ajouter [pt] pour publication type
            search_request = f"{self.search_url}?db=pubmed&term={encoded_query}+AND+journal+article[pt]&retstart={start}&retmax={batch_size}&retmode=json"
            
            print(f"[PubMed] Fetching {start}-{start+batch_size}...")
            
            try:
                req = urllib.request.Request(search_request)
                with urllib.request.urlopen(req, timeout=15) as response:
                    data = json.loads(response.read())
                
                ids = data.get('esearchresult', {}).get('idlist', [])
                
                if not ids:
                    break  # Plus de résultats
                
                time.sleep(0.35)  # Respect limite 3 req/s
                
                # Étape 2: Récupérer les métadonnées
                ids_str = ','.join(ids)
                fetch_request = f"{self.fetch_url}?db=pubmed&id={ids_str}&retmode=json"
                
                req = urllib.request.Request(fetch_request)
                with urllib.request.urlopen(req, timeout=15) as response:
                    fetch_data = json.loads(response.read())
                
                articles = fetch_data.get('result', {})
                
                for pmid in ids:
                    if pmid in articles:
                        article = articles[pmid]
                        
                        title = article.get('title', 'N/A')
                        
                        authors_list = article.get('authors', [])
                        authors = ', '.join([a.get('name', '') for a in authors_list]) if authors_list else "N/A"
                        
                        year = None
                        pubdate = article.get('pubdate', '')
                        if pubdate:
                            try:
                                year = int(pubdate.split()[0])
                            except:
                                pass
                        
                        # PubMed Central link si disponible
                        link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        
                        # Source - seulement le nom du journal (pas le volume)
                        source_str = article.get('source', 'PubMed')
                        
                        # DOI si disponible
                        doi = None
                        article_ids = article.get('articleids', [])
                        for aid in article_ids:
                            if aid.get('idtype') == 'doi':
                                doi = aid.get('value')
                                break
                        
                        results.append({
                            'title': title,
                            'authors': authors,
                            'year': year,
                            'link': link,
                            'pdf_link': None,  # PubMed ne fournit pas directement les PDFs
                            'abstract': None,
                            'source': source_str,
                            'doi': doi
                        })
                
                start += batch_size
                
                time.sleep(0.35)  # Respect limite
                    
            except Exception as e:
                print(f"[PubMed] Error: {e}")
                break
        
        print(f"[PubMed] Total retrieved: {len(results)} articles")
        return results
