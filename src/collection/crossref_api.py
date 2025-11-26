"""
Scraper pour Crossref API
Couvre: Toutes disciplines (métadonnées bibliographiques)
Limite: 50 req/s (très généreux)
"""
import urllib.request
import urllib.parse
import json
import time
from typing import List, Dict

class CrossrefScraper:
    def __init__(self):
        self.base_url = "https://api.crossref.org/works"
        self.results_per_page = 100
    
    def search(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Recherche sur Crossref
        
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
            
            encoded_query = urllib.parse.quote_plus(query)
            # Filtrer pour n'avoir que les articles de journaux (peer-reviewed)
            url = f"{self.base_url}?query={encoded_query}&filter=type:journal-article&rows={batch_size}&offset={start}"
            
            print(f"[Crossref] Fetching {start}-{start+batch_size}...")
            
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=15) as response:
                    data = json.loads(response.read())
                
                items = data.get('message', {}).get('items', [])
                
                if not items:
                    break  # Plus de résultats
                
                for item in items:
                    title_list = item.get('title', [])
                    title = title_list[0] if title_list else "N/A"
                    
                    authors_list = item.get('author', [])
                    authors = ', '.join([
                        f"{a.get('given', '')} {a.get('family', '')}".strip()
                        for a in authors_list
                    ]) if authors_list else "N/A"
                    
                    year = None
                    published = item.get('published', {}) or item.get('created', {})
                    if published and 'date-parts' in published:
                        date_parts = published['date-parts'][0]
                        if date_parts:
                            year = date_parts[0]
                    
                    link = item.get('URL') or item.get('resource', {}).get('primary', {}).get('URL')
                    
                    # Crossref fournit rarement des liens PDF directs
                    pdf_link = None
                    for link_obj in item.get('link', []):
                        if 'application/pdf' in link_obj.get('content-type', ''):
                            pdf_link = link_obj.get('URL')
                            break
                    
                    abstract = item.get('abstract')
                    
                    # Source (journal/conférence)
                    container_title = item.get('container-title', [])
                    source = container_title[0] if container_title else "Crossref"
                    
                    # DOI - Crossref fournit toujours le DOI
                    doi = item.get('DOI')
                    
                    results.append({
                        'title': title,
                        'authors': authors,
                        'year': year,
                        'link': link,
                        'pdf_link': pdf_link,
                        'abstract': abstract,
                        'source': source,
                        'doi': doi
                    })
                
                start += batch_size
                
                # Pas de limite stricte, mais on reste poli
                time.sleep(0.1)
                    
            except Exception as e:
                print(f"[Crossref] Error: {e}")
                break
        
        print(f"[Crossref] Total retrieved: {len(results)} articles")
        return results
