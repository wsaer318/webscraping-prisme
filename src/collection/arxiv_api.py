"""
Scraper pour arXiv API
Couvre: Informatique, Physique, Mathématiques, etc.
Limite: Illimitée (politesse: 1 req/3s)
"""
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
from typing import List, Dict

class ArxivScraper:
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.results_per_page = 100
    
    def search(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Recherche sur arXiv
        
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
            
            # Encoder la query
            encoded_query = urllib.parse.quote_plus(query)
            url = f"{self.base_url}?search_query=all:{encoded_query}&start={start}&max_results={batch_size}"
            
            print(f"[arXiv] Fetching {start}-{start+batch_size}...")
            
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=15) as response:
                    xml_content = response.read().decode('utf-8')
                
                # Parser le XML
                root = ET.fromstring(xml_content)
                
                # Namespace arXiv
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                
                entries = root.findall('atom:entry', ns)
                
                if not entries:
                    break  # Plus de résultats
                
                for entry in entries:
                    title_elem = entry.find('atom:title', ns)
                    title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else "N/A"
                    
                    authors_elems = entry.findall('atom:author/atom:name', ns)
                    authors = ', '.join([a.text for a in authors_elems]) if authors_elems else "N/A"
                    
                    published_elem = entry.find('atom:published', ns)
                    year = None
                    if published_elem is not None:
                        try:
                            year = int(published_elem.text[:4])
                        except:
                            pass
                    
                    link_elem = entry.find('atom:id', ns)
                    link = link_elem.text if link_elem is not None else None
                    
                    # Trouver le lien PDF
                    pdf_link = None
                    for link_tag in entry.findall('atom:link', ns):
                        if link_tag.get('title') == 'pdf':
                            pdf_link = link_tag.get('href')
                            break
                    
                    summary_elem = entry.find('atom:summary', ns)
                    abstract = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else None
                    
                    # Extraire le DOI si disponible
                    doi = None
                    doi_elem = entry.find('arxiv:doi', {'arxiv': 'http://arxiv.org/schemas/atom'})
                    if doi_elem is not None:
                        doi = doi_elem.text
                    
                    results.append({
                        'title': title,
                        'authors': authors,
                        'year': year,
                        'link': link,
                        'pdf_link': pdf_link,
                        'abstract': abstract,
                        'source': 'arXiv',
                        'doi': doi
                    })
                
                start += batch_size
                
                # Politesse: attendre 3 secondes entre les requêtes
                if start < max_results:
                    time.sleep(3)
                    
            except Exception as e:
                print(f"[arXiv] Error: {e}")
                break
        
        print(f"[arXiv] Total retrieved: {len(results)} articles")
        
        # Phase 2: Télécharger et extraire les PDFs
        print(f"[arXiv] Starting PDF download and extraction...")
        for idx, article in enumerate(results):
            if article.get('pdf_link'):
                try:
                    pdf_path, full_text, status, method = self._download_and_extract_pdf(
                        article['pdf_link'],
                        article['title']
                    )
                    article['pdf_path'] = pdf_path
                    article['full_text'] = full_text
                    article['extraction_status'] = status
                    article['extraction_method'] = method
                    print(f"  [{idx+1}/{len(results)}] {status}: {article['title'][:50]}...")
                except Exception as e:
                    print(f"  [{idx+1}/{len(results)}] FAILED: {e}")
                    article['extraction_status'] = "FAILED"
        
        return results
    
    def _download_and_extract_pdf(self, pdf_url, title):
        """Télécharge et extrait le texte d'un PDF arXiv"""
        import urllib.request
        import os
        from src.pdf_utils import extract_text_from_pdf
        
        # Nettoyer le titre pour le nom de fichier
        safe_title = "".join([c if c.isalnum() or c in (' ', '_') else '_' for c in title])[:100]
        
        # Créer le dossier
        save_dir = os.path.join('data', '0_raw', 'pdfs', 'arxiv')
        os.makedirs(save_dir, exist_ok=True)
        
        pdf_filename = f"{safe_title}.pdf"
        pdf_path = os.path.join(save_dir, pdf_filename)
        
        # Télécharger
        req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            pdf_content = response.read()
            
            # Vérifier que c'est un PDF
            if not pdf_content.startswith(b'%PDF'):
                return None, None, "FAILED", "not_pdf"
            
            with open(pdf_path, 'wb') as f:
                f.write(pdf_content)
        
        # Extraire le texte
        full_text, status, method = extract_text_from_pdf(pdf_path)
        
        return pdf_path, full_text, status, method
