"""
Scraper pour arXiv API
Couvre: Informatique, Physique, Mathématiques, etc.
Limite: Illimitée (politesse: 1 req/3s)
"""
import os
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
from typing import List, Dict
from src.pdf_utils import extract_text_from_pdf

class ArxivScraper:
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.results_per_page = 100
    
    def search(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Recherche sur arXiv avec remplacement automatique des articles sans PDF
        
        Args:
            query: Requête de recherche
            max_results: Nombre d'articles **avec PDF** souhaités
            
        Returns:
            Liste de max_results dictionnaires avec PDFs téléchargés
        """
        results_with_pdf = []
        start = 0
        max_attempts = max_results * 3  # Limite anti-boucle infinie
        attempts = 0
        
        print(f"[arXiv] Objectif : {max_results} articles avec PDF")
        
        while len(results_with_pdf) < max_results and attempts < max_attempts:
            batch_size = min(self.results_per_page, max_results * 2 - start)
            
            # Encoder la query
            encoded_query = urllib.parse.quote_plus(query)
            url = f"{self.base_url}?search_query=all:{encoded_query}&start={start}&max_results={batch_size}"
            
            print(f"[arXiv] Fetching {start}-{start+batch_size}... ({len(results_with_pdf)}/{max_results} OK)")
            
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
                    print(f"[arXiv] Plus de résultats disponibles")
                    break  # Plus de résultats
                
                # Traiter chaque article
                for entry in entries:
                    if len(results_with_pdf) >= max_results:
                        break  # Objectif atteint
                    
                    attempts += 1
                    
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
                    
                    # Extraire ArXiv ID
                    arxiv_id = None
                    if link:
                        arxiv_id = link.split('/abs/')[-1]
                    
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
                    
                    # Tentative de téléchargement PDF
                    if pdf_link:
                        try:
                            pdf_path, full_text, status, method = self._download_and_extract_pdf(
                                pdf_link,
                                title
                            )
                            
                            if status == "SUCCESS":
                                # Article VALIDE avec PDF
                                results_with_pdf.append({
                                    'title': title,
                                    'authors': authors,
                                    'year': year,
                                    'link': link,
                                    'pdf_link': pdf_link,
                                    'abstract': abstract,
                                    'source': 'arXiv',
                                    'doi': doi,
                                    'arxiv_id': arxiv_id,
                                    'pdf_path': pdf_path,
                                    'full_text': full_text,
                                    'extraction_status': status,
                                    'extraction_method': method
                                })
                                print(f"  [{len(results_with_pdf)}/{max_results}] ✓ {title[:50]}...")
                            else:
                                print(f"  [SKIP] {status}: {title[:50]}...")
                        except Exception as e:
                            print(f"  [SKIP] FAILED: {e}")
                    else:
                        print(f"  [SKIP] No PDF link: {title[:50]}...")
                
                start += batch_size
                
                # Politesse: attendre 3 secondes entre les requêtes
                if len(results_with_pdf) < max_results:
                    time.sleep(3)
                    
            except Exception as e:
                print(f"[arXiv] Error: {e}")
                break
        
        print(f"[arXiv] Terminé : {len(results_with_pdf)}/{max_results} articles avec PDF")
        
        return results_with_pdf
    
    def _download_and_extract_pdf(self, pdf_url, title):
        """Télécharge et extrait le texte d'un PDF arXiv"""
        
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
