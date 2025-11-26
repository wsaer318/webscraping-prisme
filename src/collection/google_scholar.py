import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import time
import random
import os
import ssl
import gzip
from io import BytesIO
from sqlalchemy.orm import Session
from src.database import Article, get_db

class GoogleScholarScraper:
    def __init__(self):
        self.base_url = 'https://scholar.google.com/scholar?q='
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0'
        ]
        # Ignore SSL certificate errors
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

    def get_random_headers(self):
        ua = random.choice(self.user_agents)
        headers = {
            'User-Agent': ua,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': random.choice(['en-US,en;q=0.9', 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7']),
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
        
        # Add Client Hints for Chrome-based browsers
        if 'Chrome' in ua:
            headers.update({
                'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"' if 'Windows' in ua else '"macOS"' if 'Macintosh' in ua else '"Linux"'
            })
            
        return headers

    def fetch_with_retry(self, req, max_retries=3):
        for attempt in range(max_retries):
            try:
                return urllib.request.urlopen(req, context=self.ssl_context)
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait_time = random.uniform(30, 60) * (attempt + 1)
                    print(f"Erreur 429. Attente de {wait_time:.2f}s...")
                    time.sleep(wait_time)
                else:
                    raise e
        raise Exception("Google Scholar bloque les requêtes (Erreur 429). Veuillez attendre quelques minutes.")

    def get_abstract(self, article_url):
        # Function to fetch and parse the article page to extract the abstract
        req = urllib.request.Request(article_url, headers=self.get_random_headers())
        try:
            with urllib.request.urlopen(req, context=self.ssl_context) as response:
                page_content = response.read()
                soup = BeautifulSoup(page_content, 'html.parser')
                abstract_tag = soup.find('div', class_='abstract')  # Example class name
                abstract = abstract_tag.get_text(strip=True) if abstract_tag else None
                return abstract
        except Exception as e:
            print(f"Error fetching abstract: {e}")
            return None

    def get_article(self, article, search_query="default"): 
        # Function to fetch and parse individual article pages if needed
        print(f"\nDownloading article: {article['title']}")
        link = article.get('pdf_link', article['link']) # Use PDF link if available, else fallback
        
        # Nettoyer le titre pour Windows (supprimer : * ? < > | " \ /)
        title = article.get('title')
        invalid_chars = [':', '*', '?', '<', '>', '|', '"', '\\', '/']
        for char in invalid_chars:
            title = title.replace(char, '_')
        
        # Sanitize query for folder name
        safe_query = "".join([c if c.isalnum() else "_" for c in search_query])
        
        pdf_filename = f"{title}_GS.pdf"
        req = urllib.request.Request(link, headers=self.get_random_headers())
        
        try:
            with urllib.request.urlopen(req, context=self.ssl_context) as response:
                # Vérifier si c'est bien un PDF
                content_type = response.info().get('Content-Type', '').lower()
                if 'application/pdf' not in content_type and 'binary/octet-stream' not in content_type:
                    print(f"⚠️ Warning: Content-Type is {content_type}, not PDF. Skipping.")
                    return None, None, "FAILED", "invalid_content_type"

                pdf_content = response.read()
                
                # Vérifier que c'est un vrai PDF (magic bytes)
                if not pdf_content.startswith(b'%PDF'):
                    print(f"⚠️ Warning: Downloaded content is not a PDF (probably HTML/error page). Skipping.")
                    return None
                
                if len(pdf_content) < 1000: # Moins de 1KB = probablement une erreur ou page de login
                    print("⚠️ Warning: PDF too small (<1KB). Likely corrupted or blocked.")
                    return None, None, "FAILED", "file_too_small"
                
                # Create query-specific folder
                save_dir = os.path.join('data/0_raw/pdfs', safe_query)
                os.makedirs(save_dir, exist_ok=True) 
                
                full_file_path = os.path.join(save_dir, pdf_filename)
                with open(full_file_path, 'wb') as f_pdf:
                    f_pdf.write(pdf_content)
                
                # Normaliser le chemin en absolu
                absolute_path = os.path.abspath(full_file_path)
                print(f"✅ Download successful: {absolute_path} ({len(pdf_content)} bytes)")
                
                # Retourner seulement le chemin, l'extraction se fera plus tard
                return absolute_path
        except Exception as e:
            print(f"Error downloading PDF: {e}")
            return None

    def search(self, query, num_results=5, db: Session = None, review_only=True):
        # Add a random delay to mimic human behavior and avoid server errors
        time.sleep(random.uniform(1, 3))
        
        # Créer une session de recherche
        search_session = None
        if db:
            from src.database import SearchSession
            search_session = SearchSession(
                query=query,
                num_results=num_results,
                status='ACTIVE'
            )
            db.add(search_session)
            db.commit()
            db.refresh(search_session)
            print(f"[SESSION] Created search session #{search_session.id} for query: '{query}'")
        
        encoded_search = urllib.parse.quote_plus(query)
        full_url = f"{self.base_url}{encoded_search}&num={num_results}"
        
        # Ajouter le filtre "Articles de revue" si demandé
        if review_only:
            full_url += "&as_rr=1"
        
        print(f"Scraping URL: {full_url}")
        
        req = urllib.request.Request(full_url, headers=self.get_random_headers())
        response = self.fetch_with_retry(req)
        
        if not response:
            return []

        # Lire le contenu brut
        content_bytes = response.read()
        
        # Gérer la compression gzip
        if response.info().get('Content-Encoding') == 'gzip':
            try:
                content_bytes = gzip.decompress(content_bytes)
            except Exception as e:
                print(f"Erreur décompression gzip: {e}")
        
        # Décoder en UTF-8 avec gestion des erreurs
        try:
            content_html = content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # Si le décodage UTF-8 échoue, essayer avec latin-1
            content_html = content_bytes.decode('latin-1', errors='replace')
        
        soup = BeautifulSoup(content_html, 'html.parser')
        results = []
        articles_to_save = []  # Liste temporaire pour sauvegarder en batch
        
        items = soup.find_all('div', class_='gs_r gs_or gs_scl')
        
        if not items:
            # Debug: Save HTML to see what happened
            with open("debug_google_scholar.html", "w", encoding="utf-8") as f:
                f.write(str(soup))
            
            if "robot" in str(soup).lower() or "captcha" in str(soup).lower():
                raise Exception("Google Scholar demande un CAPTCHA. Impossible de scrapper automatiquement pour le moment.")
            elif "erreur de serveur" in str(soup).lower() or "server error" in str(soup).lower():
                raise Exception("Google Scholar a renvoyé une erreur interne (Server Error). Veuillez réessayer plus tard.")
            else:
                print("Aucun élément trouvé. HTML sauvegardé dans debug_google_scholar.html")
        
        # === PHASE 1 : TÉLÉCHARGEMENT ===
        print(f"\n🔽 Phase 1: Téléchargement des PDFs...")
        for item in items:
            try:
                title_tag = item.find('h3', class_='gs_rt')
                title = title_tag.text if title_tag else "N/A"
                
                link_tag = item.find('a')
                link = link_tag['href'] if link_tag else None
                
                authors_tag = item.find('div', class_='gs_a')
                authors = authors_tag.text if authors_tag else "N/A"
                
                # Try to get abstract from snippet first
                abstract_tag = item.find('div', class_='gs_rs')
                abstract = abstract_tag.text if abstract_tag else None

                # Parsing avancé du champ auteurs pour extraire Année et Source
                year = None
                real_source = "Google Scholar"
                
                if authors_tag:
                    auth_text = authors_tag.text
                    import re
                    year_matches = re.findall(r'\b(?:19|20)\d{2}\b', auth_text)
                    if year_matches:
                        year = int(year_matches[-1])
                    
                    parts = auth_text.split(' - ')
                    if len(parts) >= 2:
                        potential_source = parts[1]
                        real_source = re.sub(r',\s*(?:19|20)\d{2}.*', '', potential_source).strip()
                        
                        if not real_source or real_source.isdigit():
                             if len(parts) > 2:
                                 real_source = parts[2].strip()
                             else:
                                 real_source = "Google Scholar"
                        
                        if not real_source:
                            real_source = "Google Scholar"

                # Try to get direct PDF link
                pdf_link = None
                ggsm = item.find('div', class_='gs_or_ggsm')
                if ggsm:
                    pdf_a = ggsm.find('a')
                    if pdf_a and 'pdf' in pdf_a.text.lower():
                        pdf_link = pdf_a['href']
                
                if not pdf_link and link and link.lower().endswith('.pdf'):
                    pdf_link = link

                article_data = {
                    "title": title,
                    "link": link,
                    "authors": authors,
                    "year": year,
                    "abstract": abstract,
                    "source": real_source,
                    "pdf_link": pdf_link
                }
                
                # Téléchargement du PDF (sans extraction)
                pdf_path = None
                if pdf_link:
                    pdf_path = self.get_article(article_data, search_query=query)
                else:
                    print(f"⏭️ Skipping download for {title}: No direct PDF link found.")
                
                # Stocker pour extraction ultérieure
                articles_to_save.append({
                    'data': article_data,
                    'pdf_path': pdf_path
                })
                
            except Exception as e:
                print(f"Error parsing item: {e}")
        
        # === PHASE 2 : EXTRACTION ===
        print(f"\n📄 Phase 2: Extraction du texte des PDFs téléchargés...")
        from src.pdf_utils import extract_text_from_pdf
        
        for article_info in articles_to_save:
            article_data = article_info['data']
            pdf_path = article_info['pdf_path']
            title = article_data['title']
            
            full_text = None
            extraction_status = "NOT_ATTEMPTED"
            extraction_method = None
            
            if pdf_path and os.path.exists(pdf_path):
                print(f"📝 Extracting text from: {os.path.basename(pdf_path)}")
                try:
                    full_text, extraction_status, extraction_method = extract_text_from_pdf(pdf_path)
                    if extraction_status == "SUCCESS":
                        print(f"   ✅ Success: {len(full_text)} characters")
                    else:
                        print(f"   ❌ Failed: {extraction_status}")
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    extraction_status = "FAILED"
                    extraction_method = "error"
            
            # Récupérer l'abstract complet si lien disponible
            if article_data['link']:
                full_abstract = self.get_abstract(article_data['link'])
                if full_abstract:
                    article_data['abstract'] = full_abstract

            results.append(article_data)

            # === PHASE 3 : SAUVE GARDE EN BDD ===
            if db:
                exists = db.query(Article).filter(Article.title == title).first()
                if not exists:
                    new_article = Article(
                        title=title,
                        link=article_data['link'],
                        authors=article_data['authors'],
                        year=article_data['year'],
                        source=article_data['source'],
                        abstract=article_data['abstract'],
                        pdf_path=pdf_path,
                        full_text=full_text,
                        text_extraction_status=extraction_status,
                        extraction_method=extraction_method,
                        status="IDENTIFIED",
                        search_session_id=search_session.id if search_session else None
                    )
                    db.add(new_article)
        
        if db:
            db.commit()
            
            # Mettre à jour les statistiques de la session
            if search_session:
                search_session.num_results = len(articles_to_save)
                search_session.successful_downloads = sum(1 for a in articles_to_save if a['pdf_path'])
                db.commit()
            
            print(f"\n[SESSION] Saved {len(articles_to_save)} articles to database (Session #{search_session.id if search_session else 'N/A'})")
            
        return results
        
        items = soup.find_all('div', class_='gs_r gs_or gs_scl')
        
        if not items:
            # Debug: Save HTML to see what happened
            with open("debug_google_scholar.html", "w", encoding="utf-8") as f:
                f.write(str(soup))
            
            if "robot" in str(soup).lower() or "captcha" in str(soup).lower():
                raise Exception("Google Scholar demande un CAPTCHA. Impossible de scrapper automatiquement pour le moment.")
            elif "erreur de serveur" in str(soup).lower() or "server error" in str(soup).lower():
                raise Exception("Google Scholar a renvoyé une erreur interne (Server Error). Veuillez réessayer plus tard.")
            else:
                print("Aucun élément trouvé. HTML sauvegardé dans debug_google_scholar.html")
        
        for item in items:
            try:
                title_tag = item.find('h3', class_='gs_rt')
                title = title_tag.text if title_tag else "N/A"
                
                link_tag = item.find('a')
                link = link_tag['href'] if link_tag else None
                
                authors_tag = item.find('div', class_='gs_a')
                authors = authors_tag.text if authors_tag else "N/A"
                
                # Try to get abstract from snippet first
                abstract_tag = item.find('div', class_='gs_rs')
                abstract = abstract_tag.text if abstract_tag else None

                # Parsing avancé du champ auteurs pour extraire Année et Source
                # Format typique: "J Schulman, F Wolski, P Dhariwal... - arXiv preprint arXiv ..., 2017 - arxiv.org"
                year = None
                real_source = "Google Scholar"
                
                if authors_tag:
                    auth_text = authors_tag.text
                    # Essayer de trouver une année (4 chiffres)
                    import re
                    # Correction: Utiliser (?:...) pour ne pas capturer le groupe 19/20 séparément, 
                    # ou simplement chercher \d{4} avec validation de plage
                    year_matches = re.findall(r'\b(?:19|20)\d{2}\b', auth_text)
                    if year_matches:
                        year = int(year_matches[-1]) # Prendre la dernière année trouvée
                    
                    # Essayer de trouver la source (souvent après le tiret)
                    # Format: "Auteurs - Source, Année - Site"
                    parts = auth_text.split(' - ')
                    if len(parts) >= 2:
                        # La source est souvent dans la 2ème partie "Journal of..., 2020"
                        # On prend la partie AVANT la dernière virgule si elle contient une année
                        potential_source = parts[1]
                        # Nettoyage plus robuste: enlever l'année et ce qui suit (souvent ", 2023")
                        real_source = re.sub(r',\s*(?:19|20)\d{2}.*', '', potential_source).strip()
                        
                        # Si la source est vide ou ressemble à une année, ignorer
                        if not real_source or real_source.isdigit():
                             # Essayer la 3ème partie si dispo (parfois le site web est la seule source fiable)
                             if len(parts) > 2:
                                 real_source = parts[2].strip()
                             else:
                                 real_source = "Google Scholar"
                        
                        if not real_source:
                            real_source = "Google Scholar"

                # Try to get direct PDF link (usually on the right side)
                pdf_link = None
                ggsm = item.find('div', class_='gs_or_ggsm')
                if ggsm:
                    pdf_a = ggsm.find('a')
                    if pdf_a and 'pdf' in pdf_a.text.lower():
                        pdf_link = pdf_a['href']
                
                # Fallback to main link if no direct PDF link found, but only if it looks like a PDF
                if not pdf_link and link and link.lower().endswith('.pdf'):
                    pdf_link = link

                article_data = {
                    "title": title,
                    "link": link,
                    "authors": authors,
                    "year": year,
                    "abstract": abstract,
                    "source": real_source,
                    "pdf_link": pdf_link # Store the specific PDF link
                }
                
                # ENRICHISSEMENT (Basé sur leonel/main.py)
                # 1. Télécharger le PDF (dans un dossier spécifique à la requête) ET extraire le texte
                # Only attempt download if we have a likely PDF link
                if pdf_link:
                    pdf_result = self.get_article(article_data, search_query=query)
                else:
                    print(f"Skipping download for {title}: No direct PDF link found.")
                    pdf_result = None
                
                if pdf_result:
                    pdf_path, full_text, extraction_status, extraction_method = pdf_result
                    if extraction_status == "FAILED":
                        print(f"⚠️ Extraction failed for {title}. PDF Path: {pdf_path}")
                else:
                    pdf_path, full_text, extraction_status, extraction_method = None, None, "NOT_ATTEMPTED", None
                
                # 2. Récupérer l'abstract complet si lien disponible
                if link:
                    full_abstract = self.get_abstract(link)
                    if full_abstract:
                        article_data['abstract'] = full_abstract

                results.append(article_data)

                # Save to DB if session provided
                if db:
                    # Check for duplicates
                    exists = db.query(Article).filter(Article.title == title).first()
                    if not exists:
                        new_article = Article(
                            title=title,
                            link=link,
                            authors=authors,
                            year=article_data['year'],
                            source=article_data['source'],
                            abstract=article_data['abstract'],
                            pdf_path=pdf_path,
                            full_text=full_text,
                            text_extraction_status=extraction_status,
                            extraction_method=extraction_method,
                            status="IDENTIFIED"
                        )
                        db.add(new_article)
            except Exception as e:
                print(f"Error parsing item: {e}")
        
        if db:
            db.commit()
            
        return results
