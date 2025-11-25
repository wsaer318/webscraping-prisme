import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import time
import random
from sqlalchemy.orm import Session
from src.database import Article, get_db

class GoogleScholarScraper:
    def __init__(self):
        self.base_url = 'https://scholar.google.com/scholar?q='
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        self.headers = {'User-Agent': self.user_agent}

    def fetch_with_retry(self, req, max_retries=3):
        for attempt in range(max_retries):
            try:
                return urllib.request.urlopen(req)
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait_time = random.uniform(30, 60) * (attempt + 1)
                    print(f"Erreur 429. Attente de {wait_time:.2f}s...")
                    time.sleep(wait_time)
                else:
                    raise e
        raise Exception("Google Scholar bloque les requêtes (Erreur 429). Veuillez attendre quelques minutes.")

    def search(self, query, num_results=10, db: Session = None):
        encoded_search = urllib.parse.quote_plus(query)
        full_url = f"{self.base_url}{encoded_search}&num={num_results}"
        
        req = urllib.request.Request(full_url, headers=self.headers)
        response = self.fetch_with_retry(req)
        
        if not response:
            return []

        soup = BeautifulSoup(response.read(), 'html.parser')
        results = []
        
        for item in soup.find_all('div', class_='gs_r gs_or gs_scl'):
            try:
                title_tag = item.find('h3', class_='gs_rt')
                title = title_tag.text if title_tag else "N/A"
                
                link_tag = item.find('a')
                link = link_tag['href'] if link_tag else None
                
                authors_tag = item.find('div', class_='gs_a')
                authors = authors_tag.text if authors_tag else "N/A"
                
                abstract_tag = item.find('div', class_='gs_rs')
                abstract = abstract_tag.text if abstract_tag else None

                article = {
                    "title": title,
                    "link": link,
                    "authors": authors,
                    "abstract": abstract,
                    "source": "Google Scholar"
                }
                results.append(article)

                # Save to DB if session provided
                if db:
                    # Check for duplicates
                    exists = db.query(Article).filter(Article.title == title).first()
                    if not exists:
                        new_article = Article(
                            title=title,
                            link=link,
                            authors=authors,
                            abstract=abstract,
                            status="IDENTIFIED"
                        )
                        db.add(new_article)
            except Exception as e:
                print(f"Error parsing item: {e}")
        
        if db:
            db.commit()
            
        return results
