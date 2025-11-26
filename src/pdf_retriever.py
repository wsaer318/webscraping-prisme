"""
Récupération automatique de PDFs pour articles sans texte complet
Utilise Unpaywall (gratuit) et ArXiv
"""
import os
import requests
import time
from pathlib import Path
from typing import Optional, Dict
from src.database import get_db, Article, ArticleHistory


class PDFRetriever:
    """Classe pour récupérer automatiquement les PDFs manquants"""
    
    def __init__(self, email: str = "your@email.com"):
        """
        Args:
            email: Email pour Unpaywall API (requis, politesse)
        """
        self.email = email
        self.unpaywall_base = "https://api.unpaywall.org/v2/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (PRISMA Research Tool)'
        })
    
    def find_pdf_url(self, doi: str = None, arxiv_id: str = None) -> Optional[str]:
        """
        Cherche une URL de PDF via Unpaywall ou ArXiv
        
        Args:
            doi: DOI de l'article
            arxiv_id: ID ArXiv
            
        Returns:
            URL du PDF ou None
        """
        # Essai 1: Unpaywall (si DOI disponible)
        if doi:
            pdf_url = self._try_unpaywall(doi)
            if pdf_url:
                return pdf_url
        
        # Essai 2: ArXiv (si ArXiv ID disponible)
        if arxiv_id:
            pdf_url = self._try_arxiv(arxiv_id)
            if pdf_url:
                return pdf_url
        
        return None
    
    def _try_unpaywall(self, doi: str) -> Optional[str]:
        """Tente de trouver un PDF via Unpaywall (Open Access)"""
        try:
            url = f"{self.unpaywall_base}{doi}?email={self.email}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Vérifier si Open Access
                if data.get("is_oa"):
                    # Chercher la meilleure version (publisher > repository)
                    best_oa = data.get("best_oa_location", {})
                    pdf_url = best_oa.get("url_for_pdf")
                    
                    if pdf_url:
                        print(f"  ✓ PDF trouvé via Unpaywall: {doi}")
                        return pdf_url
            
            time.sleep(0.5)  # Rate limiting poli
            return None
            
        except Exception as e:
            print(f"  ⚠️ Erreur Unpaywall pour {doi}: {e}")
            return None
    
    def _try_arxiv(self, arxiv_id: str) -> Optional[str]:
        """Construit l'URL PDF ArXiv"""
        try:
            # Format: http://arxiv.org/pdf/2103.12345.pdf
            clean_id = arxiv_id.replace("arXiv:", "").strip()
            pdf_url = f"http://arxiv.org/pdf/{clean_id}.pdf"
            
            # Vérifier que le PDF existe
            response = requests.head(pdf_url, timeout=5)
            if response.status_code == 200:
                print(f"  ✓ PDF trouvé via ArXiv: {arxiv_id}")
                return pdf_url
            
            return None
            
        except Exception as e:
            print(f"  ⚠️ Erreur ArXiv pour {arxiv_id}: {e}")
            return None
    
    def download_pdf(self, url: str, output_path: Path) -> bool:
        """
        Télécharge un PDF depuis une URL
        
        Args:
            url: URL du PDF
            output_path: Chemin de sauvegarde
            
        Returns:
            True si succès
        """
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Créer le dossier parent si nécessaire
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
            
        except Exception as e:
            print(f"  ❌ Échec téléchargement {url}: {e}")
            return False
    
    def process_article(self, article: Article, base_pdf_dir: Path) -> bool:
        """
        Traite un article pour récupérer son PDF
        
        Args:
            article: Instance Article
            base_pdf_dir: Dossier racine pour PDFs
            
        Returns:
            True si PDF récupéré
        """
        # Vérifier si déjà un PDF
        if article.pdf_path and os.path.exists(article.pdf_path):
            return False
        
        # Chercher URL PDF
        pdf_url = self.find_pdf_url(
            doi=article.doi,
            arxiv_id=article.arxiv_id
        )
        
        if not pdf_url:
            return False
        
        # Construire chemin de sauvegarde
        safe_title = "".join(c for c in article.title[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"{article.id}_{safe_title}.pdf"
        pdf_path = base_pdf_dir / filename
        
        # Télécharger
        if self.download_pdf(pdf_url, pdf_path):
            # Mettre à jour l'article
            article.pdf_path = str(pdf_path)
            article.pdf_url = pdf_url
            return True
        
        return False


def auto_retrieve_missing_pdfs(session_id: int = None, max_articles: int = 50):
    """
    Fonction principale pour récupérer les PDFs manquants en arrière-plan
    
    Args:
        session_id: ID de session (None = tous)
        max_articles: Nombre max d'articles à traiter
    """
    db = next(get_db())
    retriever = PDFRetriever(email="prisma.tool@research.edu")
    
    try:
        # Trouver articles sans PDF
        query = db.query(Article).filter(
            (Article.pdf_path.is_(None)) | (Article.full_text.is_(None))
        )
        
        if session_id:
            query = query.filter(Article.search_session_id == session_id)
        
        articles = query.limit(max_articles).all()
        
        if not articles:
            print("✓ Tous les articles ont déjà un PDF")
            return
        
        print(f"\n🔍 Recherche de PDFs pour {len(articles)} articles...")
        
        base_pdf_dir = Path("data/pdfs_auto")
        success_count = 0
        
        for i, article in enumerate(articles, 1):
            print(f"\n[{i}/{len(articles)}] {article.title[:60]}...")
            
            if retriever.process_article(article, base_pdf_dir):
                # Logger dans ArticleHistory
                history = ArticleHistory(
                    article_id=article.id,
                    previous_status=article.status,
                    new_status=article.status,
                    action="PDF_AUTO_RETRIEVED",
                    reason=f"PDF récupéré automatiquement via {'Unpaywall' if article.doi else 'ArXiv'}",
                    user="System"
                )
                db.add(history)
                db.commit()
                
                success_count += 1
                print(f"  ✓ PDF sauvegardé: {article.pdf_path}")
            
            # Rate limiting
            time.sleep(1)
        
        print(f"\n✅ Terminé: {success_count}/{len(articles)} PDFs récupérés")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    # Test
    auto_retrieve_missing_pdfs(max_articles=10)
