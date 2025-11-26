import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from scipy.stats import rankdata
import torch
import os

import json
from src.database import get_db, Article

class AdvancedRanker:
    def __init__(self, model_name="paraphrase-MiniLM-L3-v2"):
        """
        Initialise le ranker avancé.
        Modèle léger par défaut : paraphrase-MiniLM-L3-v2 (rapide et efficace).
        """
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            # Chargement du modèle (peut prendre un peu de temps)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Loading SentenceTransformer model: {self.model_name} on {device}...")
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def preprocess_text(self, text):
        if not text:
            return ""
        return text.lower().strip()

    def chunk_text(self, text, chunk_size=400, stride=200, max_chunks=20):
        """
        Découpe le texte en morceaux (chunks) avec chevauchement.
        max_chunks: Limite le nombre de morceaux pour éviter de saturer le CPU sur des thèses de 500 pages.
        """
        if not text:
            return []
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
            
        chunks = []
        for i in range(0, len(words), stride):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
            if len(chunks) >= max_chunks:
                break
        return chunks

    def compute_scores(self, articles: list, query: str) -> dict:
        """
        Calcule les scores avec Chunking sur le texte complet.
        Stratégie : Max Pooling sur les chunks (on garde le score du meilleur passage).
        """
        if not articles or not query:
            return {}

        final_scores = {}
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        # Move query to CPU numpy for cosine_similarity
        query_vec = query_embedding.cpu().numpy().reshape(1, -1)

        for art in articles:
            # 1. Construction du texte complet
            title = self.preprocess_text(art.get('title', ''))
            abstract = self.preprocess_text(art.get('abstract', ''))
            full_text = self.preprocess_text(art.get('full_text', ''))
            
            # On combine tout, le titre est répété pour l'importance
            combined_text = f"{title} {title} {abstract} {full_text}".strip()
            
            # 2. Chunking
            chunks = self.chunk_text(combined_text)
            if not chunks:
                final_scores[art['id']] = 0.0
                continue
                
            # 3. Encodage des chunks
            # Note: Pour beaucoup d'articles, cela peut être lent. 
            # On pourrait batcher, mais ici on fait simple article par article.
            chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)
            chunk_vecs = chunk_embeddings.cpu().numpy()
            
            # 4. Similarité Sémantique (Max Pooling)
            # On cherche si AU MOINS UN passage est pertinent
            sims = cosine_similarity(query_vec, chunk_vecs)[0]
            semantic_score = float(np.max(sims))
            
            # 5. Score BM25 (Sur Titre + Abstract uniquement pour éviter le bruit du full text sur les mots clés)
            # Ou sur le full text ? BM25 sur full text peut être bruyant. 
            # Gardons BM25 sur le résumé pour la précision "Keyword", et Embeddings sur Full Text pour le "Fond".
            bm25_text = f"{title} {abstract}"
            tokenized_doc = bm25_text.split()
            tokenized_query = self.preprocess_text(query).split()
            
            # Petit hack BM25 local pour éviter de re-créer le corpus entier à chaque fois
            # Pour être rigoureux il faudrait le corpus entier, mais ici on fait un score "local" ou on garde l'approche globale ?
            # L'approche globale est mieux pour IDF.
            # Revenons à une approche hybride simplifiée : 
            # On ne calcule pas BM25 ici article par article, c'est inefficace.
            # On va juste utiliser le score sémantique pur boosté par le chunking, 
            # car c'est ce que l'utilisateur veut ("méthodes avancées").
            # Si on veut garder BM25, il faut le faire sur tout le corpus avant la boucle.
            
            final_scores[art['id']] = semantic_score

        # Réintégration BM25 Global (Optionnel mais recommandé)
        # Pour l'instant, on renvoie le score sémantique pur du meilleur chunk
        # C'est souvent suffisant et plus "Intelligent" que BM25.
        
        return final_scores

    def suggest_threshold(self, scores: list) -> dict:
        """
        Utilise des statistiques robustes (GMM) pour suggérer un seuil de coupure.
        Inspiré de process_improved.py (gmm_bayes_cut_robust).
        """
        if not scores or len(scores) < 5:
            return {"threshold": 0.5, "method": "default (not enough data)"}
        
        scores_array = np.array(scores).reshape(-1, 1)
        
        try:
            # Essayer de fitter un GMM à 2 composants (Pertinent vs Non-pertinent)
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(scores_array)
            
            means = gmm.means_.flatten()
            # On suppose que le cluster avec la moyenne la plus haute est "Pertinent"
            # Le seuil peut être approximé par la moyenne des deux moyennes
            threshold = np.mean(means)
            
            # Raffinement : intersection des gaussiennes (simplifié ici)
            # On s'assure que le seuil est entre min et max
            threshold = np.clip(threshold, min(scores), max(scores))
            
            return {
                "threshold": float(threshold),
                "method": "Gaussian Mixture Model (Advanced)",
                "confidence": "High"
            }
        except Exception as e:
            print(f"GMM failed: {e}")
            return {"threshold": np.median(scores), "method": "Median (Fallback)"}

    def process_batch_and_update_db(self, article_ids: list, query: str):
        """
        Méthode pour le traitement en arrière-plan.
        Calcule les scores et met à jour la base de données directement.
        """
        db = next(get_db())
        try:
            articles = db.query(Article).filter(Article.id.in_(article_ids)).all()
            if not articles:
                return
            
            # Préparation des données
            articles_data = [
                {
                    "id": a.id, 
                    "title": a.title, 
                    "abstract": a.abstract,
                    "full_text": a.full_text
                } 
                for a in articles
            ]
            
            # Calcul des scores
            scores_dict = self.compute_scores(articles_data, query)
            
            # Calcul du seuil global pour ce lot (ou globalement ?)
            # Pour l'instant on stocke juste le score brut.
            # Le seuil sera recalculé dynamiquement dans l'UI sur l'ensemble des articles.
            
            # Mise à jour DB
            for article in articles:
                if article.id in scores_dict:
                    article.relevance_score = scores_dict[article.id]
                    # On pourrait stocker plus de détails dans ia_metadata si besoin
                    article.ia_metadata = json.dumps({"model": self.model_name, "processed_at": str(np.datetime64('now'))})
            
            db.commit()
            print(f"Updated scores for {len(articles)} articles.")
            
        except Exception as e:
            print(f"Error in background processing: {e}")
        finally:
            db.close()
