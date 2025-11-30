import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
import torch
import os
import json
from src.database import get_db, Article, ExclusionCriteria, AIAnalysisRun

class AdvancedRanker:
    def __init__(self, model_name="paraphrase-MiniLM-L3-v2", cross_encoder_name="cross-encoder/ms-marco-TinyBERT-L-2-v2"):
        """
        Initialise le ranker avancé.
        Modèle léger par défaut : paraphrase-MiniLM-L3-v2 (rapide et efficace).
        Cross-Encoder : ms-marco-TinyBERT-L-2-v2 (très léger et performant pour la vérification).
        """
        self.model_name = model_name
        self.cross_encoder_name = cross_encoder_name
        self._model = None
        self._cross_encoder = None
    
    @property
    def model(self):
        if self._model is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Loading SentenceTransformer model: {self.model_name} on {device}...")
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Loading CrossEncoder model: {self.cross_encoder_name} on {device}...")
            self._cross_encoder = CrossEncoder(self.cross_encoder_name, device=device)
        return self._cross_encoder

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

    def compute_scores(self, articles: list, query: str, alpha: float = 0.7) -> dict:
        """
        Calcule les scores Hybrides (Sémantique + BM25).
        alpha: Poids du score sémantique (0.0 à 1.0). 1.0 = 100% Sémantique.
        """
        if not articles or not query:
            return {}

        final_scores = {}
        
        # --- 1. Score Sémantique (Bi-Encoder) ---
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        query_vec = query_embedding.cpu().numpy().reshape(1, -1)
        
        semantic_scores = []
        corpus_texts = [] # Pour BM25
        
        for art in articles:
            # Construction du texte
            title = self.preprocess_text(art.get('title', ''))
            abstract = self.preprocess_text(art.get('abstract', ''))
            full_text = self.preprocess_text(art.get('full_text', ''))
            
            # Texte combiné pour chunks
            combined_text = f"{title} {title} {abstract} {full_text}".strip()
            
            # Texte pour BM25 (tokenisé plus tard)
            corpus_texts.append(combined_text)
            
            # Chunking & Embedding
            chunks = self.chunk_text(combined_text)
            if not chunks:
                semantic_scores.append(0.0)
                continue
                
            chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)
            chunk_vecs = chunk_embeddings.cpu().numpy()
            
            # Max Pooling
            sims = cosine_similarity(query_vec, chunk_vecs)[0]
            semantic_scores.append(float(np.max(sims)))
            
        # --- 2. Score Lexical (BM25) ---
        # Tokenisation simple pour BM25
        tokenized_corpus = [doc.split(" ") for doc in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # --- 3. Normalisation & Fusion ---
        def normalize(scores):
            if not scores: return []
            arr = np.array(scores)
            min_val, max_val = arr.min(), arr.max()
            if max_val == min_val: return np.zeros_like(arr)
            return (arr - min_val) / (max_val - min_val)

        norm_semantic = normalize(semantic_scores)
        norm_bm25 = normalize(bm25_scores)
        
        # Fusion pondérée
        for idx, art in enumerate(articles):
            hybrid_score = (alpha * norm_semantic[idx]) + ((1 - alpha) * norm_bm25[idx])
            final_scores[art['id']] = float(hybrid_score)
            
            # Stockage des détails pour debug/analyse
            # On pourrait stocker ça dans ia_metadata si besoin
            # art['debug_scores'] = {'sem': norm_semantic[idx], 'bm25': norm_bm25[idx]}

        return final_scores

    def check_exclusion_criteria(self, article_text: str, criteria: list) -> dict:
        """
        Vérifie si l'article correspond à l'un des critères d'exclusion.
        Retourne un dict avec :
        - 'best_reason': le label du meilleur critère (ou None)
        - 'all_scores': dict {label: score} pour tous les critères
        """
        if not criteria or not article_text:
            return {"best_reason": None, "all_scores": {}}
            
        # On prépare les paires (Texte, Description Critère)
        # On utilise le début du texte (Titre + Abstract) pour la rapidité
        # Le Cross-Encoder est lent sur les longs textes, on tronque à 200 mots
        truncated_text = " ".join(article_text.split()[:200])
        
        pairs = [[truncated_text, c.description] for c in criteria]
        
        # Prédiction (retourne des scores logits ou probabilités selon le modèle)
        scores = self.cross_encoder.predict(pairs)
        
        # Créer un dict avec tous les scores
        all_scores = {criteria[i].label: float(scores[i]) for i in range(len(criteria))}
        
        # Trouver le meilleur score
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        # Seuil empirique pour TinyBERT (souvent autour de 0 ou 1 en logit, ou > 0.5 en proba)
        # UPDATE: Après calibration, les scores sont des logits négatifs (ex: -9 vs -11).
        # On fixe le seuil à -11.0 pour être permissif et suggérer des raisons même avec confiance modérée.
        best_reason = None
        if best_score > -11.0: 
            best_reason = criteria[best_idx].label
        
        return {"best_reason": best_reason, "all_scores": all_scores}

    def suggest_threshold(self, scores: list) -> dict:
        """
        Utilise des statistiques robustes (GMM) pour suggérer un seuil de coupure.
        """
        if not scores or len(scores) < 5:
            return {"threshold": 0.5, "method": "default (not enough data)"}
        
        scores_array = np.array(scores).reshape(-1, 1)
        
        try:
            # Essayer de fitter un GMM à 2 composants (Pertinent vs Non-pertinent)
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(scores_array)
            
            means = gmm.means_.flatten()
            threshold = np.mean(means)
            threshold = np.clip(threshold, min(scores), max(scores))
            
            return {
                "threshold": float(threshold),
                "method": "Gaussian Mixture Model (Advanced)",
                "confidence": "High"
            }
        except Exception as e:
            print(f"GMM failed: {e}")
            return {"threshold": np.median(scores), "method": "Median (Fallback)"}

    def is_valid_abstract(self, text: str) -> bool:
        """
        Vérifie si le texte ressemble à un vrai abstract.
        Retourne False si c'est une Préface, une Table des matières, ou du texte brut mal extrait.
        """
        if not text:
            return False
            
        text_clean = text.strip()
        if len(text_clean) < 50: # Augmenté à 50 chars
            return False
            
        text_lower = text_clean.lower()
        
        # Heuristiques de rejet (Faux abstracts)
        invalid_starts = ["preface", "table of contents", "contents", "chapter 1", "acknowledgments"]
        
        # Vérification élargie (pas seulement startswith, car parfois il y a du bruit au début)
        # On regarde dans les 200 premiers caractères
        start_text = text_lower[:200]
        for term in invalid_starts:
            if term in start_text:
                return False
        
        # Détection de livres (Recherche dans tout le texte car "This book" peut être à la fin de l'intro)
        if "this book" in text_lower or "in this book" in text_lower:
            return False
        
        # Détection de contenu brut (Légendes de figures, artefacts)
        if "figure 1" in text_lower or "fig. 1" in text_lower:
            return False
            
        return True

    def process_batch_and_update_db(self, article_ids: list, query: str, alpha: float = 0.7):
        """
        Méthode pour le traitement en arrière-plan avec sauvegarde incrémentale.
        Calcule BM25 globalement, puis Sémantique par petits lots.
        """
        db = next(get_db())
        try:
            articles = db.query(Article).filter(Article.id.in_(article_ids)).all()
            if not articles:
                return
            
            print(f"Starting incremental analysis for {len(articles)} articles...")
            
            # Récupérer les critères d'exclusion actifs
            criteria = db.query(ExclusionCriteria).filter(ExclusionCriteria.active == 1).all()
            
            # === LOGGING TRACABILITÉ : Créer l'entrée d'analyse ===
            from datetime import datetime
            start_time = datetime.utcnow()
            
            # Snapshot des critères actifs
            criteria_snapshot = json.dumps([
                {"label": c.label, "description": c.description}
                for c in criteria
            ]) if criteria else None
            
            # Déterminer session_id depuis les articles
            session_id = articles[0].search_session_id if articles else None
            
            # Créer l'entrée de run
            analysis_run = AIAnalysisRun(
                search_session_id=session_id,
                model_name=self.model_name,
                alpha=alpha,
                active_criteria=criteria_snapshot,
                articles_targeted=len(articles),
                articles_scored=0,
                articles_failed=0,
                status="RUNNING"
            )
            db.add(analysis_run)
            db.commit()  # Commit pour avoir l'ID
            run_id = analysis_run.id
            
            # --- ÉTAPE 1 : Calcul BM25 Global (Rapide) ---
            corpus_texts = []
            valid_articles = [] # Articles avec texte valide pour BM25
            
            for art in articles:
                # Validation Abstract (Règle dure)
                if not self.is_valid_abstract(art.abstract):
                    art.relevance_score = 0.0
                    art.suggested_reason = "Pas d'abstract" 
                    art.ia_metadata = json.dumps({"model": "RuleBased", "reason": "Invalid/Missing Abstract"})
                    # On sauvegarde tout de suite les rejetés
                    continue
                
                valid_articles.append(art)
                
                # Préparation texte BM25
                title = self.preprocess_text(art.title or '')
                abstract = self.preprocess_text(art.abstract or '')
                full_text = self.preprocess_text(art.full_text or '')
                combined_text = f"{title} {title} {abstract} {full_text}".strip()
                corpus_texts.append(combined_text)
            
            # Commit initial pour les rejetés (abstract invalide)
            db.commit()
            
            if not valid_articles:
                return

            # Calcul BM25
            tokenized_corpus = [doc.split(" ") for doc in corpus_texts]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.lower().split(" ")
            bm25_scores = bm25.get_scores(tokenized_query)
            
            # Normalisation BM25
            def normalize(scores):
                if not scores.any(): return np.array([])
                min_val, max_val = scores.min(), scores.max()
                if max_val == min_val: return np.zeros_like(scores)
                return (scores - min_val) / (max_val - min_val)

            norm_bm25 = normalize(bm25_scores)
            
            # Mapping ID -> Score BM25 Normalisé
            bm25_map = {art.id: score for art, score in zip(valid_articles, norm_bm25)}
            
            # --- ÉTAPE 2 : Calcul Sémantique Incrémental (Lent) ---
            # On traite par petits lots pour sauvegarder souvent
            BATCH_SIZE = 5
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            query_vec = query_embedding.cpu().numpy().reshape(1, -1)
            
            total_processed = 0
            
            for i in range(0, len(valid_articles), BATCH_SIZE):
                batch_articles = valid_articles[i:i+BATCH_SIZE]
                
                for art in batch_articles:
                    try:
                        # Préparation texte
                        title = self.preprocess_text(art.title or '')
                        abstract = self.preprocess_text(art.abstract or '')
                        full_text = self.preprocess_text(art.full_text or '')
                        combined_text = f"{title} {title} {abstract} {full_text}".strip()
                        
                        # Chunking & Embedding
                        chunks = self.chunk_text(combined_text)
                        
                        semantic_score = 0.0
                        if chunks:
                            chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)
                            chunk_vecs = chunk_embeddings.cpu().numpy()
                            # Max Pooling (Cosine Similarity brut : -1 à 1)
                            sims = cosine_similarity(query_vec, chunk_vecs)[0]
                            semantic_score = float(np.max(sims))
                            
                            # Clip pour rester cohérent (0 à 1)
                            semantic_score = max(0.0, min(1.0, semantic_score))
                        
                        # Score Hybride
                        # Note: On utilise le score sémantique BRUT (0-1) au lieu de normalisé par batch
                        # C'est plus stable pour l'incrémental.
                        bm25_val = bm25_map.get(art.id, 0.0)
                        hybrid_score = (alpha * semantic_score) + ((1 - alpha) * bm25_val)
                        
                        art.relevance_score = float(hybrid_score)
                        
                        # Vérification Critères (si score faible)
                        if hybrid_score < 0.4 and criteria:
                            text_for_check = f"{art.title} {art.abstract}"
                            criteria_result = self.check_exclusion_criteria(text_for_check, criteria)
                            
                            if criteria_result["best_reason"]:
                                art.suggested_reason = criteria_result["best_reason"]
                            
                            art.ia_metadata = json.dumps({
                                "model": self.model_name, 
                                "processed_at": str(np.datetime64('now')),
                                "criteria_scores": criteria_result["all_scores"],
                                "components": {"sem": semantic_score, "bm25": bm25_val}
                            })
                        else:
                            art.ia_metadata = json.dumps({
                                "model": self.model_name, 
                                "processed_at": str(np.datetime64('now')),
                                "components": {"sem": semantic_score, "bm25": bm25_val}
                            })
                            
                    except Exception as e_art:
                        print(f"Error processing article {art.id}: {e_art}")
                        art.ia_metadata = json.dumps({"error": str(e_art)})
                
                # Sauvegarde intermédiaire
                db.commit()
                total_processed += len(batch_articles)
                print(f"Progress: {total_processed}/{len(valid_articles)} articles saved.")
            
            # === LOGGING TRACABILITÉ : Finaliser ===
            end_time = datetime.utcnow()
            duration = int((end_time - start_time).total_seconds())
            
            analysis_run = db.query(AIAnalysisRun).filter(AIAnalysisRun.id == run_id).first()
            if analysis_run:
                analysis_run.articles_scored = total_processed
                analysis_run.articles_failed = len(valid_articles) - total_processed
                analysis_run.completed_at = end_time
                analysis_run.duration_seconds = duration
                analysis_run.status = "COMPLETED"
                db.commit()
            
        except Exception as e:
            print(f"Critical error in background processing: {e}")
            # Marquer le run comme échoué
            try:
                analysis_run = db.query(AIAnalysisRun).filter(AIAnalysisRun.id == run_id).first()
                if analysis_run:
                    analysis_run.status = "FAILED"
                    analysis_run.error_log = str(e)
                    analysis_run.completed_at = datetime.utcnow()
                    db.commit()
            except:
                pass
        finally:
            db.close()
