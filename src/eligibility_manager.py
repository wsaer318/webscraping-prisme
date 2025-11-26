# -*- coding: utf-8 -*-
"""
Gestionnaire de la phase Éligibilité
Fonctions pour gérer les critères, décisions, et statistiques
"""
from typing import List, Dict, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func
from src.database import Article, ArticleHistory, get_db
import sqlite3
from datetime import datetime


class EligibilityManager:
    """Gestionnaire pour la phase d'éligibilité"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_articles_to_review(self, limit: int = None) -> List[Article]:
        """Récupère les articles SCREENED_IN à réviser"""
        query = self.db.query(Article).filter(
            Article.status == "SCREENED_IN"
        ).order_by(Article.relevance_score.desc())
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def get_review_stats(self) -> Dict:
        """Statistiques de progression"""
        total_screened_in = self.db.query(Article).filter(
            Article.status == "SCREENED_IN"
        ).count()
        
        reviewed = self.db.query(Article).filter(
            (Article.status == "INCLUDED") | (Article.status == "EXCLUDED_ELIGIBILITY")
        ).count()
        
        included = self.db.query(Article).filter(
            Article.status == "INCLUDED"
        ).count()
        
        excluded = self.db.query(Article).filter(
            Article.status == "EXCLUDED_ELIGIBILITY"
        ).count()
        
        return {
            "total_to_review": total_screened_in,
            "reviewed": reviewed,
            "included": included,
            "excluded": excluded,
            "remaining": total_screened_in,
            "progress_pct": 0 if total_screened_in == 0 else (reviewed / (reviewed + total_screened_in)) * 100
        }
    
    def save_decision(
        self,
        article_id: int,
        decision: str,
        reasons: List[str] = None,
        notes: str = "",
        reviewer: str = "User"
    ) -> bool:
        """
        Enregistre une décision d'éligibilité
        
        Args:
            article_id: ID de l'article
            decision: "INCLUDED" ou "EXCLUDED_ELIGIBILITY"
            reasons: Liste des raisons d'exclusion (si applicable)
            notes: Notes du reviewer
            reviewer: Nom du reviewer
        """
        try:
            article = self.db.query(Article).filter(Article.id == article_id).first()
            
            if not article:
                return False
            
            # Sauvegarder état précédent
            previous_status = article.status
            
            # Mettre à jour article
            article.status = decision
            article.eligibility_notes = notes
            article.reviewed_at = datetime.now()
            article.reviewer = reviewer
            
            # Logger dans ArticleHistory
            history = ArticleHistory(
                article_id=article_id,
                previous_status=previous_status,
                new_status=decision,
                action="ELIGIBILITY_REVIEW",
                reason=", ".join(reasons) if reasons else "N/A",
                user=reviewer
            )
            
            self.db.add(history)
            self.db.commit()
            
            return True
            
        except Exception as e:
            self.db.rollback()
            print(f"❌ Erreur sauvegarde décision: {e}")
            return False
    
    def get_active_criteria(self) -> List[Dict]:
        """Récupère les critères d'éligibilité actifs"""
        conn = sqlite3.connect('data/prisma.db')
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT id, label, description, type 
        FROM eligibility_criteria 
        WHERE active = 1
        ORDER BY type DESC, label ASC
        """)
        
        criteria = []
        for row in cursor.fetchall():
            criteria.append({
                "id": row[0],
                "label": row[1],
                "description": row[2],
                "type": row[3]
            })
        
        conn.close()
        return criteria
    
    def get_exclusion_stats(self) -> Dict:
        """Statistiques des raisons d'exclusion"""
        # Récupérer tous les historiques d'exclusion
        histories = self.db.query(ArticleHistory).filter(
            ArticleHistory.new_status == "EXCLUDED_ELIGIBILITY"
        ).all()
        
        reason_counts = {}
        for h in histories:
            if h.reason:
                # Séparer si plusieurs raisons
                reasons = h.reason.split(", ")
                for r in reasons:
                    r = r.strip()
                    if r and r != "N/A":
                        reason_counts[r] = reason_counts.get(r, 0) + 1
        
        return reason_counts


def get_next_article_to_review(current_id: int = None) -> Article:
    """
    Récupère le prochain article à réviser
    
    Args:
        current_id: ID de l'article actuel (pour skip)
    """
    db = next(get_db())
    
    query = db.query(Article).filter(
        Article.status == "SCREENED_IN"
    ).order_by(Article.relevance_score.desc())
    
    if current_id:
        query = query.filter(Article.id != current_id)
    
    article = query.first()
    db.close()
    
    return article


if __name__ == "__main__":
    # Test
    db = next(get_db())
    manager = EligibilityManager(db)
    
    print("Stats:", manager.get_review_stats())
    print("\nCritères actifs:", manager.get_active_criteria())
    
    db.close()
