# -*- coding: utf-8 -*-
"""
Module d'analytics pour statistiques PRISMA
"""
from src.database import get_db, Article, ArticleHistory
from typing import Dict, List
from collections import Counter
import json


def get_global_stats(db) -> Dict:
    """Statistiques globales du workflow PRISMA"""
    
    stats = {}
    
    # Comptages par statut
    stats['identified'] = db.query(Article).count()
    stats['screened_out'] = db.query(Article).filter(Article.status == 'SCREENED_OUT').count()
    stats['screened_in'] = db.query(Article).filter(Article.status == 'SCREENED_IN').count()
    stats['excluded_eligibility'] = db.query(Article).filter(Article.status == 'EXCLUDED_ELIGIBILITY').count()
    stats['included'] = db.query(Article).filter(Article.status == 'INCLUDED').count()
    
    # Taux
    if stats['identified'] > 0:
        stats['screening_rate'] = (stats['screened_out'] + stats['screened_in']) / stats['identified'] * 100
        stats['inclusion_rate'] = stats['included'] / stats['identified'] * 100
    else:
        stats['screening_rate'] = 0
        stats['inclusion_rate'] = 0
    
    # PDFs
    stats['with_pdf'] = db.query(Article).filter(Article.pdf_path.isnot(None)).count()
    stats['with_fulltext'] = db.query(Article).filter(Article.full_text.isnot(None)).count()
    
    return stats


def get_exclusion_distribution(db) -> Dict:
    """Distribution des raisons d'exclusion par phase"""
    
    distribution = {
        'screening': {},
        'eligibility': {}
    }
    
    # Phase Screening
    screened_out = db.query(Article).filter(Article.status == 'SCREENED_OUT').all()
    screening_reasons = [a.exclusion_reason for a in screened_out if a.exclusion_reason]
    distribution['screening'] = dict(Counter(screening_reasons))
    
    # Phase Éligibilité
    excluded_elig = db.query(ArticleHistory).filter(
        ArticleHistory.new_status == 'EXCLUDED_ELIGIBILITY'
    ).all()
    
    elig_reasons = []
    for h in excluded_elig:
        if h.reason and h.reason != "N/A":
            # Séparer si plusieurs raisons
            reasons = h.reason.split(", ")
            elig_reasons.extend(reasons)
    
    distribution['eligibility'] = dict(Counter(elig_reasons))
    
    return distribution


def get_temporal_distribution(db) -> Dict:
    """Distribution temporelle des articles"""
    
    articles = db.query(Article.year).filter(Article.year.isnot(None)).all()
    years = [a.year for a in articles]
    
    year_counts = dict(Counter(years))
    
    # Trier par année
    sorted_years = sorted(year_counts.items())
    
    return {
        'years': [y[0] for y in sorted_years],
        'counts': [y[1] for y in sorted_years]
    }


def get_source_distribution(db) -> Dict:
    """Distribution par source"""
    
    articles = db.query(Article.source).filter(Article.source.isnot(None)).all()
    sources = [a.source for a in articles]
    
    return dict(Counter(sources))


def get_included_articles_summary(db) -> List[Dict]:
    """Résumé des articles inclus pour tableau"""
    
    included = db.query(Article).filter(Article.status == 'INCLUDED').all()
    
    summary = []
    for article in included:
        summary.append({
            'id': article.id,
            'title': article.title,
            'authors': article.authors,
            'year': article.year,
            'source': article.source,
            'doi': article.doi,
            'link': article.link,
            'reviewer': article.reviewer or 'N/A',
            'reviewed_at': article.reviewed_at.strftime('%Y-%m-%d') if article.reviewed_at else 'N/A'
        })
    
    return summary


def get_review_timeline(db) -> Dict:
    """Timeline des décisions de revue"""
    
    histories = db.query(ArticleHistory).filter(
        ArticleHistory.action.in_(['SCREENING_DECISION', 'ELIGIBILITY_REVIEW'])
    ).all()
    
    timeline = []
    for h in histories:
        timeline.append({
            'date': h.timestamp.strftime('%Y-%m-%d'),
            'action': h.action,
            'decision': h.new_status,
            'user': h.user
        })
    
    # Grouper par date
    from collections import defaultdict
    daily_counts = defaultdict(int)
    
    for event in timeline:
        daily_counts[event['date']] += 1
    
    return {
        'dates': list(daily_counts.keys()),
        'counts': list(daily_counts.values())
    }


if __name__ == "__main__":
    # Test
    db = next(get_db())
    
    print("Stats globales:", get_global_stats(db))
    print("\nDistribution exclusions:", get_exclusion_distribution(db))
    print("\nDistribution temporelle:", get_temporal_distribution(db))
    print("\nSources:", get_source_distribution(db))
    
    db.close()
