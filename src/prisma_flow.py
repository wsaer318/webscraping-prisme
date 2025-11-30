# -*- coding: utf-8 -*-
"""
Génération du diagramme PRISMA Flow
Conforme aux standards PRISMA 2020 - VERSION DYNAMIQUE avec traçabilité
"""
from src.database import (
    get_db, Article, ArticleHistory, SearchSession,
    SemanticFilterRun, AIAnalysisRun
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import json


def get_prisma_counts(db, session_id=None):
    """
    Récupère les comptages pour le diagramme PRISMA depuis la traçabilité
    
    Args:
        db: Session database
        session_id: ID de session optionnel pour filtrer
        
    Returns:
        dict avec tous les comptages et métadonnées
    """
    counts = {}
    
    # Filtre session
    session_filter = (Article.search_session_id == session_id) if session_id else True
    
    # === IDENTIFICATION ===
    counts['identified'] = db.query(Article).filter(session_filter).count()
    
    # Déduplication (placeholder pour future implémentation)
    counts['after_dedup'] = counts['identified']
    
    # === PRE-SCREENING (Semantic Filter) ===
    semantic_run = db.query(SemanticFilterRun).filter(
        SemanticFilterRun.search_session_id == session_id if session_id else True
    ).order_by(SemanticFilterRun.created_at.desc()).first()
    
    if semantic_run:
        counts['semantic_filter_applied'] = True
        counts['semantic_concepts'] = json.loads(semantic_run.concepts)
        counts['semantic_threshold'] = semantic_run.threshold
        counts['semantic_mode'] = semantic_run.mode
        counts['semantic_excluded'] = semantic_run.articles_excluded
        counts['after_semantic'] = semantic_run.articles_retained
    else:
        counts['semantic_filter_applied'] = False
        counts['semantic_excluded'] = 0
        counts['after_semantic'] = counts['after_dedup']
    
    # === SCREENING ===
    counts['screened_in'] = db.query(Article).filter(
        session_filter,
        Article.status == 'SCREENED_IN'
    ).count()
    
    counts['screened_out'] = db.query(Article).filter(
        session_filter,
        Article.status == 'EXCLUDED_SCREENING'
    ).count()
    
    counts['screened'] = counts['screened_in'] + counts['screened_out']
    
    # === ELIGIBILITY ===
    counts['eligibility_assessed'] = db.query(Article).filter(
        session_filter,
        Article.status.in_(['INCLUDED', 'EXCLUDED_ELIGIBILITY'])
    ).count()
    
    counts['excluded_eligibility'] = db.query(Article).filter(
        session_filter,
        Article.status == 'EXCLUDED_ELIGIBILITY'
    ).count()
    
    # === INCLUDED ===
    counts['included'] = db.query(Article).filter(
        session_filter,
        Article.status == 'INCLUDED'
    ).count()
    
    return counts


def generate_prisma_diagram(output_path: str = None) -> str:
    """
    Génère le diagramme PRISMA flow
    
    Args:
        output_path: Chemin de sauvegarde (défaut: data/prisma_flow.png)
        
    Returns:
        Chemin du fichier généré
    """
    db = next(get_db())
    counts = get_prisma_counts(db)
    db.close()
    
    if output_path is None:
        output_path = Path("data") / "prisma_flow.png"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Couleurs
    color_id = '#E3F2FD'  # Bleu clair
    color_screen = '#FFF9C4'  # Jaune clair
    color_elig = '#F0F4C3'  # Vert clair
    color_inc = '#C8E6C9'  # Vert
    color_exc = '#FFCDD2'  # Rouge clair
    
    # Fonction helper pour créer des boîtes
    def add_box(x, y, w, h, text, color, fontsize=11, weight='normal'):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=color,
            linewidth=1.5
        )
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text,
                ha='center', va='center',
                fontsize=fontsize, weight=weight,
                wrap=True)
    
    # Fonction helper pour flèches
    def add_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # ===== DIAGRAMME =====
    
    # Identification
    add_box(2, 12, 6, 1.2, 
            f"Articles identifiés\n(n = {counts['identified']})",
            color_id, fontsize=12, weight='bold')
    
    add_arrow(5, 12, 5, 10.5)
    
    # Pré-filtre sémantique (si appliqué)
    y_current = 10.5
    if counts['semantic_filter_applied']:
        concepts_str = ', '.join(counts['semantic_concepts'][:2])  # Max 2 concepts
        if len(counts['semantic_concepts']) > 2:
            concepts_str += '...'
        
        add_box(2, y_current-1.2, 6, 1.2,
                f"Après pré-filtre sémantique\n({concepts_str}, seuil={counts['semantic_threshold']})\n(n = {counts['after_semantic']})",
                '#FFF3E0', fontsize=10, weight='bold')
        
        # Exclus sémantique (droite)
        add_box(8.3, y_current-1.2, 1.5, 1.2,
                f"Exclus\nsémantique\n(n = {counts['semantic_excluded']})",
                color_exc, fontsize=9)
        
        add_arrow(5, y_current-1.2, 5, y_current-2.7)
        y_current = y_current - 2.7
    else:
        add_arrow(5, y_current, 5, y_current-1.5)
        y_current = y_current - 1.5
    
    # Screening
    add_box(2, y_current-1.2, 6, 1.2,
            f"Articles screenés\n(n = {counts['screened']})",
            color_screen, fontsize=12, weight='bold')
    
    # Exclus screening
    add_box(0.2, y_current-1.2, 1.5, 1.2,
            f"Exclus\nscreening\n(n = {counts['screened_out']})",
            color_exc, fontsize=9)
    
    add_arrow(5, y_current-1.2, 5, y_current-2.7)
    y_current = y_current - 2.7
    
    # Éligibilité
    add_box(2, y_current-1.2, 6, 1.2,
            f"Éligibilité évaluée\n(texte complet)\n(n = {counts['eligibility_assessed']})",
            color_elig, fontsize=11, weight='bold')
    
    # Exclus éligibilité
    add_box(0.2, y_current-1.2, 1.5, 1.2,
            f"Exclus\néligibilité\n(n = {counts['excluded_eligibility']})",
            color_exc, fontsize=9)
    
    add_arrow(5, y_current-1.2, 5, y_current-2.7)
    y_current = y_current - 2.7
    
    # Inclusion finale
    add_box(2, y_current-1.2, 6, 1.2,
            f"Articles inclus\ndans la synthèse\n(n = {counts['included']})",
            color_inc, fontsize=12, weight='bold')
    
    # Titre
    ax.text(5, 13.5, 'PRISMA 2020 Flow Diagram',
           ha='center', fontsize=16, weight='bold')
    
    # Légende phases
    phase_y = 2.5
    ax.text(1, phase_y, 'Phases:', fontsize=10, weight='bold')
    ax.text(1, phase_y-0.4, '• Identification', fontsize=9, color='#1976D2')
    ax.text(1, phase_y-0.7, '• Screening', fontsize=9, color='#F57C00')
    ax.text(1, phase_y-1.0, '• Éligibilité', fontsize=9, color='#689F38')
    ax.text(1, phase_y-1.3, '• Inclusion', fontsize=9, color='#388E3C')
    
    # Statistiques
    stats_y = 1.0
    taux_screening = (counts['screened_in'] / counts['screened'] * 100) if counts['screened'] > 0 else 0
    taux_eligibility = (counts['included'] / counts['eligibility_assessed'] * 100) if counts['eligibility_assessed'] > 0 else 0
    taux_global = (counts['included'] / counts['identified'] * 100) if counts['identified'] > 0 else 0
    
    ax.text(5, stats_y, f"Taux de sélection global : {taux_global:.1f}%",
           ha='center', fontsize=10, style='italic')
    
    # Sauvegarder
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Diagramme PRISMA généré : {output_path}")
    
    return str(output_path)


if __name__ == "__main__":
    # Test
    path = generate_prisma_diagram()
    print(f"Diagramme sauvegardé : {path}")
