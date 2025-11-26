# -*- coding: utf-8 -*-
"""
Génération du diagramme PRISMA Flow
Conforme aux standards PRISMA 2020
"""
from src.database import get_db, Article, ArticleHistory
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path


def get_prisma_counts(db):
    """Récupère les comptages pour le diagramme PRISMA"""
    
    counts = {}
    
    # Identification : TOUS les articles dans la BDD (historique complet)
    counts['identified'] = db.query(Article).count()
    
    # Screening : Articles qui ont été traités (pas ceux encore IDENTIFIED)
    counts['screened'] = db.query(Article).filter(
        Article.status != 'IDENTIFIED'  # Tous sauf IDENTIFIED = ont été screenés
    ).count()
    
    # Exclus au screening
    counts['screened_out'] = db.query(Article).filter(
        Article.status == 'SCREENED_OUT'
    ).count()
    
    # Passés au screening (pour éligibilité)
    counts['screened_in'] = db.query(Article).filter(
        Article.status.in_(['SCREENED_IN', 'INCLUDED', 'EXCLUDED_ELIGIBILITY'])
    ).count()
    
    # Éligibilité : Articles évalués en full-text
    counts['eligibility_assessed'] = db.query(Article).filter(
        Article.status.in_(['INCLUDED', 'EXCLUDED_ELIGIBILITY'])
    ).count()
    
    # Exclus éligibilité
    counts['excluded_eligibility'] = db.query(Article).filter(
        Article.status == 'EXCLUDED_ELIGIBILITY'
    ).count()
    
    # Inclusion finale
    counts['included'] = db.query(Article).filter(
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
    
    # Screening
    add_box(2, 9.3, 6, 1.2,
            f"Articles screenés\n(n = {counts['screened']})",
            color_screen, fontsize=12, weight='bold')
    
    # Exclus screening
    add_box(0.2, 9.3, 1.5, 1.2,
            f"Exclus\nscreening\n(n = {counts['screened_out']})",
            color_exc, fontsize=9)
    
    add_arrow(5, 9.3, 5, 7.8)
    
    # Éligibilité
    add_box(2, 6.6, 6, 1.2,
            f"Éligibilité évaluée\n(texte complet)\n(n = {counts['eligibility_assessed']})",
            color_elig, fontsize=11, weight='bold')
    
    # Exclus éligibilité
    add_box(0.2, 6.6, 1.5, 1.2,
            f"Exclus\néligibilité\n(n = {counts['excluded_eligibility']})",
            color_exc, fontsize=9)
    
    add_arrow(5, 6.6, 5, 5.1)
    
    # Inclusion finale
    add_box(2, 3.9, 6, 1.2,
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
