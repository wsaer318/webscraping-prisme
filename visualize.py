# -*- coding: utf-8 -*-
"""
Module de Visualisation pour le Pipeline Amélioré
==================================================

Génère des graphiques informatifs :
- Distribution des scores (embedding, BM25, fusionnés)
- Visualisation des clusters (t-SNE/UMAP)
- Matrices de similarité
- Statistiques détaillées
- Comparaison avant/après filtrage

Usage:
    python visualize.py --report articles_report.json --output visualizations/
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Configuration style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Palette de couleurs
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'warning': '#F77F00',
    'danger': '#D62828',
    'neutral': '#6C757D',
    'light': '#E9ECEF'
}

class PipelineVisualizer:
    """Génère des visualisations pour le pipeline"""
    
    def __init__(self, report_path: str, csv_path: Optional[str] = None):
        """
        Args:
            report_path: Chemin vers articles_report.json
            csv_path: Chemin optionnel vers articles_final.csv
        """
        self.report_path = report_path
        self.csv_path = csv_path
        
        # Charger les données
        with open(report_path, 'r', encoding='utf-8') as f:
            self.report = json.load(f)
        
        if csv_path and os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path, encoding='utf-8')
        else:
            self.df = None
    
    def plot_score_distributions(self, output_path: str):
        """Distribution des scores (embedding, BM25, final)"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Distribution des Scores', fontsize=16, fontweight='bold')
        
        if self.df is not None and all(col in self.df.columns for col in ['score_embed', 'score_bm25', 'score']):
            # Score embedding
            ax = axes[0, 0]
            ax.hist(self.df['score_embed'], bins=30, color=COLORS['primary'], alpha=0.7, edgecolor='black')
            ax.axvline(self.df['score_embed'].mean(), color=COLORS['danger'], linestyle='--', 
                      linewidth=2, label=f'Moyenne: {self.df["score_embed"].mean():.4f}')
            ax.set_xlabel('Score Embedding')
            ax.set_ylabel('Frequence')
            ax.set_title('Distribution - Scores Embedding')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Score BM25
            ax = axes[0, 1]
            ax.hist(self.df['score_bm25'], bins=30, color=COLORS['secondary'], alpha=0.7, edgecolor='black')
            ax.axvline(self.df['score_bm25'].mean(), color=COLORS['danger'], linestyle='--', 
                      linewidth=2, label=f'Moyenne: {self.df["score_bm25"].mean():.2f}')
            ax.set_xlabel('Score BM25')
            ax.set_ylabel('Frequence')
            ax.set_title('Distribution - Scores BM25')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Score final
            ax = axes[1, 0]
            ax.hist(self.df['score'], bins=30, color=COLORS['accent'], alpha=0.7, edgecolor='black')
            threshold = self.report['thresholds']['value']
            ax.axvline(threshold, color=COLORS['danger'], linestyle='--', 
                      linewidth=2, label=f'Seuil: {threshold:.4f}')
            ax.axvline(self.df['score'].mean(), color=COLORS['success'], linestyle=':', 
                      linewidth=2, label=f'Moyenne: {self.df["score"].mean():.4f}')
            ax.set_xlabel('Score Final')
            ax.set_ylabel('Frequence')
            ax.set_title('Distribution - Scores Finaux (Fusionnes)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Statistiques textuelles
            ax = axes[1, 1]
            ax.axis('off')
            
            stats_text = f"""
STATISTIQUES DES SCORES

Scores Embedding:
  - Min:     {self.df['score_embed'].min():.4f}
  - Max:     {self.df['score_embed'].max():.4f}
  - Moyenne: {self.df['score_embed'].mean():.4f}
  - Mediane: {self.df['score_embed'].median():.4f}
  - Std:     {self.df['score_embed'].std():.4f}

Scores BM25:
  - Min:     {self.df['score_bm25'].min():.2f}
  - Max:     {self.df['score_bm25'].max():.2f}
  - Moyenne: {self.df['score_bm25'].mean():.2f}
  - Mediane: {self.df['score_bm25'].median():.2f}
  - Std:     {self.df['score_bm25'].std():.2f}

Scores Finaux:
  - Min:     {self.df['score'].min():.4f}
  - Max:     {self.df['score'].max():.4f}
  - Moyenne: {self.df['score'].mean():.4f}
  - Mediane: {self.df['score'].median():.4f}
  - Std:     {self.df['score'].std():.4f}
            """
            ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                   verticalalignment='center', bbox=dict(boxstyle='round', 
                   facecolor=COLORS['light'], alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Distribution des scores -> {output_path}")
    
    def plot_threshold_analysis(self, output_path: str):
        """Analyse du seuillage"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Analyse du Seuillage Automatique', fontsize=16, fontweight='bold')
        
        if self.df is not None and 'score' in self.df.columns:
            scores = self.df['score'].values
            threshold = self.report['thresholds']['value']
            threshold_source = self.report['thresholds']['source']
            
            # Histogramme avec zones
            ax = axes[0]
            n, bins, patches = ax.hist(scores, bins=40, color=COLORS['neutral'], 
                                       alpha=0.6, edgecolor='black')
            
            # Colorer les zones
            for i, patch in enumerate(patches):
                if bins[i] < threshold:
                    patch.set_facecolor(COLORS['danger'])
                    patch.set_alpha(0.5)
                else:
                    patch.set_facecolor(COLORS['success'])
                    patch.set_alpha(0.7)
            
            ax.axvline(threshold, color='black', linestyle='--', linewidth=3, 
                      label=f'Seuil ({threshold_source}): {threshold:.4f}')
            ax.set_xlabel('Score Final')
            ax.set_ylabel('Frequence')
            ax.set_title('Repartition: Rejetes vs Acceptes')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Informations sur le seuillage
            ax = axes[1]
            ax.axis('off')

            counts = self.report['counts']
            thr_meta = self.report['thresholds'].get('metadata', {})

            # Calculs des taux de rétention cohérents
            threshold_retention = (counts['after_threshold'] / counts['after_filters']) * 100
            global_retention = (counts['final_selected'] / counts['total_initial']) * 100

            info_text = f"""
METHODE DE SEUILLAGE
{'='*40}

Source: {threshold_source}
Seuil calcule: {threshold:.4f}

RESULTATS DU FILTRAGE
{'='*40}

Articles initiaux:    {counts['total_initial']}
Apres filtres:        {counts['after_filters']}
Apres seuillage:      {counts['after_threshold']}
Selection finale:     {counts['final_selected']}

TAUX DE RETENTION
{'='*40}

Seuillage (vs filtres): {threshold_retention:.1f}%
Global (vs initial):    {global_retention:.1f}%
Articles rejetes:       {counts['after_filters'] - counts['after_threshold']}
            """
            
            # Ajouter détails méthode si disponible
            if threshold_source == 'ensemble' and thr_meta:
                info_text += f"\n\nDETAILS ENSEMBLE\n{'='*40}\n"
                if 'n_methods_valid' in thr_meta:
                    info_text += f"Methodes valides: {thr_meta['n_methods_valid']}\n"
                if 'weights' in thr_meta:
                    info_text += "\nPonderations:\n"
                    for method, weight in thr_meta['weights'].items():
                        info_text += f"  {method}: {weight:.3f}\n"
            
            ax.text(0.1, 0.5, info_text, fontsize=9, family='monospace',
                   verticalalignment='center', bbox=dict(boxstyle='round', 
                   facecolor=COLORS['light'], alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Analyse du seuillage -> {output_path}")
    
    def plot_clusters_2d(self, output_path: str):
        """Visualisation 2D des clusters (t-SNE)"""
        if self.df is None or 'cluster_id' not in self.df.columns:
            print("[WARN] Pas de donnees de clustering disponibles")
            return
        
        # Pour la visualisation 2D, on a besoin des embeddings
        # Comme on n'a pas directement les embeddings, on va créer un proxy avec PCA sur les scores
        # Note: idéalement, on sauvegarderait les embeddings dans le CSV
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Visualisation des Clusters', fontsize=16, fontweight='bold')
        
        cluster_info = self.report.get('clustering', {})
        n_clusters = cluster_info.get('n_clusters', 0)
        
        # Créer features pour visualisation (proxy basé sur les scores disponibles)
        if all(col in self.df.columns for col in ['score_embed', 'score_bm25', 'score_title', 'score_abstract', 'score_body']):
            X_proxy = self.df[['score_embed', 'score_bm25', 'score_title', 'score_abstract', 'score_body']].values
        else:
            X_proxy = self.df[['score_embed', 'score_bm25', 'score']].values
        
        # t-SNE
        if len(X_proxy) > 2:
            perplexity = min(30, len(X_proxy) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            X_2d = tsne.fit_transform(X_proxy)
            
            ax = axes[0]
            clusters = self.df['cluster_id'].values
            
            # Palette de couleurs pour les clusters
            unique_clusters = sorted(set(clusters))
            colors = sns.color_palette("husl", len(unique_clusters))
            color_map = {cl: colors[i] for i, cl in enumerate(unique_clusters)}
            
            for cluster_id in unique_clusters:
                mask = clusters == cluster_id
                label = f'Cluster {cluster_id}' if cluster_id != -1 else 'Bruit'
                color = color_map[cluster_id] if cluster_id != -1 else COLORS['neutral']
                marker = 'o' if cluster_id != -1 else 'x'
                size = 100 if cluster_id != -1 else 50
                
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                          c=[color], label=label, s=size, alpha=0.7, 
                          edgecolors='black', linewidth=0.5, marker=marker)
            
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_title('Projection t-SNE des Articles')
            ax.legend(loc='best', ncol=2)
            ax.grid(True, alpha=0.3)
        
        # Statistiques par cluster
        ax = axes[1]
        ax.axis('off')
        
        cluster_stats = self.df.groupby('cluster_id').agg({
            'score': ['count', 'mean', 'std'],
            'score_embed': 'mean',
            'score_bm25': 'mean'
        }).round(4)
        
        stats_text = f"""
STATISTIQUES DES CLUSTERS
{'='*50}

Nombre de clusters: {n_clusters}
Articles en clusters: {(self.df['cluster_id'] != -1).sum()}
Articles bruit (-1): {(self.df['cluster_id'] == -1).sum()}

Qualite du clustering:
  - Silhouette:         {cluster_info.get('silhouette', 'N/A')}
  - Calinski-Harabasz:  {cluster_info.get('calinski_harabasz', 'N/A')}
  - Davies-Bouldin:     {cluster_info.get('davies_bouldin', 'N/A')}

Methode: {cluster_info.get('method', 'N/A')}
        """
        
        if cluster_info.get('method') == 'dbscan':
            stats_text += f"\nEpsilon: {cluster_info.get('eps', 'N/A'):.4f}"
            stats_text += f"\nMin samples: {cluster_info.get('min_samples', 'N/A')}"
        
        stats_text += f"\n\nRepartition par cluster:\n{'-'*50}\n"
        for cluster_id in sorted(set(self.df['cluster_id'])):
            count = (self.df['cluster_id'] == cluster_id).sum()
            pct = count / len(self.df) * 100
            stats_text += f"\nCluster {cluster_id:2d}: {count:3d} articles ({pct:5.1f}%)"
        
        ax.text(0.05, 0.5, stats_text, fontsize=9, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', 
               facecolor=COLORS['light'], alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Visualisation des clusters -> {output_path}")
    
    def plot_pipeline_flow(self, output_path: str):
        """Diagramme de flux du pipeline"""
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('Flux du Pipeline - Entonnoir de Filtrage', fontsize=16, fontweight='bold')
        
        counts = self.report['counts']
        
        # Étapes et comptages
        stages = [
            ('Articles initiaux', counts['total_initial']),
            ('Filtres (langue, longueur)', counts['after_filters']),
            ('Deduplication', counts['after_filters']),  # Assumé identique après correction
            ('Seuillage', counts['after_threshold']),
            ('Selection finale', counts['final_selected'])
        ]
        
        # Calculer les largeurs proportionnelles
        max_count = stages[0][1]
        y_positions = np.arange(len(stages))[::-1]
        
        colors_stages = [COLORS['neutral'], COLORS['primary'], COLORS['secondary'], 
                        COLORS['accent'], COLORS['success']]
        
        for i, ((stage, count), y_pos) in enumerate(zip(stages, y_positions)):
            width = (count / max_count) * 0.8
            x_center = 0.5
            x_left = x_center - width / 2
            
            # Rectangle
            rect = Rectangle((x_left, y_pos - 0.3), width, 0.6, 
                            facecolor=colors_stages[i], edgecolor='black', 
                            linewidth=2, alpha=0.7)
            ax.add_patch(rect)
            
            # Texte
            ax.text(x_center, y_pos, f'{stage}\n{count} articles', 
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
            
            # Flèche vers l'étape suivante
            if i < len(stages) - 1:
                ax.annotate('', xy=(x_center, y_pos - 0.5), xytext=(x_center, y_pos - 0.3),
                           arrowprops=dict(arrowstyle='->', lw=2, color='black'))
                
                # Afficher les pertes
                if i > 0:
                    loss = stages[i-1][1] - count
                    if loss > 0:
                        loss_pct = (loss / stages[i-1][1]) * 100
                        ax.text(x_center + 0.45, y_pos + 0.15, 
                               f'-{loss} (-{loss_pct:.1f}%)', 
                               fontsize=9, color=COLORS['danger'], 
                               fontweight='bold',
                               bbox=dict(boxstyle='round', facecolor='white', 
                                       edgecolor=COLORS['danger'], alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, len(stages))
        ax.axis('off')
        
        # Légende finale - calcul cohérent avec le pipeline
        global_retention = (counts['final_selected'] / counts['total_initial']) * 100
        legend_text = f"Taux de retention global: {global_retention:.1f}%\n(Selection finale / Articles initiaux)"
        ax.text(0.5, -0.5, legend_text, ha='center', fontsize=11,
               fontweight='bold', color=COLORS['success'],
               bbox=dict(boxstyle='round', facecolor=COLORS['light'],
                       edgecolor=COLORS['success'], linewidth=2, alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Flux du pipeline -> {output_path}")
    
    def plot_top_articles(self, output_path: str, top_n: int = 10):
        """Graphique des N meilleurs articles"""
        if self.df is None:
            print("[WARN] Pas de CSV disponible")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle(f'Top {top_n} Articles Selectionnes', fontsize=16, fontweight='bold')
        
        # Prendre les top_n
        top_df = self.df.head(top_n).copy()
        
        # Préparer les données
        y_pos = np.arange(len(top_df))
        scores = top_df['score'].values
        titles = [t[:60] + '...' if len(t) > 60 else t for t in top_df['title']]
        
        # Barres horizontales
        bars = ax.barh(y_pos, scores, color=COLORS['primary'], alpha=0.7, edgecolor='black')
        
        # Colorer en fonction du cluster
        if 'cluster_id' in top_df.columns:
            unique_clusters = sorted(set(top_df['cluster_id']))
            colors = sns.color_palette("husl", len(unique_clusters))
            color_map = {cl: colors[i] for i, cl in enumerate(unique_clusters)}
            
            for i, (bar, cluster_id) in enumerate(zip(bars, top_df['cluster_id'])):
                if cluster_id != -1:
                    bar.set_color(color_map[cluster_id])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{i+1}. {t}" for i, t in enumerate(titles)], fontsize=9)
        ax.set_xlabel('Score Final', fontsize=11)
        ax.set_title('Classement par pertinence', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        
        # Annotations
        for i, (y, score) in enumerate(zip(y_pos, scores)):
            ax.text(score + 0.0002, y, f'{score:.4f}', 
                   va='center', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Top articles -> {output_path}")
    
    def plot_score_correlation(self, output_path: str):
        """Matrice de corrélation entre les scores"""
        if self.df is None:
            print("[WARN] Pas de CSV disponible")
            return
        
        score_cols = ['score_title', 'score_abstract', 'score_body', 'score_embed', 'score_bm25', 'score']
        available_cols = [col for col in score_cols if col in self.df.columns]
        
        if len(available_cols) < 3:
            print("[WARN] Pas assez de colonnes de scores")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.suptitle('Correlations entre les Scores', fontsize=16, fontweight='bold')
        
        corr_matrix = self.df[available_cols].corr()
        
        # Heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax, vmin=-1, vmax=1)
        
        ax.set_title('Matrice de correlation (Pearson)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Correlations -> {output_path}")
    
    def plot_similarity_heatmap(self, output_path: str):
        """Heatmap de similarité entre articles sélectionnés"""
        similarity_matrix = self.report.get('similarity_matrix')
        
        if similarity_matrix is None:
            print("[WARN] Matrice de similarite non disponible")
            return
        
        if len(similarity_matrix) == 0:
            print("[WARN] Matrice de similarite vide")
            return
        
        fig, ax = plt.subplots(figsize=(14, 12))
        fig.suptitle('Matrice de Similarite (Cosine) entre Articles', fontsize=16, fontweight='bold')
        
        sim_array = np.array(similarity_matrix)
        
        # Créer la heatmap
        im = ax.imshow(sim_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Similarite Cosine', rotation=270, labelpad=20)
        
        # Annotations pour petites matrices
        if len(sim_array) <= 30:
            for i in range(len(sim_array)):
                for j in range(len(sim_array)):
                    text = ax.text(j, i, f'{sim_array[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=7)
        
        # Labels
        if self.df is not None and len(self.df) == len(sim_array):
            labels = [f"{i+1}" for i in range(len(sim_array))]
            ax.set_xticks(np.arange(len(sim_array)))
            ax.set_yticks(np.arange(len(sim_array)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)
        
        ax.set_xlabel('Article ID', fontsize=11)
        ax.set_ylabel('Article ID', fontsize=11)
        ax.set_title(f'Similarite entre {len(sim_array)} articles selectionnes', fontsize=12)
        
        # Statistiques
        upper_tri = sim_array[np.triu_indices_from(sim_array, k=1)]
        avg_sim = np.mean(upper_tri)
        min_sim = np.min(upper_tri)
        max_sim = np.max(upper_tri)
        
        stats_text = f"Similarite moyenne: {avg_sim:.3f}\nMin: {min_sim:.3f} | Max: {max_sim:.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=COLORS['light'], alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Heatmap de similarite -> {output_path}")
    
    def plot_text_length_distributions(self, output_path: str):
        """Distribution des longueurs de texte"""
        text_lengths = self.report.get('text_lengths')
        
        if not text_lengths or self.df is None:
            print("[WARN] Donnees de longueur de texte non disponibles")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Distribution des Longueurs de Texte', fontsize=16, fontweight='bold')
        
        # Longueurs titres
        ax = axes[0, 0]
        if 'title' in self.df.columns:
            lengths = self.df['title'].str.len()
            ax.hist(lengths, bins=30, color=COLORS['primary'], alpha=0.7, edgecolor='black')
            ax.axvline(lengths.mean(), color=COLORS['danger'], linestyle='--', 
                      linewidth=2, label=f'Moyenne: {lengths.mean():.0f}')
            ax.set_xlabel('Longueur (caracteres)')
            ax.set_ylabel('Frequence')
            ax.set_title('Distribution - Titres')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Longueurs abstracts
        ax = axes[0, 1]
        if 'abstract' in self.df.columns:
            lengths = self.df['abstract'].str.len()
            ax.hist(lengths, bins=30, color=COLORS['secondary'], alpha=0.7, edgecolor='black')
            ax.axvline(lengths.mean(), color=COLORS['danger'], linestyle='--', 
                      linewidth=2, label=f'Moyenne: {lengths.mean():.0f}')
            ax.set_xlabel('Longueur (caracteres)')
            ax.set_ylabel('Frequence')
            ax.set_title('Distribution - Abstracts')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Longueurs body
        ax = axes[1, 0]
        if 'body' in self.df.columns:
            lengths = self.df['body'].str.len()
            ax.hist(lengths, bins=30, color=COLORS['accent'], alpha=0.7, edgecolor='black')
            ax.axvline(lengths.mean(), color=COLORS['danger'], linestyle='--', 
                      linewidth=2, label=f'Moyenne: {lengths.mean():.0f}')
            ax.set_xlabel('Longueur (caracteres)')
            ax.set_ylabel('Frequence')
            ax.set_title('Distribution - Corps')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Statistiques
        ax = axes[1, 1]
        ax.axis('off')
        
        stats_text = f"""
STATISTIQUES DE LONGUEUR

Titres:
  - Moyenne: {text_lengths['title']['mean']:.0f} caracteres
  - Mediane: {text_lengths['title']['median']:.0f}
  - Min:     {text_lengths['title']['min']}
  - Max:     {text_lengths['title']['max']}

Abstracts:
  - Moyenne: {text_lengths['abstract']['mean']:.0f} caracteres
  - Mediane: {text_lengths['abstract']['median']:.0f}
  - Min:     {text_lengths['abstract']['min']}
  - Max:     {text_lengths['abstract']['max']}

Corps:
  - Moyenne: {text_lengths['body']['mean']:.0f} caracteres
  - Mediane: {text_lengths['body']['median']:.0f}
  - Min:     {text_lengths['body']['min']}
  - Max:     {text_lengths['body']['max']}
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', 
               facecolor=COLORS['light'], alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Distribution des longueurs -> {output_path}")
    
    def plot_cluster_boxplots(self, output_path: str):
        """Box plots des scores par cluster"""
        if self.df is None or 'cluster_id' not in self.df.columns:
            print("[WARN] Pas de donnees de clustering disponibles")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Distribution des Scores par Cluster', fontsize=16, fontweight='bold')
        
        clusters = sorted(set(self.df['cluster_id']))
        
        # Score final par cluster
        ax = axes[0, 0]
        data_score = [self.df[self.df['cluster_id'] == c]['score'].values for c in clusters]
        bp = ax.boxplot(data_score, labels=[f'C{c}' for c in clusters], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(COLORS['primary'])
            patch.set_alpha(0.7)
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Score Final')
        ax.set_title('Scores Finaux par Cluster')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Score embedding par cluster
        ax = axes[0, 1]
        if 'score_embed' in self.df.columns:
            data_embed = [self.df[self.df['cluster_id'] == c]['score_embed'].values for c in clusters]
            bp = ax.boxplot(data_embed, labels=[f'C{c}' for c in clusters], patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(COLORS['secondary'])
                patch.set_alpha(0.7)
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('Score Embedding')
            ax.set_title('Scores Embedding par Cluster')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Score BM25 par cluster
        ax = axes[1, 0]
        if 'score_bm25' in self.df.columns:
            data_bm25 = [self.df[self.df['cluster_id'] == c]['score_bm25'].values for c in clusters]
            bp = ax.boxplot(data_bm25, labels=[f'C{c}' for c in clusters], patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(COLORS['accent'])
                patch.set_alpha(0.7)
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('Score BM25')
            ax.set_title('Scores BM25 par Cluster')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Statistiques par cluster
        ax = axes[1, 1]
        ax.axis('off')
        
        cluster_dists = self.report.get('cluster_distributions', {})
        
        stats_text = "STATISTIQUES PAR CLUSTER\n" + "="*40 + "\n\n"
        for cluster_id in clusters:
            if str(cluster_id) in cluster_dists or cluster_id in cluster_dists:
                cdata = cluster_dists.get(str(cluster_id), cluster_dists.get(cluster_id, {}))
                stats_text += f"Cluster {cluster_id}:\n"
                stats_text += f"  Articles: {cdata.get('count', 0)}\n"
                stats_text += f"  Score moy: {cdata.get('score_mean', 0):.4f}\n"
                stats_text += f"  Score std: {cdata.get('score_std', 0):.4f}\n\n"
        
        ax.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', 
               facecolor=COLORS['light'], alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Box plots par cluster -> {output_path}")
    
    def plot_clustering_quality_radar(self, output_path: str):
        """Radar chart des métriques de qualité du clustering"""
        cluster_info = self.report.get('clustering', {})
        
        if not cluster_info:
            print("[WARN] Pas de metriques de clustering disponibles")
            return
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        fig.suptitle('Qualite du Clustering - Radar Chart', fontsize=16, fontweight='bold')
        
        # Métriques (normalisées sur [0, 1])
        metrics = []
        labels = []
        
        # Silhouette (déjà sur [-1, 1], normaliser à [0, 1])
        if 'silhouette' in cluster_info:
            silhouette = cluster_info['silhouette']
            metrics.append((silhouette + 1) / 2)  # [-1,1] -> [0,1]
            labels.append(f'Silhouette\n({silhouette:.3f})')
        
        # Calinski-Harabasz (normaliser avec sigmoid)
        if 'calinski_harabasz' in cluster_info:
            ch = cluster_info['calinski_harabasz']
            ch_norm = 1 / (1 + np.exp(-ch / 50))  # Sigmoid
            metrics.append(ch_norm)
            labels.append(f'Calinski-H\n({ch:.2f})')
        
        # Davies-Bouldin (inverser et normaliser, optimal proche de 0)
        if 'davies_bouldin' in cluster_info:
            db = cluster_info['davies_bouldin']
            db_norm = 1 / (1 + db)  # Plus faible = meilleur
            metrics.append(db_norm)
            labels.append(f'Davies-B\n({db:.3f})')
        
        # Ratio clusters/bruit
        n_clusters = cluster_info.get('n_clusters', 0)
        n_noise = cluster_info.get('n_noise', 0)
        if n_clusters + n_noise > 0:
            cluster_ratio = n_clusters / (n_clusters + n_noise)
            metrics.append(cluster_ratio)
            labels.append(f'Ratio Clusters\n({cluster_ratio:.3f})')
        
        # Diversité (si disponible)
        diversity = self.report.get('stats', {}).get('diversity_cosine_mean')
        if diversity is not None:
            # Normaliser diversité (0.1 = bon, 0.5 = excellent)
            div_norm = min(diversity / 0.5, 1.0)
            metrics.append(div_norm)
            labels.append(f'Diversite\n({diversity:.3f})')
        
        if len(metrics) < 3:
            print("[WARN] Pas assez de metriques pour le radar chart")
            return
        
        # Angles pour chaque métrique
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        metrics += metrics[:1]  # Fermer le polygone
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, metrics, 'o-', linewidth=2, color=COLORS['primary'], label='Score')
        ax.fill(angles, metrics, alpha=0.25, color=COLORS['primary'])
        
        # Zone de référence (bon niveau = 0.7)
        ref_values = [0.7] * len(angles)
        ax.plot(angles, ref_values, '--', linewidth=1.5, color=COLORS['success'], 
               alpha=0.5, label='Reference (0.7)')
        
        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'])
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        # Interprétation
        avg_score = np.mean(metrics[:-1])
        quality_text = "Excellent" if avg_score > 0.8 else "Bon" if avg_score > 0.6 else "Moyen" if avg_score > 0.4 else "Faible"
        
        fig.text(0.5, 0.05, f'Score global: {avg_score:.3f} ({quality_text})', 
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=COLORS['light'], alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Radar chart qualite -> {output_path}")
    
    def plot_score_comparison_table(self, output_path: str):
        """Table comparative des méthodes de scoring"""
        if self.df is None:
            print("[WARN] Pas de CSV disponible")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle('Comparaison des Methodes de Scoring', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Préparer les données
        score_cols = ['score_title', 'score_abstract', 'score_body', 'score_embed', 'score_bm25', 'score']
        available_cols = [col for col in score_cols if col in self.df.columns]
        
        if len(available_cols) < 3:
            print("[WARN] Pas assez de scores disponibles")
            return
        
        # Calculer statistiques
        table_data = []
        for col in available_cols:
            col_name = col.replace('score_', '').replace('score', 'final').title()
            table_data.append([
                col_name,
                f"{self.df[col].min():.4f}",
                f"{self.df[col].max():.4f}",
                f"{self.df[col].mean():.4f}",
                f"{self.df[col].median():.4f}",
                f"{self.df[col].std():.4f}",
                f"{(self.df[col] > self.df[col].median()).sum()}"
            ])
        
        # Créer la table
        table = ax.table(cellText=table_data,
                        colLabels=['Methode', 'Min', 'Max', 'Moyenne', 'Mediane', 'Std', 'N > Mediane'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Styliser
        for i in range(len(available_cols) + 1):
            if i == 0:
                for j in range(7):
                    table[(i, j)].set_facecolor(COLORS['primary'])
                    table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                for j in range(7):
                    if j == 0:
                        table[(i, j)].set_facecolor(COLORS['light'])
                        table[(i, j)].set_text_props(weight='bold')
                    else:
                        table[(i, j)].set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Table comparative -> {output_path}")
    
    def plot_embeddings_3d(self, output_path: str):
        """Projection 3D des embeddings (optionnelle)"""
        embeddings_file = self.report['metadata'].get('embeddings_file')
        
        if not embeddings_file or not os.path.exists(embeddings_file):
            print("[WARN] Fichier embeddings non disponible")
            return
        
        try:
            E = np.load(embeddings_file)
            
            if len(E) < 10:
                print("[WARN] Pas assez d'articles pour projection 3D")
                return
            
            # PCA 3D
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3, random_state=42)
            E_3d = pca.fit_transform(E)
            
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            fig.suptitle('Projection 3D des Embeddings (PCA)', fontsize=16, fontweight='bold')
            
            # Colorer par cluster
            if self.df is not None and 'cluster_id' in self.df.columns:
                clusters = self.df['cluster_id'].values
                unique_clusters = sorted(set(clusters))
                colors = sns.color_palette("husl", len(unique_clusters))
                color_map = {cl: colors[i] for i, cl in enumerate(unique_clusters)}
                
                for cluster_id in unique_clusters:
                    mask = clusters == cluster_id
                    label = f'Cluster {cluster_id}' if cluster_id != -1 else 'Bruit'
                    color = color_map[cluster_id] if cluster_id != -1 else COLORS['neutral']
                    marker = 'o' if cluster_id != -1 else 'x'
                    
                    ax.scatter(E_3d[mask, 0], E_3d[mask, 1], E_3d[mask, 2],
                             c=[color], label=label, s=100, alpha=0.7, 
                             edgecolors='black', linewidth=0.5, marker=marker)
            else:
                ax.scatter(E_3d[:, 0], E_3d[:, 1], E_3d[:, 2], 
                          c=COLORS['primary'], s=100, alpha=0.7)
            
            ax.set_xlabel('PC1 ({:.1f}%)'.format(pca.explained_variance_ratio_[0] * 100))
            ax.set_ylabel('PC2 ({:.1f}%)'.format(pca.explained_variance_ratio_[1] * 100))
            ax.set_zlabel('PC3 ({:.1f}%)'.format(pca.explained_variance_ratio_[2] * 100))
            ax.legend(loc='best', ncol=2)
            
            total_var = pca.explained_variance_ratio_.sum() * 100
            ax.set_title(f'Variance expliquee: {total_var:.1f}%', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OK] Projection 3D -> {output_path}")
            
        except Exception as e:
            print(f"[WARN] Erreur projection 3D: {e}")
    
    def generate_all(self, output_dir: str = "visualizations"):
        """Génère toutes les visualisations"""
        Path(output_dir).mkdir(exist_ok=True)
        
        print("\n" + "="*70)
        print("  GENERATION DES VISUALISATIONS")
        print("="*70 + "\n")
        
        # Visualisations de base (6)
        self.plot_score_distributions(os.path.join(output_dir, "01_score_distributions.png"))
        self.plot_threshold_analysis(os.path.join(output_dir, "02_threshold_analysis.png"))
        self.plot_pipeline_flow(os.path.join(output_dir, "03_pipeline_flow.png"))
        self.plot_clusters_2d(os.path.join(output_dir, "04_clusters_2d.png"))
        self.plot_top_articles(os.path.join(output_dir, "05_top_articles.png"))
        self.plot_score_correlation(os.path.join(output_dir, "06_score_correlation.png"))
        
        # Nouvelles visualisations enrichies (6)
        self.plot_similarity_heatmap(os.path.join(output_dir, "07_similarity_heatmap.png"))
        self.plot_text_length_distributions(os.path.join(output_dir, "08_text_lengths.png"))
        self.plot_cluster_boxplots(os.path.join(output_dir, "09_cluster_boxplots.png"))
        self.plot_clustering_quality_radar(os.path.join(output_dir, "10_quality_radar.png"))
        self.plot_score_comparison_table(os.path.join(output_dir, "11_score_table.png"))
        self.plot_embeddings_3d(os.path.join(output_dir, "12_embeddings_3d.png"))
        
        print("\n" + "="*70)
        print(f"  12 VISUALISATIONS GENEREES DANS: {output_dir}/")
        print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Visualisations pour le pipeline ameliore")
    parser.add_argument("--report", type=str, default="articles_report.json",
                       help="Chemin vers le rapport JSON")
    parser.add_argument("--csv", type=str, default="articles_final.csv",
                       help="Chemin vers le CSV final")
    parser.add_argument("--output", type=str, default="visualizations",
                       help="Repertoire de sortie")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.report):
        print(f"[ERREUR] Fichier rapport introuvable: {args.report}")
        print("Executez d'abord le pipeline avec run_improved.py")
        return 1
    
    csv_path = args.csv if os.path.exists(args.csv) else None
    
    visualizer = PipelineVisualizer(args.report, csv_path)
    visualizer.generate_all(args.output)
    
    return 0

if __name__ == "__main__":
    exit(main())

