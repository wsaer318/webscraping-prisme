# Visualisations du Pipeline Amélioré

Ce dossier contient les visualisations générées automatiquement après l'exécution du pipeline de tri d'articles scientifiques.

## 📊 Graphiques Disponibles

### 1. Distribution des Scores (`01_score_distributions.png`)
**Objectif** : Analyser la distribution des différents types de scores calculés par le pipeline.

**Contenu** :
- **Score Embedding** : Distribution des scores de similarité sémantique (0-1)
- **Score BM25** : Distribution des scores lexicaux (0-∞)
- **Score Final** : Distribution des scores fusionnés après RRF/linear_z/rank_pct
- **Statistiques** : Tableau récapitulatif (min, max, moyenne, médiane, écart-type)

**Interprétation** :
- Une distribution bimodale indique une bonne séparation entre articles pertinents et non-pertinents
- La moyenne du score final doit être proche du seuil calculé
- Un écart-type élevé suggère une forte variabilité dans la pertinence

---

### 2. Analyse du Seuillage (`02_threshold_analysis.png`)
**Objectif** : Visualiser l'efficacité du seuillage automatique et la répartition des articles.

**Contenu** :
- **Histogramme coloré** : Zone rouge (rejetés) vs zone verte (acceptés)
- **Ligne de seuil** : Valeur calculée par la méthode ensemble
- **Statistiques de filtrage** : Nombre d'articles à chaque étape
- **Détails de la méthode** : Pondérations et contributions de chaque méthode (GMM, KDE, Otsu, Jenks)

**Interprétation** :
- Le seuil doit idéalement se situer dans la vallée entre deux modes
- Le taux de rétention indique le niveau de sélectivité du pipeline
- Les méthodes avec un poids élevé ont plus d'influence sur le seuil final

---

### 3. Flux du Pipeline (`03_pipeline_flow.png`)
**Objectif** : Visualiser l'entonnoir de filtrage avec les pertes à chaque étape.

**Contenu** :
- **Diagramme de Sankey** : Largeur proportionnelle au nombre d'articles
- **Étapes du pipeline** :
  1. Articles initiaux (corpus brut)
  2. Filtres langue/longueur
  3. Déduplication
  4. Seuillage automatique
  5. Sélection finale diversifiée
- **Pertes annotées** : Nombre et pourcentage d'articles rejetés à chaque étape

**Interprétation** :
- Un taux de rétention global < 50% peut indiquer un filtrage trop strict
- Les plus grosses pertes devraient être au seuillage (filtrage pertinence)
- Le taux de rétention final indique l'efficacité du pipeline

---

### 4. Visualisation des Clusters (`04_clusters_2d.png`)
**Objectif** : Représenter les articles dans un espace 2D pour visualiser les groupes thématiques.

**Contenu** :
- **Projection t-SNE** : Réduction dimensionnelle des embeddings en 2D
- **Clusters colorés** : Chaque cluster a une couleur unique
- **Points de bruit** : Marqués avec un 'x' en gris (cluster_id = -1)
- **Statistiques** : Qualité du clustering (silhouette, Calinski-Harabasz, Davies-Bouldin)
- **Répartition** : Nombre d'articles par cluster

**Interprétation** :
- Des clusters bien séparés indiquent des thèmes distincts
- Un silhouette score > 0.7 indique un excellent clustering
- Beaucoup de points de bruit (-1) peut suggérer un epsilon trop faible pour DBSCAN

**Métriques de qualité** :
- **Silhouette** : [-1, 1], optimal > 0.7
- **Calinski-Harabasz** : Plus élevé = meilleur (pas de borne supérieure)
- **Davies-Bouldin** : Plus faible = meilleur, optimal < 1.0

---

### 5. Top Articles (`05_top_articles.png`)
**Objectif** : Afficher les articles les plus pertinents sélectionnés par le pipeline.

**Contenu** :
- **Barres horizontales** : Top 10 articles par score décroissant
- **Couleurs** : Codage par cluster d'appartenance
- **Titres** : Tronqués à 60 caractères pour lisibilité
- **Scores annotés** : Valeur exacte du score final

**Interprétation** :
- L'écart entre le 1er et le 10e indique la concentration de la pertinence
- Des articles du même cluster en haut suggèrent une forte cohérence thématique
- Un bon pipeline devrait avoir des scores > seuil pour tous les articles affichés

---

### 6. Corrélations entre Scores (`06_score_correlation.png`)
**Objectif** : Analyser les relations entre les différentes composantes de scoring.

**Contenu** :
- **Matrice de corrélation** : Heatmap avec coefficients de Pearson
- **Corrélations analysées** :
  - score_title vs score_abstract
  - score_embed vs score_bm25
  - score_final vs composantes individuelles

**Interprétation** :
- **Corrélation élevée (> 0.8)** : Les deux scores capturent des informations similaires
- **Corrélation faible (< 0.3)** : Les scores sont complémentaires (bon pour la fusion)
- **Corrélation négative** : Conflit potentiel entre métriques (rare)

**Valeurs attendues** :
- `score_embed` ≈ `score_final` : Dominance de la similarité sémantique
- `score_bm25` vs `score_embed` : Modérée (0.3-0.6) → complémentarité
- `score_title` vs `score_abstract` : Élevée (0.6-0.8) → cohérence

---

## 🔄 Régénération des Visualisations

Pour régénérer les visualisations après une nouvelle exécution du pipeline :

```bash
python generate_visualizations.py
```

Ou avec des chemins personnalisés :

```bash
python visualize.py --report articles_report.json --csv articles_final.csv --output visualizations
```

---

## 🛠️ Dépendances Requises

Les visualisations nécessitent :
```bash
pip install matplotlib seaborn scikit-learn
```

Ou installation complète :
```bash
pip install -r requirements_improved.txt
```

---

## 📈 Conseils d'Interprétation

### Signes d'un pipeline performant :
✅ Distribution bimodale des scores finaux  
✅ Silhouette score > 0.7  
✅ Taux de rétention entre 20% et 50%  
✅ Corrélation modérée entre BM25 et embedding (0.3-0.6)  
✅ Clusters bien séparés dans t-SNE  

### Signaux d'alerte :
⚠️ Distribution unimodale → seuil mal calibré  
⚠️ Silhouette score < 0.5 → clustering faible  
⚠️ Taux de rétention < 10% → filtrage trop strict  
⚠️ Corrélation BM25/embedding > 0.9 → redondance  
⚠️ Tous les articles dans un seul cluster → epsilon trop large  

---

## 📝 Personnalisation

Pour personnaliser les visualisations, modifiez les paramètres dans `visualize.py` :

- **Nombre d'articles top** : `plot_top_articles(top_n=20)`
- **Perplexité t-SNE** : Ajustez la ligne `perplexity = min(30, len(X_proxy) - 1)`
- **Couleurs** : Modifiez le dictionnaire `COLORS` en début de fichier
- **Taille des figures** : `plt.rcParams['figure.figsize'] = (14, 10)`

---

## 📊 Export des Figures

Toutes les figures sont exportées en PNG haute résolution (300 DPI) pour :
- Inclusion dans des rapports
- Présentations
- Publications scientifiques

Pour changer le format d'export, modifiez dans `visualize.py` :
```python
plt.savefig(output_path, dpi=300, format='pdf')  # PDF au lieu de PNG
```

---

*Généré automatiquement par le Pipeline Amélioré de Tri d'Articles Scientifiques*

