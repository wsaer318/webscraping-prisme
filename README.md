# 🔬 Pipeline Avancé de Tri d'Articles Scientifiques

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://github.com/psf/black)

Un système intelligent et robuste de filtrage, tri et sélection d'articles scientifiques utilisant l'apprentissage automatique, le traitement du langage naturel (NLP) et des techniques statistiques avancées.

---

## 📋 Table des Matières

- [Vue d'ensemble](#-vue-densemble)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du Projet](#-structure-du-projet)
- [Configuration](#-configuration)
- [Résultats et Visualisations](#-résultats-et-visualisations)
- [Technologies Utilisées](#-technologies-utilisées)
- [Contribution](#-contribution)
- [License](#-license)

---

## 🎯 Vue d'ensemble

Ce pipeline implémente un système de bout-en-bout pour :
- **Filtrer** et nettoyer des articles scientifiques
- **Scorer** leur pertinence par rapport à une requête
- **Détecter** les doublons et quasi-doublons
- **Clusteriser** les articles par similarité sémantique
- **Sélectionner** les meilleurs articles en optimisant la diversité et la pertinence
- **Visualiser** les résultats avec des graphiques interactifs

Le système est conçu pour être **robuste**, **sécurisé** et **scalable**, avec une attention particulière portée à la validation statistique et à la qualité du code.

---

## ✨ Fonctionnalités

### 🔍 Analyse et Filtrage
- **Nettoyage robuste** : Sanitisation HTML, normalisation Unicode, détection de langue
- **Détection de doublons** : Identification des doublons exacts et quasi-doublons par hashing et similarité
- **Validation de sécurité** : Protection contre XSS, injection SQL, attaques DoS

### 📊 Scoring Multi-Critères
- **BM25** : Score de pertinence lexicale
- **Embeddings sémantiques** : Similarité cosinus avec Sentence-BERT
- **Scores combinés** : Z-scores robustes utilisant la déviation médiane absolue (MAD)

### 🎲 Clustering Intelligent
- **DBSCAN** : Clustering avec epsilon adaptatif
- **HDBSCAN** : Clustering hiérarchique robuste (optionnel)
- **Métriques de qualité** : Silhouette, Calinski-Harabasz, Davies-Bouldin

### 🎯 Sélection Optimale
- **Seuillage multi-méthode** : GMM, KDE, Otsu, méthode d'ensemble
- **MMR (Maximal Marginal Relevance)** : Optimisation pertinence/diversité
- **Facility Location** : Sélection submodulaire pour représentativité maximale

### 📈 Visualisations
- Distributions des scores
- Analyse des seuils
- Diagrammes de flux (pipeline)
- Projections 2D/3D des embeddings
- Heatmaps de similarité
- Graphiques radar de qualité

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Articles CSV   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │ ← Nettoyage, normalisation, validation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │ ← Sentence-BERT (modèle multilingue)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Scoring     │ ← BM25 + Similarité sémantique
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Deduplication  │ ← Détection doublons/quasi-doublons
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Clustering    │ ← DBSCAN/HDBSCAN
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Thresholding   │ ← Seuillage adaptatif multi-méthode
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      MMR        │ ← Sélection finale optimale
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Résultats    │ ← CSV + JSON + Visualisations
└─────────────────┘
```

---

## 🚀 Installation

### Prérequis
- **Python** : 3.8 ou supérieur
- **RAM** : Minimum 8 GB (16 GB recommandé pour gros corpus)
- **Système** : Windows 10+, Linux, macOS

### Installation des dépendances

```bash
# Créer un environnement virtuel (recommandé)
python -m venv venv

# Activer l'environnement
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# Installer les dépendances
pip install -r requirements_improved.txt
```

### Installation rapide (dépendances minimales)

```bash
pip install numpy pandas scipy scikit-learn torch sentence-transformers rank-bm25 langdetect ftfy
```

---

## 💻 Utilisation

### Génération de données de test

```bash
python generate_data.py --n-pos 200 --n-neg 150 --seed 42 --out articles_fictifs.csv
```

### Exécution du pipeline

```python
from process_improved import ArticlePipeline, PipelineConfig

# Configuration
config = PipelineConfig(
    query_main="intelligence artificielle machine learning",
    threshold_method="ensemble",
    cluster_method="dbscan",
    mmr_topk=50,
    mmr_lambda=0.7
)

# Initialisation et exécution
pipeline = ArticlePipeline(config)
pipeline.load_data("articles_fictifs.csv")
df_final = pipeline.run()

# Sauvegarde des résultats
pipeline.save_results("articles_final.csv", "articles_report.json")
```

### Génération des visualisations

```bash
python generate_visualizations.py
```

Les visualisations seront générées dans le dossier `visualizations/`.

---

## 📁 Structure du Projet

```
projet_filtre/
├── 📄 README.md                          # Documentation principale
├── 📄 requirements_improved.txt          # Dépendances Python
├── 🐍 generate_data.py                   # Générateur d'articles fictifs
├── 🐍 process_improved.py                # Pipeline principal (cœur du système)
├── 🐍 generate_visualizations.py         # Script de génération de graphiques
├── 🐍 visualize.py                       # Utilitaires de visualisation
├── 📊 articles_fictifs.csv               # Données d'entrée (exemple)
├── 📊 articles_final.csv                 # Résultats finaux
├── 📊 articles_final_embeddings.npy      # Embeddings sauvegardés
├── 📋 articles_report.json               # Rapport détaillé JSON
├── 📝 pipeline.log                       # Logs d'exécution
└── 📁 visualizations/                    # Graphiques générés
    ├── 01_score_distributions.png
    ├── 02_threshold_analysis.png
    ├── 03_pipeline_flow.png
    ├── 04_clusters_2d.png
    ├── 05_top_articles.png
    ├── 06_score_correlation.png
    ├── 07_similarity_heatmap.png
    ├── 08_text_lengths.png
    ├── 09_cluster_boxplots.png
    ├── 10_quality_radar.png
    ├── 11_score_table.png
    ├── 12_embeddings_3d.png
    └── README.md
```

---

## ⚙️ Configuration

Le pipeline est hautement configurable via la classe `PipelineConfig` :

### Paramètres principaux

| Paramètre | Description | Valeurs | Défaut |
|-----------|-------------|---------|--------|
| `query_main` | Requête de recherche principale | string | `""` |
| `threshold_method` | Méthode de seuillage | `"ensemble"`, `"gmm"`, `"kde"`, `"otsu"` | `"ensemble"` |
| `cluster_method` | Algorithme de clustering | `"dbscan"`, `"hdbscan"` | `"dbscan"` |
| `mmr_topk` | Nombre d'articles finaux | int | `50` |
| `mmr_lambda` | Balance pertinence/diversité | 0.0-1.0 | `0.7` |
| `min_abstract_len` | Longueur minimale d'abstract | int | `50` |
| `dedup_threshold` | Seuil de déduplication | 0.0-1.0 | `0.95` |

### Limites de sécurité

```python
config.max_text_len = 50000        # Limite contre attaques DoS
config.max_embedding_batch = 256   # Taille de batch pour embeddings
config.sanitize_html = True        # Nettoyage HTML actif
```

---

## 📊 Résultats et Visualisations

### Fichiers de sortie

#### `articles_final.csv`
Articles sélectionnés avec scores et métadonnées :
- `url`, `title`, `abstract`, `body`
- `bm25_score`, `semantic_score`, `combined_score`
- `cluster_id`, `is_cluster_rep`
- `mmr_score`, `rank`

#### `articles_report.json`
Rapport détaillé incluant :
- Statistiques globales
- Métriques de clustering
- Analyse de sensibilité
- Diagnostics statistiques
- Logs d'exécution

### Visualisations disponibles

1. **Distributions des scores** : Histogrammes BM25, sémantique, combiné
2. **Analyse des seuils** : Méthodes de seuillage comparées
3. **Pipeline flow** : Diagramme de flux Sankey
4. **Clusters 2D** : Projection t-SNE/UMAP
5. **Top articles** : Barres horizontales des meilleurs scores
6. **Corrélations** : Scatter plots entre scores
7. **Heatmap de similarité** : Matrice de similarité sémantique
8. **Longueurs de texte** : Distribution des longueurs
9. **Boxplots par cluster** : Scores par cluster
10. **Radar de qualité** : Métriques multidimensionnelles
11. **Table de scores** : Tableau formaté
12. **Embeddings 3D** : Visualisation interactive 3D

---

## 🛠️ Technologies Utilisées

### Machine Learning & NLP
- **PyTorch** : Framework deep learning
- **Sentence-Transformers** : Embeddings sémantiques (BERT multilingue)
- **scikit-learn** : Clustering, métriques, preprocessing
- **rank-bm25** : Algorithme BM25 pour scoring lexical

### Traitement de Données
- **NumPy** : Calculs numériques optimisés
- **Pandas** : Manipulation de données tabulaires
- **SciPy** : Statistiques avancées

### Visualisation
- **Matplotlib** : Graphiques statiques
- **Seaborn** : Visualisations statistiques

### Utilitaires
- **ftfy** : Correction d'encodage Unicode
- **langdetect** : Détection automatique de langue
- **HDBSCAN** : Clustering hiérarchique (optionnel)

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez le projet
2. Créez une branche feature (`git checkout -b feature/amelioration`)
3. Committez vos changements (`git commit -m 'Ajout nouvelle fonctionnalité'`)
4. Poussez vers la branche (`git push origin feature/amelioration`)
5. Ouvrez une Pull Request

### Standards de code
- **Style** : Black (formatage automatique)
- **Linting** : Ruff
- **Type hints** : mypy
- **Tests** : pytest (couverture > 80%)

---

## 📝 License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 📧 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub.

---

**Développé avec ❤️ pour la recherche scientifique**