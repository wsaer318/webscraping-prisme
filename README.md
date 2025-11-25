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

# Note : Sur Windows, si vous rencontrez des problèmes avec PyTorch :
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Installation rapide (dépendances minimales)

```bash
pip install numpy pandas scipy scikit-learn torch sentence-transformers rank-bm25 langdetect ftfy
```

---

## 💻 Utilisation

### Génération de données de test

```bash
python generate_data.py --n-pos 200 --n-neg 150 --seed 42 --out data/articles_fictifs.csv
```

### Exécution du pipeline

#### Méthode simple (configuration par défaut)

```bash
python process_improved.py
```

Le pipeline utilisera les paramètres par défaut :
- Requête principale : "l'effet de la lumière sur le comportement des chats"
- Fichier d'entrée : `data/articles_fictifs.csv`
- Fichier de sortie : `data/articles_final.csv`
- Rapport : `articles_report.json`

#### Méthode avancée (configuration personnalisée)

```python
from process_improved import Config, main

# Créer une configuration personnalisée
config = Config(
    query_main="intelligence artificielle machine learning",
    input_csv="data/articles_fictifs.csv",
    output_csv="data/articles_final.csv",
    report_json="articles_report.json",
    threshold_method="ensemble",
    cluster_method="hdbscan",  # ou "dbscan", "graph_cc"
    mmr_topk=50,
    mmr_lambda=0.7,
    fusion_method="rrf",  # ou "linear_z", "rank_pct"
    batch_size=16,
    use_gpu=False,  # Mettre à True si GPU disponible
)

# Exécuter le pipeline
report = main(config)
```

### Génération des visualisations

```bash
python generate_visualizations.py
```

Ou avec des chemins personnalisés :

```bash
python visualize.py --report articles_report.json --csv data/articles_final.csv --output visualizations
```

Les visualisations seront générées dans le dossier `visualizations/`.

---

## 📁 Structure du Projet

```
projet_filtre/
├── 📄 README.md                          # Documentation principale
├── 📄 requirements_improved.txt          # Dépendances Python
├── 📄 .gitignore                         # Fichiers ignorés par Git
├── 🐍 generate_data.py                   # Générateur d'articles fictifs pour tests
├── 🐍 process_improved.py                # Pipeline principal (cœur du système, ~1850 lignes)
├── 🐍 generate_visualizations.py         # Script wrapper pour générer les graphiques
├── 🐍 visualize.py                       # Module de visualisation (classe PipelineVisualizer)
├── 📁 data/                             # Données du projet
│   ├── 📊 articles_fictifs.csv           # Données d'entrée (générées par generate_data.py)
│   └── 📊 articles_final.csv             # Résultats finaux du pipeline
├── 📁 .cache_embeddings/                # Cache des embeddings (créé automatiquement)
├── 📊 articles_final_embeddings.npy      # Embeddings sauvegardés des articles sélectionnés
├── 📋 articles_report.json               # Rapport détaillé JSON avec métriques
├── 📝 pipeline.log                       # Logs d'exécution structurés
└── 📁 visualizations/                    # Graphiques générés
    ├── 01_score_distributions.png       # Distribution des scores (embedding, BM25, final)
    ├── 02_threshold_analysis.png         # Analyse du seuillage automatique
    ├── 03_pipeline_flow.png              # Diagramme de flux (entonnoir de filtrage)
    ├── 04_clusters_2d.png                # Projection t-SNE des clusters
    ├── 05_top_articles.png               # Top articles par score
    ├── 06_score_correlation.png          # Matrice de corrélation entre scores
    ├── 07_similarity_heatmap.png         # Heatmap de similarité entre articles
    ├── 08_text_lengths.png                # Distribution des longueurs de texte
    ├── 09_cluster_boxplots.png           # Boxplots des scores par cluster
    ├── 10_quality_radar.png              # Radar chart des métriques de qualité
    ├── 11_score_table.png                # Table comparative des méthodes de scoring
    ├── 12_embeddings_3d.png              # Projection 3D des embeddings (PCA)
    └── README.md                         # Documentation des visualisations
```

### Fichiers générés automatiquement

Lors de l'exécution du pipeline, les fichiers suivants sont créés :
- `articles_report.json` : Rapport complet avec statistiques, métriques et diagnostics
- `data/articles_final.csv` : Articles sélectionnés avec scores détaillés
- `articles_final_embeddings.npy` : Embeddings des articles finaux (pour visualisation 3D)
- `pipeline.log` : Logs d'exécution (format structuré)
- `.cache_embeddings/` : Cache des embeddings pour éviter les recalculs

---

## ⚙️ Configuration

Le pipeline est hautement configurable via la classe `Config` dans `process_improved.py` :

### Paramètres principaux

| Paramètre | Description | Valeurs possibles | Défaut |
|-----------|-------------|-------------------|--------|
| `query_main` | Requête de recherche principale | string | `"l'effet de la lumière sur le comportement des chats"` |
| `input_csv` | Fichier CSV d'entrée | chemin relatif/absolu | `"data/articles_fictifs.csv"` |
| `output_csv` | Fichier CSV de sortie | chemin relatif/absolu | `"data/articles_final.csv"` |
| `threshold_method` | Méthode de seuillage | `"ensemble"`, `"gmm"`, `"kde"`, `"otsu"` | `"ensemble"` |
| `cluster_method` | Algorithme de clustering | `"hdbscan"`, `"dbscan"`, `"graph_cc"` | `"hdbscan"` |
| `fusion_method` | Méthode de fusion BM25/embedding | `"rrf"`, `"linear_z"`, `"rank_pct"` | `"rrf"` |
| `mmr_topk` | Nombre d'articles finaux | int | `50` |
| `mmr_lambda` | Balance pertinence/diversité | 0.0-1.0 | `0.7` |
| `min_abstract_len` | Longueur minimale d'abstract | int | `30` |
| `dedup_threshold` | Seuil de déduplication | 0.0-1.0 | `0.985` |
| `batch_size` | Taille de batch pour embeddings | int | `16` |
| `use_gpu` | Utiliser GPU si disponible | bool | `False` |

### Paramètres avancés

```python
config = Config(
    # Modèle d'embeddings
    model_id="intfloat/multilingual-e5-small",  # Modèle Sentence-BERT
    
    # Pooling du body (longs textes)
    body_pooling="attn",  # "attn" (attention query-aware) ou "maxmean"
    body_chunk_size=600,  # Taille des chunks
    body_chunk_stride=400,  # Pas de fenêtre glissante
    
    # Poids pour scoring multi-champs
    w_title=0.5,    # Poids titre
    w_abs=0.3,      # Poids abstract
    w_body=0.2,     # Poids body
    
    # Fusion des scores
    fusion_bm25_weight=0.3,  # Poids BM25
    fusion_embed_weight=0.7, # Poids embeddings
    
    # Clustering HDBSCAN
    hdbscan_min_cluster_size=5,
    hdbscan_min_samples=2,
    hdbscan_cluster_selection_method="eom",  # "eom" ou "leaf"
    
    # Sécurité
    max_text_len=1_000_000,  # Limite contre attaques DoS
    allowed_langs=("fr", "en"),  # Langues acceptées
)
```

### Exemple de configuration personnalisée

```python
from process_improved import Config, main

config = Config(
    query_main="machine learning deep learning neural networks",
    input_csv="data/mes_articles.csv",
    output_csv="data/resultats.csv",
    threshold_method="gmm",  # Utiliser GMM au lieu d'ensemble
    cluster_method="dbscan",  # DBSCAN classique
    mmr_topk=100,  # Sélectionner 100 articles
    mmr_lambda=0.6,  # Plus de diversité (lambda plus bas)
    use_gpu=True,  # Accélérer avec GPU
    batch_size=32,  # Batch plus grand si GPU disponible
)

report = main(config)
```

---

## 📊 Résultats et Visualisations

### Fichiers de sortie

#### `data/articles_final.csv`
Articles sélectionnés avec scores et métadonnées :
- Colonnes originales : `url`, `title`, `abstract`, `body`, `lang_hint`, `author`, `journal`, `published_at`, `doi`, `quality_type`
- Scores calculés :
  - `score_title` : Score de similarité du titre
  - `score_abstract` : Score de similarité de l'abstract
  - `score_body` : Score de similarité du corps
  - `score_embed` : Score embedding combiné (pondéré)
  - `score_bm25` : Score BM25 lexical
  - `score` : Score final après fusion
- Métadonnées de traitement :
  - `cluster_id` : ID du cluster (ou -1 pour bruit)
  - `rank` : Rang final après sélection MMR

#### `articles_report.json`
Rapport détaillé incluant :
- **Métadonnées** : Version, timestamps, durée d'exécution
- **Configuration** : Tous les paramètres utilisés
- **Compteurs** : Nombre d'articles à chaque étape du pipeline
- **Seuils** : Méthode utilisée, valeur, métadonnées (pour ensemble : poids de chaque méthode)
- **Clustering** : Métriques (silhouette, Calinski-Harabasz, Davies-Bouldin), nombre de clusters
- **Sélection** : Quotas par cluster, statistiques, méthode MMR/Facility Location
- **Statistiques** : Min/max/moyenne/std des scores, diversité cosine
- **Longueurs de texte** : Statistiques pour titre, abstract, body
- **Distributions par cluster** : Scores moyens par cluster
- **Matrice de similarité** : Similarité entre articles sélectionnés (si < 100 articles)

#### `articles_final_embeddings.npy`
Tableau NumPy (N, D) contenant les embeddings L2-normalisés des articles sélectionnés. Utilisé pour la visualisation 3D.

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

## 🐛 Dépannage

### Problèmes courants

**Erreur : "HDBSCAN non disponible"**
```bash
pip install hdbscan
```

**Erreur avec torch sur Windows**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Mémoire insuffisante**
- Réduisez `batch_size` dans la config (par exemple `8` au lieu de `16`)
- Traitez les données par lots en divisant le CSV d'entrée

**Cache des embeddings corrompu**
```bash
# Supprimer le cache (sera régénéré automatiquement)
rm -rf .cache_embeddings  # Linux/Mac
rmdir /s .cache_embeddings  # Windows PowerShell
```

**Visualisations manquantes**
- Vérifiez que `articles_report.json` et `data/articles_final.csv` existent
- Exécutez d'abord le pipeline : `python process_improved.py`
- Puis générez les visualisations : `python generate_visualizations.py`

---

## 📧 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub.

---

**Développé avec ❤️ pour la recherche scientifique**