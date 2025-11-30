# 🛠️ Documentation Technique - PRISMA Review Manager

## 🏗️ Architecture du Projet

Le projet est une application web **Streamlit** structurée en plusieurs pages, s'appuyant sur une base de données **SQLite** et des modules Python pour la logique métier (Scraping, NLP, Analytics).

### Structure des Dossiers

```
projet_prisma/
├── app.py                      # Point d'entrée (Dashboard)
├── pages/                      # Pages de l'application Streamlit
│   ├── 1_Recherche.py          # Interface de collecte (Scraping)
│   ├── 2_Screening.py          # Interface de tri (Titre/Abstract)
│   ├── 3_Eligibilite.py        # Interface de revue (Full Text)
│   ├── 4_Analyse.py            # Dashboard analytique & Reporting
│   └── 0_Base_de_donnees.py    # Explorateur de données brut
├── src/                        # Logique métier (Package Python)
│   ├── collection/             # Modules de scraping (arXiv, PubMed, etc.)
│   ├── database.py             # Modèles SQLAlchemy & Connexion DB
│   ├── advanced_sorting.py     # Moteur de ranking IA (Embeddings)
│   ├── concept_filter.py       # Moteur de filtrage par mots-clés
│   ├── pdf_retriever.py        # Téléchargement automatique de PDFs
│   ├── analytics.py            # Calcul des statistiques
│   ├── exporters.py            # Export CSV/Excel/BibTeX
│   └── ui_utils.py             # Utilitaires UI (CSS Premium)
├── data/                       # Stockage des données
│   ├── prisma.db               # Base de données SQLite
│   └── pdfs/                   # Stockage des fichiers PDF
├── static/                     # Ressources statiques
│   └── styles/                 # Fichiers CSS
└── requirements.txt            # Dépendances du projet
```

---

## 🗄️ Schéma de Base de Données

L'application utilise **SQLAlchemy** (ORM) avec **SQLite**.

### 1. `SearchSession` (Sessions de recherche)
Regroupe les articles importés lors d'une même opération de recherche.
- `id` (PK): Identifiant unique
- `query`: Requête utilisée
- `created_at`: Date de création
- `num_results`: Nombre d'articles trouvés
- `status`: État de la session (ACTIVE, ARCHIVED)

### 2. `Article` (Table principale)
Contient toutes les métadonnées et l'état de chaque article.
- **Identification**
  - `id` (PK), `title`, `authors`, `year`, `source`, `doi`, `link`
- **Contenu**
  - `abstract`: Résumé
  - `full_text`: Texte complet extrait
  - `pdf_path`: Chemin local du fichier PDF
- **Statut PRISMA (`status`)**
  - `IDENTIFIED`: Importé brut
  - `EXCLUDED_SEMANTIC_FILTER`: Rejeté par le pré-tri sémantique
  - `SCREENED_IN`: Retenu après lecture Titre/Abstract
  - `EXCLUDED_SCREENING`: Rejeté après lecture Titre/Abstract
  - `EXCLUDED_ELIGIBILITY`: Rejeté après lecture Texte Complet
  - `INCLUDED`: Inclus dans la revue finale
- **Analyse IA**
  - `relevance_score`: Score de pertinence (0-1) calculé par Cross-Encoder
  - `suggested_reason`: Justification suggérée par l'IA
  - `ia_metadata`: Détails techniques (JSON)

### 3. `ArticleHistory` (Traçabilité)
Enregistre chaque changement d'état pour l'audit.
- `article_id` (FK), `previous_status`, `new_status`, `timestamp`, `user`

### 4. `ExclusionCriteria` & `EligibilityCriteria`
Critères configurables pour justifier les exclusions.

---

## 🧠 Moteurs d'Analyse (IA & NLP)

### 1. Ranking Sémantique (`src.advanced_sorting`)
Utilise `sentence-transformers` pour trier les articles par pertinence.
- **Modèle Bi-Encoder** (`paraphrase-MiniLM-L3-v2`): Pour l'encodage rapide des vecteurs.
- **Cross-Encoder** (optionnel): Pour le re-ranking précis.

### 2. Filtrage par Concepts (`src.concept_filter`)
Permet de filtrer les articles contenant des mots-clés spécifiques.
- Supporte les opérateurs booléens (AND/OR).
- Recherche dans le Titre, l'Abstract et le Full Text (via chunking).

### 3. Extraction de PDF (`src.pdf_retriever`)
- Tente de télécharger le PDF via `Unpaywall` (API gratuite) ou `ArXiv`.
- Utilise `PyMuPDF` (fitz) pour extraire le texte brut du PDF pour l'analyse.

---

## 💻 Guide de Développement

### Installation de l'environnement
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Ajouter une nouvelle page
1. Créer un fichier `pages/X_NomPage.py`.
2. Importer `st` et `load_premium_css`.
3. Appeler `load_premium_css()` au début.

### Modifier le modèle de données
1. Éditer `src/database.py`.
2. **Attention**: SQLite ne supporte pas bien les migrations `ALTER TABLE`. Pour des changements majeurs, il est souvent plus simple de supprimer `prisma.db` (si en dev) ou d'utiliser un script de migration manuel (créer nouvelle table, copier données, renommer).
