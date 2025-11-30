# Documentation Technique - PRISMA Review Manager

## Architecture du Projet

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
│   ├── concept_filter.py       # Moteur de filtrage sémantique
│   ├── enrichment.py           # Enrichissement citations (Semantic Scholar)
│   ├── pdf_retriever.py        # Téléchargement automatique de PDFs
│   ├── analytics.py            # Calcul des statistiques
│   ├── exporters.py            # Export CSV/Excel/BibTeX
│   ├── llm_generator.py        # Génération critères IA
│   └── ui_utils.py             # Utilitaires UI (CSS Premium)
├── data/                       # Stockage des données
│   ├── prisma.db               # Base de données SQLite
│   └── 0_raw/pdfs/             # Stockage des fichiers PDF par session
├── static/                     # Ressources statiques
│   └── styles/                 # Fichiers CSS
├── scripts/                    # Scripts utilitaires et migrations
└── requirements.txt            # Dépendances du projet
```

---

## Schéma de Base de Données

L'application utilise **SQLAlchemy** (ORM) avec **SQLite**.

### 1. `SearchSession` (Sessions de recherche)
Regroupe les articles importés lors d'une même opération de recherche.
- `id` (PK): Identifiant unique
- `query`: Requête utilisée
- `created_at`: Date de création
- `num_results`: Nombre d'articles trouvés
- `successful_downloads`: Nombre de PDFs téléchargés
- `status`: État de la session (ACTIVE, ARCHIVED)

### 2. `Article` (Table principale)
Contient toutes les métadonnées et l'état de chaque article.

#### Identification
- `id` (PK), `title`, `authors`, `year`, `source`, `link`
- `doi`: Digital Object Identifier
- `arxiv_id`: Identifiant arXiv (si applicable)

#### Contenu
- `abstract`: Résumé
- `full_text`: Texte complet extrait
- `pdf_path`: Chemin local du fichier PDF
- `text_extraction_status`: Statut de l'extraction (SUCCESS, FAILED, NOT_ATTEMPTED)
- `extraction_method`: Méthode utilisée (pymupdf, ocr, etc.)

#### Statut PRISMA (`status`)
- `IDENTIFIED`: Importé brut (Étape 1)
- `EXCLUDED_SEMANTIC_FILTER`: Rejeté par le pré-tri sémantique
- `SCREENED_IN`: Retenu après lecture Titre/Abstract (Étape 2)
- `EXCLUDED_SCREENING`: Rejeté après screening
- `ELIGIBLE`: Éligible après revue texte complet (Étape 3)
- `EXCLUDED_ELIGIBILITY`: Rejeté après revue texte complet
- `INCLUDED`: Inclus dans la revue finale (Étape 4)

#### Analyse IA
- `relevance_score`: Score de pertinence (0-1) calculé par Cross-Encoder
- `suggested_reason`: Justification suggérée par l'IA
- `ia_metadata`: Détails techniques (JSON)

#### Métriques Externes (Nouveau)
- `citation_count`: Nombre de citations (Semantic Scholar)
  - Mis à jour automatiquement lors de la recherche
  - Peut être rafraîchi manuellement
  - Support DOI et arXiv ID

#### Relations
- `search_session_id` (FK): Lien vers la session de recherche
- `history`: Liste des changements d'état

### 3. `ArticleHistory` (Traçabilité)
Enregistre chaque changement d'état pour l'audit PRISMA.
- `article_id` (FK), `previous_status`, `new_status`
- `timestamp`, `user`, `action`, `reason`

### 4. Tables de Configuration
- `ExclusionCriteria`: Critères d'exclusion pour le Screening
- `EligibilityCriteria`: Critères pour la phase d'Éligibilité
- `SemanticFilterRun`: Historique des filtres sémantiques
- `AIAnalysisRun`: Historique des analyses IA

---

## Moteurs d'Analyse (IA & NLP)

### 1. Ranking Sémantique (`src/advanced_sorting.py`)
Utilise `sentence-transformers` pour trier les articles par pertinence.
- **Modèle Bi-Encoder** (`paraphrase-MiniLM-L3-v2`): Encodage rapide des vecteurs
- **Cross-Encoder** (optionnel): Re-ranking précis
- **Chunking intelligent**: Traitement des textes longs par segments

### 2. Filtrage Sémantique (`src/concept_filter.py`)
Filtre les articles par concepts avec recherche sémantique.
- Supporte opérateurs booléens (AND/OR)
- Recherche dans Titre, Abstract et Full Text
- Utilise embeddings pour matching sémantique

### 3. Enrichissement Métadonnées (`src/enrichment.py`) **NOUVEAU**
Module d'enrichissement via l'API Semantic Scholar.

#### Fonctionnalités
- Récupération automatique du nombre de citations
- Support DOI et arXiv ID
- Rate limiting intelligent (100 req/5min)
- Mise à jour en arrière-plan

#### Processus
1. **Déclenchement automatique** après chaque recherche
2. **Identification** : DOI ou arXiv ID
3. **Appel API** : `https://api.semanticscholar.org/graph/v1/paper/{ID}`
4. **Mise à jour** : Colonne `citation_count` dans la BDD
5. **Rate limiting** : 3 secondes entre requêtes (respect strict)

#### Utilisation
```python
from src.enrichment import enrich_session_articles

# Enrichir tous les articles d'une session
enrich_session_articles(session_id=1)
```

### 4. Extraction de PDF (`src/pdf_retriever.py`)
- Téléchargement via Unpaywall API et arXiv
- Extraction de texte avec PyMuPDF
- **Cache intelligent** : Skip si PDF déjà présent
- Organisation par session : `data/0_raw/pdfs/session_{id}_{query}/`

---

## Workflow Automatisé

### Recherche et Enrichissement Automatique

Lors d'une recherche arXiv, **3 processus s'exécutent en parallèle en arrière-plan** :

```python
# Thread 1: Analyse IA
thread_ia = threading.Thread(target=run_background_analysis, args=(article_ids, query))

# Thread 2: Récupération PDFs manquants
thread_pdf = threading.Thread(target=auto_retrieve_missing_pdfs, args=(session_id, limit))

# Thread 3: Enrichissement citations (NOUVEAU)
thread_citations = threading.Thread(target=enrich_session_articles, args=(session_id,))
```

#### Avantages
- Non-bloquant pour l'utilisateur
- Traitement parallèle efficace
- Respect automatique des rate limits
- Traçabilité complète via `ArticleHistory`

### Optimisations Majeures

1. **Cache PDF** : Skip téléchargement si fichier existe (>1KB)
2. **Search Scope** : Recherche uniquement sur titre et abstract pour arXiv
3. **Session-based storage** : PDFs organisés par session pour traçabilité
4. **Rate limit adherence** : Pause de 3s entre requêtes Semantic Scholar

---

## Interface Utilisateur

### Pages Principales

#### 1. Recherche (`pages/1_Recherche.py`)
- **Mode Automatique** : arXiv avec enrichissement auto
- **Mode Avancé** : Multi-sources (PubMed, Crossref, Google Scholar)
- **Enrichissement automatique** :
  - Scores IA
  - PDFs manquants
  - Citations Semantic Scholar (nouveau)

#### 2. Screening (`pages/2_Screening.py`)
- Tri par score IA
- Pré-tri sémantique avec annulation possible
- **Affichage citations** : `[Score IA] [Citations] Titre`
- Décisions assistées par IA
- Recalcul des scores

#### 3. Éligibilité (`pages/3_Eligibilite.py`)
- Revue texte complet
- Critères personnalisables
- Gestion undo des décisions

#### 4. Analyse (`pages/4_Analyse.py`)
- Diagramme de flux PRISMA
- Export des résultats
- Statistiques complètes

### Gestion des Erreurs UX

#### Bouton "Restaurer les articles"
Disponible dans la sidebar de Screening si des articles sont exclus par filtre sémantique.
- **Emplacement** : Toujours visible (avant vérification des articles)
- **Fonction** : Restaure `EXCLUDED_SEMANTIC_FILTER` → `IDENTIFIED`
- **Traçabilité** : Enregistre l'action dans `ArticleHistory`

---

## Guide de Développement

### Installation de l'environnement
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Ajouter une nouvelle page
1. Créer un fichier `pages/X_NomPage.py`
2. Importer et appeler `load_premium_css()` au début
3. Utiliser `get_db()` pour accéder à la BDD

### Modifier le modèle de données
1. Éditer `src/database.py`
2. **Migration SQLite** : Créer script dans `scripts/` pour ALTER TABLE
3. Exemple : Ajout colonne `citation_count`
   ```python
   import sqlite3
   conn = sqlite3.connect('data/prisma.db')
   conn.execute('ALTER TABLE articles ADD COLUMN citation_count INTEGER DEFAULT 0')
   conn.commit()
   ```

### Ajouter un nouvel enrichissement
1. Créer module dans `src/` (ex: `src/impact_metrics.py`)
2. Ajouter colonne(s) dans `Article` model
3. Créer thread dans `pages/1_Recherche.py` après recherche
4. Respecter rate limits APIs externes

---

## Dépendances Principales

### Core
- `streamlit`: Framework UI
- `sqlalchemy`: ORM
- `pandas`: Manipulation données

### Scraping
- `requests`: HTTP calls
- `beautifulsoup4`: Parsing HTML (Google Scholar)

### NLP & IA
- `sentence-transformers`: Embeddings sémantiques
- `transformers`: Modèles Hugging Face

### PDF
- `PyMuPDF` (fitz): Extraction texte
- `pypdf`: Manipulation PDF

### Nouvelles dépendances
- Aucune nouvelle (utilise `requests` déjà présent pour Semantic Scholar)

---

## API Externes

### Semantic Scholar API **NOUVEAU**
- **Endpoint** : `https://api.semanticscholar.org/graph/v1/paper/`
- **Rate Limit** : 100 requêtes / 5 minutes (gratuit)
- **Support** : DOI et arXiv ID
- **Champs** : `citationCount` uniquement
- **Gestion erreurs** : 404 (non trouvé), 429 (rate limit)

### arXiv API
- **Endpoint** : `http://export.arxiv.org/api/query`
- **Scope** : Titre et Abstract uniquement (optimisé)
- **Rate Limit** : 3 secondes entre requêtes

### Unpaywall API
- Récupération PDFs en accès libre

---

## Changelog Récent

### Citation Counting Feature
- Nouvelle colonne `Article.citation_count`
- Module `src/enrichment.py` créé
- Enrichissement automatique en arrière-plan
- Interface Screening : badge citations visible
- Support DOI + arXiv ID
- Rate limiting respecté (3s/req)

### Optimisations
- Cache PDF : skip si fichier existe
- Search scope arXiv : titre + abstract uniquement
- Organisation session-based des PDFs
- Bouton undo filtre sémantique toujours visible

### UI/UX
- Suppression des emojis (préférence utilisateur)
- Bouton "Restaurer les articles" accessible même sans articles
- Messages d'erreur améliorés

---

## Maintenance

### Reset d'un filtre sémantique
Si tous les articles sont exclus par erreur, utiliser la sidebar "Gestion des Filtres" > "Restaurer les articles".

### Rafraîchir les citations
Les citations peuvent être rafraîchies en relançant une recherche (le cache PDF évite les retéléchargements).
