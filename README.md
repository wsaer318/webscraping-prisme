
## 🚀 Démarrage Rapide

### Prérequis
- Python 3.10 ou supérieur
- Clé API OpenAI (optionnel, pour fonctionnalités génératives futures)

### Installation

```bash
# 1. Cloner le projet
git clone https://github.com/votre-repo/prisma-manager.git
cd prisma-manager

# 2. Créer un environnement virtuel
python -m venv .venv
# Windows :
.venv\Scripts\activate
# Mac/Linux :
source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Lancement de l'Application

```bash
streamlit run app.py
```
L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`.

---

## ✨ Fonctionnalités Clés

### 1. 🔍 Collecte Multi-Sources
- Recherche unifiée sur **arXiv**, **PubMed**, **Google Scholar** et **Crossref**.
- Déduplication automatique intelligente (basée sur DOI et similarité de titre).
- Téléchargement automatique des PDFs (via Unpaywall/ArXiv).

### 2. 🧠 Tri Assisté par IA (Screening)
- **Ranking Sémantique** : Les articles sont triés par pertinence par rapport à votre requête grâce à des modèles de langage (Transformers).
- **Filtrage par Concepts** : Sélectionnez uniquement les articles contenant des concepts clés (ex: "Machine Learning" AND "Healthcare").
- **Interface Optimisée** : Décision rapide (Inclure/Exclure) sur Titre & Abstract.

### 3. 📋 Éligibilité & Revue
- Lecteur de PDF intégré (extraction texte brut).
- Surlignage des mots-clés.
- Gestion des critères d'inclusion/exclusion personnalisables.

### 4. 📊 Analyse & Reporting
- **Diagramme PRISMA 2020** généré automatiquement.
- Statistiques en temps réel (taux d'inclusion, distribution par année/source).
- Exports complets : **CSV**, **Excel**, **JSON**, **BibTeX**.

---

## 📚 Documentation

Pour aller plus loin, consultez nos guides spécialisés :

- **👩‍💻 [Documentation Technique](TECHNICAL_DOCS.md)** : Architecture du code, schéma de base de données, détails des algorithmes IA.
- **📈 [Guide Analyste de Données](DATA_ANALYST_GUIDE.md)** : Flux de données, définitions des métriques, guide SQL et formats d'export.

---

## 🛠️ Technologies

- **Interface** : Streamlit
- **Base de données** : SQLite + SQLAlchemy
- **NLP/IA** : Sentence-Transformers (HuggingFace), PyTorch, scikit-learn
- **Scraping** : BeautifulSoup4, Requests, PyMuPDF
- **Visualisation** : Matplotlib, Altair

## 🚀 Démarrage Rapide

### Prérequis
- Python 3.10 ou supérieur
- Clé API OpenAI (optionnel, pour fonctionnalités génératives futures)

### Installation

```bash
# 1. Cloner le projet
git clone https://github.com/votre-repo/prisma-manager.git
cd prisma-manager

# 2. Créer un environnement virtuel
python -m venv .venv
# Windows :
.venv\Scripts\activate
# Mac/Linux :
source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Lancement de l'Application

```bash
streamlit run app.py
```
L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`.

---

## ✨ Fonctionnalités Clés

### 1. 🔍 Collecte Multi-Sources
- Recherche unifiée sur **arXiv**, **PubMed**, **Google Scholar** et **Crossref**.
- Déduplication automatique intelligente (basée sur DOI et similarité de titre).
- Téléchargement automatique des PDFs (via Unpaywall/ArXiv).

### 2. 🧠 Tri Assisté par IA (Screening)
- **Ranking Sémantique** : Les articles sont triés par pertinence par rapport à votre requête grâce à des modèles de langage (Transformers).
- **Filtrage par Concepts** : Sélectionnez uniquement les articles contenant des concepts clés (ex: "Machine Learning" AND "Healthcare").
- **Interface Optimisée** : Décision rapide (Inclure/Exclure) sur Titre & Abstract.

### 3. 📋 Éligibilité & Revue
- Lecteur de PDF intégré (extraction texte brut).
- Surlignage des mots-clés.
- Gestion des critères d'inclusion/exclusion personnalisables.

### 4. 📊 Analyse & Reporting
- **Diagramme PRISMA 2020** généré automatiquement.
- Statistiques en temps réel (taux d'inclusion, distribution par année/source).
- Exports complets : **CSV**, **Excel**, **JSON**, **BibTeX**.

---

## 📚 Documentation

Pour aller plus loin, consultez nos guides spécialisés :

- **👩‍💻 [Documentation Technique](TECHNICAL_DOCS.md)** : Architecture du code, schéma de base de données, détails des algorithmes IA.
- **📈 [Guide Analyste de Données](DATA_ANALYST_GUIDE.md)** : Flux de données, définitions des métriques, guide SQL et formats d'export.

---

## 🛠️ Technologies

- **Interface** : Streamlit
- **Base de données** : SQLite + SQLAlchemy
- **NLP/IA** : Sentence-Transformers (HuggingFace), PyTorch, scikit-learn
- **Scraping** : BeautifulSoup4, Requests, PyMuPDF
- **Visualisation** : Matplotlib, Altair

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Voir `TECHNICAL_DOCS.md` pour comprendre l'architecture avant de proposer une Pull Request.

## 📝 Licence

Copyright © 2025 wsear318. Tous droits réservés.
Ce projet est sous licence propriétaire. Toute reproduction ou distribution non autorisée est interdite. Voir le fichier [LICENSE](LICENSE) pour plus de détails.