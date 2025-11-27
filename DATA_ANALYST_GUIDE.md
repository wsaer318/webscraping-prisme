# 📊 Guide Analyste de Données - PRISMA Review Manager

Ce document technique détaille les méthodologies statistiques, les algorithmes de NLP et les structures de données utilisés dans PRISMA Review Manager. Il est destiné aux Data Scientists, Bibliométriciens et Analystes.

---

## 🧠 Méthodologie Algorithmique & Statistique

### 1. Recherche Hybride (Hybrid Search)

Le classement des articles utilise une approche hybride combinant **Recherche Sémantique** (sens) et **Recherche Lexicale** (mots-clés exacts).

$$ Score_{final} = \alpha \cdot \text{Norm}(S_{sem}) + (1-\alpha) \cdot \text{Norm}(S_{lex}) $$
*Avec $\alpha = 0.7$ (poids prépondérant à la sémantique).*

#### A. Score Sémantique (Bi-Encoder)
Chaque article $D$ est découpé en segments (chunks) $c_i$ de taille fixe (400 mots).
Le score $S_{sem}$ est la similarité cosinus maximale (Max Pooling) entre l'embedding de la requête $\mathbf{v}_Q$ et les embeddings des chunks :

$$ S_{sem}(D, Q) = \max_{i} \left( \frac{\mathbf{v}_Q \cdot \mathbf{v}_{c_i}}{\|\mathbf{v}_Q\| \|\mathbf{v}_{c_i}\|} \right) $$

**Modèle :** `paraphrase-MiniLM-L3-v2` (384 dimensions).

#### B. Score Lexical (BM25)
Utilise l'algorithme **BM25Okapi** pour capturer la fréquence des termes exacts, ce qui est crucial pour les acronymes ou termes techniques spécifiques que le modèle sémantique pourrait manquer.

$$ S_{lex}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})} $$

**Paramètres standards :** $k_1 = 1.5$, $b = 0.75$.

#### C. Normalisation & Fusion
Les scores bruts (Cosinus [-1, 1] et BM25 [0, $\infty$]) sont normalisés par **Min-Max Scaling** avant la fusion linéaire pour garantir une contribution équilibrée.

### 2. Suggestion de Seuil (Gaussian Mixture Model)

Pour suggérer automatiquement un seuil de coupure (threshold) entre articles pertinents et non-pertinents, nous utilisons une approche non-supervisée basée sur un **Modèle de Mélange Gaussien (GMM)**.

On suppose que la distribution des scores de similarité $S$ suit un mélange de deux distributions normales (bimodale) :
1.  Distribution du bruit (articles non pertinents) : $\mathcal{N}(\mu_0, \sigma_0^2)$
2.  Distribution du signal (articles pertinents) : $\mathcal{N}(\mu_1, \sigma_1^2)$ avec $\mu_1 > \mu_0$.

Le seuil optimal $T$ est estimé par la moyenne des espérances :
$$ T \approx \frac{\mu_0 + \mu_1}{2} $$

*Note : Si le GMM ne converge pas (trop peu de données), la médiane est utilisée comme fallback robuste.*

### 3. Classification Zero-Shot (Cross-Encoder)

Pour la vérification des critères d'exclusion (ex: "Animal Study"), nous utilisons un modèle **Cross-Encoder** (`ms-marco-TinyBERT-L-2-v2`). Contrairement au Bi-Encoder, ce modèle prend en entrée la paire concaténée (Texte, Critère) et prédit directement un score d'entailment (logit).

$$ \text{Logit}(D, C) = f_{\theta}(\text{concat}(D, C)) $$

**Seuil de décision :** Calibré empiriquement à **-11.0** (logits bruts) pour maximiser le rappel (Recall) et minimiser les Faux Négatifs lors du screening.

---

## 🔄 Flux de Données & ETL

### Pipeline d'Ingestion
1.  **Raw Data** : JSON/XML provenant des APIs (arXiv, PubMed).
2.  **Normalization** : Nettoyage Unicode (`ftfy`), détection de langue.
3.  **Deduplication** :
    *   Clé primaire stricte : `DOI` (si disponible).
    *   Clé floue : `NormalizedTitle` (minuscules, sans ponctuation).
    *   $$ \text{norm}(t) = \text{lower}(\text{remove\_punctuation}(t)) $$

### Métriques de Performance (KPIs)

| Métrique | Formule | Interprétation |
| :--- | :--- | :--- |
| **Inclusion Rate** | $N_{included} / N_{identified}$ | Spécificité de la requête initiale. Cible : 1-5% pour revues larges. |
| **Screening Efficiency** | $N_{screened\_out} / \Delta t$ | Vitesse de tri (articles/heure). |
| **Inter-Rater Agreement** | (Prévu v2) Kappa de Cohen | Accord entre plusieurs reviewers. |

---

## 💾 Schéma de Données & Exports

### Structure SQL (Relationnelle)
Le schéma est normalisé (3NF) pour garantir l'intégrité.
*   `articles` (Table de faits)
*   `search_sessions` (Dimension Temps/Requête)
*   `article_history` (Dimension Audit/Log)

### Exports pour Analyse Avancée

#### 1. JSON Complet (Pour NLP/ML)
Contient les structures imbriquées et les métadonnées IA.
```json
{
  "id": 123,
  "title": "Deep Learning...",
  "ia_metadata": {
    "model": "paraphrase-MiniLM-L3-v2",
    "criteria_scores": {"Animal Study": -8.4, "Review": -12.1}
  },
  "history": [...]
}
```

#### 2. CSV Plat (Pour Excel/Tableau/PowerBI)
Aplatit les données pour l'analyse pivot.
*   `ia_score` : Score de pertinence (float).
*   `decision_final` : Statut final (catégorique).

---

## 🛠️ Outils Recommandés

Pour explorer la base de données `prisma.db` (SQLite) :
*   **Python** : `pandas.read_sql("SELECT * FROM articles", con)`
*   **R** : `RSQLite` pour l'analyse bibliométrique (`bibliometrix`).
*   **SQL** : Requêtes directes pour agrégations complexes.

**Exemple d'analyse de tendance (SQL) :**
```sql
-- Taux d'inclusion par année
SELECT 
    year,
    COUNT(*) as total,
    SUM(CASE WHEN status = 'INCLUDED' THEN 1 ELSE 0 END) as included,
    ROUND(AVG(relevance_score), 3) as avg_ai_score
FROM articles
GROUP BY year
HAVING total > 5
ORDER BY year DESC;
```
