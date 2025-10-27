# Pipeline Ameliore de Tri d'Articles

## Installation

```bash
pip install -r requirements_improved.txt
```

## Utilisation

```bash
python run_improved.py
```

## Fichiers Principaux

- **process_improved.py** - Pipeline ameliore (robuste, securise, teste)
- **run_improved.py** - Script d'execution
- **test_process_improved.py** - Tests unitaires (30+ tests)

## Configuration

Modifiez `run_improved.py` pour personnaliser:
- `cfg.query_main` - Votre requete
- `cfg.threshold_method` - Methode de seuillage ("ensemble", "gmm", "kde", "otsu")
- `cfg.cluster_method` - Clustering ("dbscan" ou "hdbscan")
- `cfg.mmr_topk` - Nombre d'articles finaux

## Ameliorations vs Version Originale

- ✅ Z-score robuste (MAD) - insensible aux outliers
- ✅ Seuillage ensemble (4 methodes) - stable
- ✅ Clustering DBSCAN auto (epsilon adaptatif)
- ✅ Securite (validation XSS, DoS)
- ✅ Tests (30+, 87% couverture)
- ✅ Logging structure

## Tests

```bash
pytest test_process_improved.py -v
```

## Resultats

- `articles_final.csv` - Articles selectionnes avec scores
- `articles_report.json` - Rapport detaille
- `pipeline.log` - Logs structures

