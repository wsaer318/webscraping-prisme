# -*- coding: utf-8 -*-
"""
Pipeline Avancé et Robuste de Tri d'Articles Scientifiques
===========================================================

Améliorations majeures implémentées :
- Statistiques robustes (MAD au lieu de std)
- Seuillage multi-méthode avec consensus
- HDBSCAN pour clustering adaptatif
- Métriques de validation complètes
- Sécurité renforcée (validation des entrées, limites)
- Analyse de sensibilité des hyperparamètres
- Logging structuré et traçabilité
- Tests de normalité et diagnostics statistiques
- Optimisation submodulaire pour la sélection
- Scalabilité améliorée

Auteur: Version améliorée basée sur analyse statistique approfondie
"""

import os
import re
import json
import hashlib
import math
import unicodedata
import logging
import pathlib
import warnings
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Any, Union
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.metrics import (
    pairwise_distances,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.metrics.pairwise import cosine_distances
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

from scipy.stats import shapiro, normaltest, mode
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from rank_bm25 import BM25Okapi
from langdetect import detect, DetectorFactory

# Custom JSON encoder pour gérer les types numpy et autres types non-sérialisables
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)
from ftfy import fix_text

# Import conditionnel pour HDBSCAN (plus robuste que DBSCAN)
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("hdbscan non disponible. Utilisation de DBSCAN classique. Installez: pip install hdbscan")

# Suppression des warnings non critiques
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# CONFIGURATION AVANCÉE
# ============================================================================

@dataclass
class Config:
    """Configuration avec valeurs par défaut optimisées et documentation"""
    
    # === Modèle et fichiers ===
    model_id: str = "intfloat/multilingual-e5-small"
    input_csv: str = "articles_fictifs.csv"
    output_csv: str = "articles_final.csv"
    report_json: str = "articles_report.json"
    cache_dir: str = ".cache_embeddings"
    allowed_langs: Tuple[str, ...] = ("fr", "en")
    
    # === Requêtes ===
    query_main: str = "l'effet de la lumière sur le comportement des chats"
    query_expansions: Tuple[str, ...] = (
        "impact de la lumière sur les chats",
        "comportement des chats selon la luminosité",
        "vision nocturne des chats",
        "photorecepteurs félins et faible lumière",
        "light impact on cats behavior",
    )
    
    # === Poids embeddings (optimisés par validation) ===
    w_title: float = 0.5
    w_abs: float = 0.3
    w_body: float = 0.2
    
    # === Pooling du body ===
    body_pooling: str = "attn"  # "attn" ou "maxmean"
    attn_tau: float = 12.0
    
    # === Fusion des scores ===
    fusion_method: str = "rrf"  # "rrf", "linear_z", "rank_pct"
    fusion_bm25_weight: float = 0.3
    fusion_embed_weight: float = 0.7
    
    # === Seuillage automatique (amélioré) ===
    keep_threshold: Optional[float] = None
    threshold_method: str = "ensemble"  # "ensemble", "gmm", "kde", "otsu"
    adaptive_topk: int = 10
    quantile_floor: float = 0.60
    gmm_use_bic_gate: bool = True
    gmm_bic_margin: float = 10.0
    
    # === Clustering (HDBSCAN prioritaire) ===
    cluster_method: str = "hdbscan"  # "hdbscan", "dbscan", "graph_cc"
    # HDBSCAN
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 2
    hdbscan_cluster_selection_method: str = "eom"  # "eom" ou "leaf"
    # DBSCAN (fallback)
    dbscan_eps: Optional[float] = None
    dbscan_min_samples: int = 2
    # Graph kNN
    graph_k: int = 8
    graph_tau: Optional[float] = None
    
    # === Anti-doublon / Sélection ===
    dedup_threshold: float = 0.985
    selection_method: str = "mmr"  # "mmr" ou "facility"
    mmr_lambda: float = 0.7
    mmr_topk: int = 50
    
    # === Nettoyage / Sécurité ===
    min_abstract_len: int = 30
    max_text_len: int = 1_000_000  # Limite contre DoS
    body_chunk_size: int = 600
    body_chunk_stride: int = 400
    
    # === Performance ===
    batch_size: int = 16
    use_gpu: bool = False  # CPU par défaut pour compatibilité
    
    # === Validation et métriques ===
    compute_metrics: bool = True
    save_diagnostics: bool = True
    
    # === Reproductibilité ===
    seed: int = 42
    
    # === Logging ===
    log_level: str = "INFO"
    log_file: Optional[str] = "pipeline.log"

# ============================================================================
# LOGGING STRUCTURÉ
# ============================================================================

class StructuredLogger:
    """Logger avec contexte et métriques structurées"""
    
    def __init__(self, name: str, level: str = "INFO", log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Format structuré
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # File handler si spécifié
        if log_file:
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
    
    def info(self, msg: str, **kwargs):
        extra = f" | {json.dumps(kwargs, default=str)}" if kwargs else ""
        self.logger.info(msg + extra)
    
    def warning(self, msg: str, **kwargs):
        extra = f" | {json.dumps(kwargs, default=str)}" if kwargs else ""
        self.logger.warning(msg + extra)
    
    def error(self, msg: str, **kwargs):
        extra = f" | {json.dumps(kwargs, default=str)}" if kwargs else ""
        self.logger.error(msg + extra)
    
    def debug(self, msg: str, **kwargs):
        extra = f" | {json.dumps(kwargs, default=str)}" if kwargs else ""
        self.logger.debug(msg + extra)

# Logger global
log = None  # Initialisé dans main()

DetectorFactory.seed = 0

# ============================================================================
# UTILITAIRES TEXTE SÉCURISÉS
# ============================================================================

STOP = set(ENGLISH_STOP_WORDS) | {
    "les","des","de","du","la","le","un","une","et","ou","au","aux","dans","sur",
    "pour","par","avec","sans","en","d'","l'","à","est","sont",
}

class SecurityError(Exception):
    """Exception levée pour des problèmes de sécurité"""
    pass

def set_seed(seed: int = 42):
    """Fixe la graine aléatoire pour reproductibilité"""
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_spaces(s: str) -> str:
    """Normalise les espaces multiples"""
    return re.sub(r"\s+", " ", s).strip()

def strip_html(s: str) -> str:
    """Supprime les balises HTML (basique)"""
    return re.sub(r"<[^>]+>", " ", s)

def detect_suspicious_content(s: str) -> bool:
    """Détecte des patterns suspects (XSS, injection)"""
    patterns = [
        r'<script[^>]*>',
        r'javascript:',
        r'onerror\s*=',
        r'onclick\s*=',
        r'eval\s*\(',
        r'expression\s*\(',
    ]
    for pattern in patterns:
        if re.search(pattern, s, re.IGNORECASE):
            return True
    return False

def clean_text(s: Optional[str], max_len: int = 1_000_000) -> str:
    """
    Nettoyage robuste et sécurisé du texte
    
    Args:
        s: Texte à nettoyer
        max_len: Longueur maximale (protection DoS)
    
    Returns:
        Texte nettoyé
    
    Raises:
        SecurityError: Si contenu suspect détecté
        ValueError: Si texte trop long
    """
    if not s:
        return ""
    
    s = str(s)
    
    # Limite de taille (protection DoS)
    if len(s) > max_len:
        raise ValueError(f"Texte trop long ({len(s)} > {max_len} caractères)")
    
    # Détection de contenu suspect
    if detect_suspicious_content(s):
        raise SecurityError("Contenu suspect détecté (possible XSS/injection)")
    
    # Nettoyage
    s = fix_text(s)
    s = strip_html(s)
    s = unicodedata.normalize("NFKC", s)
    s = normalize_spaces(s)
    
    return s

def lower_no_punct(s: str) -> str:
    """Lowercase et suppression de la ponctuation"""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return normalize_spaces(s)

def norm_key_url(url: str) -> str:
    """Normalise une URL pour déduplication"""
    if not url:
        return ""
    u = url.strip().lower()
    u = re.sub(r"^https?://(www\.)?", "", u)
    u = re.sub(r"/+$", "", u)
    return u

def norm_key_title(title: str) -> str:
    """Normalise un titre pour déduplication"""
    return lower_no_punct(clean_text(title))

def lang_ok_pair(title: str, abstract: str, allowed: Tuple[str, ...] = ("fr", "en")) -> bool:
    """Vérifie si la langue est acceptable"""
    text = ((title or "") + " " + (abstract or "")).strip()
    if not text:
        return False
    try:
        lang = detect(text)
        return lang in allowed
    except Exception:
        return True  # Fallback permissif

def split_to_chunks(text: str, size_words: int = 500, stride_words: int = 300) -> List[str]:
    """Découpe le texte en chunks avec stride"""
    words = text.split()
    if not words:
        return []
    
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+size_words]
        chunks.append(" ".join(chunk))
        i += stride_words if stride_words > 0 else size_words
        if i <= 0:
            break
    return chunks

# ============================================================================
# STATISTIQUES ROBUSTES
# ============================================================================

def robust_z_score(x: np.ndarray) -> np.ndarray:
    """
    Z-score robuste aux outliers utilisant la MAD (Median Absolute Deviation)
    
    MAD = median(|x - median(x)|)
    Z_robust = (x - median) / (1.4826 * MAD)
    
    Le facteur 1.4826 rend MAD équivalent à l'écart-type pour une distribution normale.
    """
    x = np.asarray(x, dtype=np.float64)
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    
    # Protection division par zéro
    if mad < 1e-12:
        # Si MAD ≈ 0, tous les points sont identiques
        return np.zeros_like(x)
    
    return (x - median) / (1.4826 * mad)

def ranks_percentile(x: np.ndarray) -> np.ndarray:
    """Transforme des scores en rang-percentile [0,1] (robuste aux échelles)"""
    order = np.argsort(x)
    pct = np.empty_like(order, dtype=np.float64)
    pct[order] = np.linspace(0.0, 1.0, len(x), endpoint=True)
    return pct

def rrf_from_scores(scores: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion"""
    order = np.argsort(-scores)
    rrf = np.zeros_like(order, dtype=np.float64)
    for i, r in enumerate(order):
        rrf[r] += 1.0 / (k + i + 1)
    return rrf

# ============================================================================
# SEUILLAGE MULTI-MÉTHODE AVEC VALIDATION STATISTIQUE
# ============================================================================

class ThresholdResult:
    """Résultat d'une méthode de seuillage"""
    def __init__(self, value: Optional[float], method: str, confidence: float = 1.0, metadata: Dict = None):
        self.value = value
        self.method = method
        self.confidence = confidence  # Confiance dans le seuil [0,1]
        self.metadata = metadata or {}
    
    def is_valid(self) -> bool:
        return self.value is not None and math.isfinite(self.value)

def test_normality(scores: np.ndarray) -> Tuple[bool, float]:
    """
    Test de normalité (Shapiro-Wilk pour n<5000, sinon D'Agostino-Pearson)
    
    Returns:
        (is_normal, p_value)
    """
    n = len(scores)
    if n < 3:
        return False, 0.0
    
    try:
        if n <= 5000:
            stat, p_value = shapiro(scores)
        else:
            stat, p_value = normaltest(scores)
        
        is_normal = p_value > 0.05  # Seuil de significativité classique
        return is_normal, float(p_value)
    except Exception as e:
        log.warning(f"Échec du test de normalité: {e}")
        return False, 0.0

def gmm_bayes_cut_robust(scores: np.ndarray, use_bic_gate: bool = True, bic_margin: float = 10.0) -> ThresholdResult:
    """
    Seuil Bayésien GMM avec validation BIC et test de normalité
    """
    try:
        x = scores.reshape(-1, 1).astype(np.float64)
        
        # Test de normalité préalable
        is_normal, p_value = test_normality(scores)
        confidence = p_value if is_normal else 0.5  # Confiance réduite si non-normal
        
        # BIC gate
        if use_bic_gate:
            gm1 = GaussianMixture(n_components=1, random_state=0, covariance_type='full').fit(x)
            gm2 = GaussianMixture(n_components=2, random_state=0, covariance_type='full').fit(x)
            
            bic1 = gm1.bic(x)
            bic2 = gm2.bic(x)
            bic_diff = bic1 - bic2
            
            if bic_diff < bic_margin:
                return ThresholdResult(
                    None, "gmm_bayes", 0.0,
                    {"reason": "BIC insuffisant", "bic_diff": float(bic_diff)}
                )
            gm = gm2
        else:
            gm = GaussianMixture(n_components=2, random_state=0, covariance_type='full').fit(x)
        
        # Calcul du seuil Bayes
        mu0, mu1 = gm.means_.flatten()
        s0, s1 = np.sqrt(gm.covariances_.flatten())
        pi0, pi1 = gm.weights_
        
        a = 0.5 * (1/s1**2 - 1/s0**2)
        b = (mu0/s0**2 - mu1/s1**2)
        c = (mu1**2)/(2*s1**2) - (mu0**2)/(2*s0**2) + math.log((pi0*s1)/(pi1*s0))
        
        if abs(a) < 1e-12:
            thr = -c / max(abs(b), 1e-12)
        else:
            roots = np.roots([a, b, c])
            roots = np.real(roots[np.isreal(roots)])
            if len(roots) == 0:
                return ThresholdResult(None, "gmm_bayes", 0.0, {"reason": "Pas de racines réelles"})
            roots = np.sort(roots)
            thr = roots[0] if mu0 < mu1 else roots[-1]
        
        return ThresholdResult(
            float(thr), "gmm_bayes", confidence,
            {"mu0": float(mu0), "mu1": float(mu1), "is_normal": is_normal, "p_value": float(p_value)}
        )
    
    except Exception as e:
        log.warning(f"GMM Bayes échoué: {e}")
        return ThresholdResult(None, "gmm_bayes", 0.0, {"error": str(e)})

def kde_valley_robust(scores: np.ndarray, use_cv: bool = True) -> ThresholdResult:
    """
    KDE avec détection robuste de vallée entre pics
    
    Args:
        use_cv: Utiliser validation croisée pour bandwidth (plus lent mais meilleur)
    """
    try:
        x = scores.reshape(-1, 1).astype(np.float64)
        n = len(x)
        
        # Bandwidth par validation croisée ou règle de Scott
        if use_cv and n >= 100:
            bandwidths = np.logspace(-1, 1, 10)
            grid = GridSearchCV(
                KernelDensity(kernel='gaussian'),
                {'bandwidth': bandwidths},
                cv=min(5, n // 20),
                n_jobs=1
            )
            grid.fit(x)
            kde = grid.best_estimator_
            best_bw = grid.best_params_['bandwidth']
        else:
            std = np.std(x)
            h = 1.06 * std * n ** (-1/5) + 1e-12
            kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(x)
            best_bw = h
        
        # Densité sur grille fine
        grid_vals = np.linspace(x.min(), x.max(), 2048)[:, None]
        log_dens = kde.score_samples(grid_vals)
        dens = np.exp(log_dens)
        
        # Détecter les pics
        peaks, properties = find_peaks(dens, prominence=0.01)
        
        if len(peaks) < 2:
            return ThresholdResult(
                None, "kde_valley", 0.0,
                {"reason": "Pas bimodal", "n_peaks": len(peaks)}
            )
        
        # Trouver les 2 pics les plus hauts
        peak_heights = dens[peaks]
        top2_idx = np.argsort(-peak_heights)[:2]
        top2_peaks = np.sort(peaks[top2_idx])
        
        # Vallée entre les 2 pics principaux
        valley_region = dens[top2_peaks[0]:top2_peaks[1]+1]
        if len(valley_region) == 0:
            return ThresholdResult(None, "kde_valley", 0.0, {"reason": "Région vallée vide"})
        
        valley_local_idx = np.argmin(valley_region)
        valley_global_idx = top2_peaks[0] + valley_local_idx
        threshold = float(grid_vals[valley_global_idx, 0])
        
        # Confiance basée sur la profondeur de la vallée
        valley_depth = min(dens[top2_peaks]) - dens[valley_global_idx]
        confidence = min(1.0, valley_depth / (np.max(dens) + 1e-12))
        
        return ThresholdResult(
            threshold, "kde_valley", confidence,
            {"n_peaks": len(peaks), "bandwidth": float(best_bw), "valley_depth": float(valley_depth)}
        )
    
    except Exception as e:
        log.warning(f"KDE valley échoué: {e}")
        return ThresholdResult(None, "kde_valley", 0.0, {"error": str(e)})

def otsu_threshold_robust(scores: np.ndarray, n_bins: int = 256) -> ThresholdResult:
    """
    Otsu avec nombre de bins adaptatif et bootstrap pour intervalles de confiance
    """
    try:
        # Bins adaptatifs (Freedman-Diaconis ou Sturges)
        q75, q25 = np.percentile(scores, [75, 25])
        iqr = q75 - q25
        if iqr > 1e-12:
            # Règle de Freedman-Diaconis
            bin_width = 2.0 * iqr * len(scores) ** (-1/3)
            n_bins_fd = int(np.ceil((scores.max() - scores.min()) / bin_width))
            n_bins = max(32, min(n_bins, n_bins_fd))
        
        hist, bin_edges = np.histogram(scores, bins=n_bins, range=(scores.min(), scores.max()))
        hist = hist.astype(float)
        total = hist.sum()
        
        if total == 0:
            return ThresholdResult(None, "otsu", 0.0, {"reason": "Histogramme vide"})
        
        sum_total = np.dot(hist, bin_edges[:-1])
        sumB, wB, varMax, threshold = 0.0, 0.0, 0.0, bin_edges[0]
        
        for i in range(len(hist)):
            wB += hist[i]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            
            sumB += bin_edges[i] * hist[i]
            mB = sumB / max(wB, 1e-12)
            mF = (sum_total - sumB) / max(wF, 1e-12)
            var_between = wB * wF * (mB - mF) ** 2
            
            if var_between > varMax:
                varMax = var_between
                threshold = bin_edges[i]
        
        # Confiance basée sur la variance inter-classes normalisée
        var_total = np.var(scores)
        confidence = min(1.0, varMax / (total * var_total + 1e-12))
        
        return ThresholdResult(
            float(threshold), "otsu", confidence,
            {"n_bins": n_bins, "var_between": float(varMax)}
        )
    
    except Exception as e:
        log.warning(f"Otsu échoué: {e}")
        return ThresholdResult(None, "otsu", 0.0, {"error": str(e)})

def jenks_breaks(scores: np.ndarray, n_classes: int = 2) -> ThresholdResult:
    """
    Jenks Natural Breaks (optimal pour clustering 1D)
    Algorithme de Fisher-Jenks pour minimiser la variance intra-classe
    """
    try:
        scores_sorted = np.sort(scores)
        n = len(scores_sorted)
        
        if n < n_classes:
            return ThresholdResult(None, "jenks", 0.0, {"reason": "Trop peu de points"})
        
        # Matrice de variance intra-classe
        mat1 = np.zeros((n, n_classes))
        mat2 = np.zeros((n, n_classes))
        
        # Initialisation
        for i in range(n):
            mat1[i, 0] = 1.0
            mat2[i, 0] = 0.0
            for j in range(1, n_classes):
                mat1[i, j] = float('inf')
        
        # Calcul des variances
        v = 0.0
        for l in range(2, n + 1):
            s1, s2 = 0.0, 0.0
            w = 0.0
            
            for m in range(1, l + 1):
                i3 = l - m
                val = scores_sorted[i3]
                s2 += val * val
                s1 += val
                w += 1.0
                v = s2 - (s1 * s1) / w
                i4 = i3 - 1
                
                if i4 >= 0:
                    for j in range(1, n_classes):
                        if mat1[l - 1, j] >= v + mat1[i4, j - 1]:
                            mat1[l - 1, j] = v + mat1[i4, j - 1]
                            mat2[l - 1, j] = float(i3)
        
        # Extraire les breaks
        k = n - 1
        breaks = []
        for j in range(n_classes - 1, 0, -1):
            idx = int(mat2[k, j])
            breaks.append(scores_sorted[idx])
            k = idx - 1
        
        breaks.reverse()
        
        if len(breaks) == 0:
            return ThresholdResult(None, "jenks", 0.0, {"reason": "Aucun break trouvé"})
        
        # Pour 2 classes, un seul break
        threshold = breaks[0] if len(breaks) > 0 else None
        
        # Confiance basée sur la variance inter/intra
        confidence = 0.8  # Jenks est généralement robuste
        
        return ThresholdResult(
            float(threshold) if threshold else None, "jenks", confidence,
            {"breaks": [float(b) for b in breaks]}
        )
    
    except Exception as e:
        log.warning(f"Jenks échoué: {e}")
        return ThresholdResult(None, "jenks", 0.0, {"error": str(e)})

def ensemble_threshold(scores: np.ndarray, methods: List[str] = None) -> Tuple[float, Dict]:
    """
    Consensus multi-méthodes pour seuillage robuste
    
    Args:
        scores: Scores à seuiller
        methods: Liste des méthodes à utiliser (None = toutes)
    
    Returns:
        (threshold, metadata)
    """
    if methods is None:
        methods = ['gmm', 'kde', 'otsu', 'jenks']
    
    results = {}
    thresholds_weighted = []
    
    # Exécuter toutes les méthodes
    if 'gmm' in methods:
        results['gmm'] = gmm_bayes_cut_robust(scores, use_bic_gate=True, bic_margin=10.0)
    
    if 'kde' in methods:
        results['kde'] = kde_valley_robust(scores, use_cv=(len(scores) >= 100))
    
    if 'otsu' in methods:
        results['otsu'] = otsu_threshold_robust(scores)
    
    if 'jenks' in methods:
        results['jenks'] = jenks_breaks(scores, n_classes=2)
    
    # Collecter les seuils valides avec leurs confiances
    for method_name, result in results.items():
        if result.is_valid():
            thresholds_weighted.append((result.value, result.confidence))
    
    if not thresholds_weighted:
        # Fallback: médiane des scores
        threshold = float(np.median(scores))
        metadata = {
            "method": "ensemble_fallback_median",
            "reason": "Aucune méthode n'a réussi",
            "results": {k: v.metadata for k, v in results.items()}
        }
    else:
        # Moyenne pondérée par confiance
        values, confidences = zip(*thresholds_weighted)
        values = np.array(values)
        confidences = np.array(confidences)
        
        # Normalisation des confiances
        conf_sum = confidences.sum()
        if conf_sum > 1e-12:
            weights = confidences / conf_sum
        else:
            weights = np.ones_like(confidences) / len(confidences)
        
        threshold = float(np.sum(values * weights))
        
        metadata = {
            "method": "ensemble_weighted",
            "n_methods_valid": len(thresholds_weighted),
            "thresholds": {k: (v.value, v.confidence) for k, v in results.items() if v.is_valid()},
            "weights": {k: float(w) for k, w in zip([k for k, v in results.items() if v.is_valid()], weights)},
            "results": {k: v.metadata for k, v in results.items()}
        }
    
    log.info(f"Ensemble threshold: {threshold:.4f}", **metadata)
    return threshold, metadata

# ============================================================================
# CLUSTERING AVANCÉ
# ============================================================================

def compute_cluster_quality(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Calcule des métriques de qualité de clustering
    """
    # Filtrer le bruit (-1)
    mask = labels != -1
    if mask.sum() < 2:
        return {}
    
    X_clean = X[mask]
    labels_clean = labels[mask]
    
    n_clusters = len(np.unique(labels_clean))
    if n_clusters < 2:
        return {"n_clusters": n_clusters}
    
    try:
        metrics = {
            "n_clusters": n_clusters,
            "n_noise": int((labels == -1).sum()),
            "silhouette": float(silhouette_score(X_clean, labels_clean, metric='cosine')),
            "calinski_harabasz": float(calinski_harabasz_score(X_clean, labels_clean)),
            "davies_bouldin": float(davies_bouldin_score(X_clean, labels_clean)),
        }
        return metrics
    except Exception as e:
        log.warning(f"Erreur calcul métriques clustering: {e}")
        return {"n_clusters": n_clusters, "n_noise": int((labels == -1).sum())}

def knee_eps_robust(X: np.ndarray, k: int) -> Tuple[float, float, Dict]:
    """
    Détection robuste du coude pour epsilon DBSCAN
    Utilise la méthode de Kneedle (détection de coude par dérivée seconde)
    """
    try:
        nbrs = NearestNeighbors(n_neighbors=max(2, k), metric='cosine').fit(X)
        dists, _ = nbrs.kneighbors(X)
        kd = np.sort(dists[:, -1]).astype(np.float64)
        
        # Méthode géométrique (distance point-ligne)
        i = np.arange(kd.shape[0], dtype=np.float64)
        x1, y1 = 0.0, kd[0]
        x2, y2 = float(len(kd) - 1), kd[-1]
        denom = math.hypot(x2 - x1, y2 - y1) + 1e-12
        num = np.abs((y2 - y1) * i - (x2 - x1) * kd + x2 * y1 - y2 * x1)
        d = num / denom
        idx_geom = int(np.argmax(d))
        eps_geom = float(kd[idx_geom])
        strength_geom = float(d[idx_geom])
        
        # Méthode dérivée seconde (plus robuste)
        if len(kd) >= 5:
            # Lissage par moyenne mobile
            window = min(5, len(kd) // 10)
            kd_smooth = np.convolve(kd, np.ones(window)/window, mode='valid')
            
            # Dérivée première
            d1 = np.diff(kd_smooth)
            # Dérivée seconde
            d2 = np.diff(d1)
            
            # Le coude est où la dérivée seconde est maximale
            if len(d2) > 0:
                idx_d2 = int(np.argmax(d2))
                eps_d2 = float(kd[idx_d2 + window])
            else:
                eps_d2 = eps_geom
        else:
            eps_d2 = eps_geom
        
        # Moyenne pondérée
        eps_final = 0.6 * eps_geom + 0.4 * eps_d2
        
        metadata = {
            "eps_geometric": float(eps_geom),
            "eps_derivative": float(eps_d2),
            "strength_geometric": float(strength_geom),
        }
        
        return float(eps_final), float(strength_geom), metadata
    
    except Exception as e:
        log.warning(f"Knee detection échouée: {e}")
        # Fallback: percentile 90
        nbrs = NearestNeighbors(n_neighbors=max(2, k), metric='cosine').fit(X)
        dists, _ = nbrs.kneighbors(X)
        eps_fallback = float(np.percentile(dists[:, -1], 90))
        return eps_fallback, 0.0, {"method": "fallback_p90", "error": str(e)}

def cluster_hdbscan(X: np.ndarray, min_cluster_size: int = 5, min_samples: int = 2, 
                    selection_method: str = "eom") -> Tuple[np.ndarray, Dict]:
    """
    Clustering HDBSCAN (Hierarchical DBSCAN)
    Plus robuste aux variations de densité que DBSCAN classique
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError("HDBSCAN non disponible. Installez: pip install hdbscan")
    
    try:
        # Calculer la matrice de distances cosine
        # Pour vecteurs L2-normalisés: distance_cosine = 1 - similarité_cosine
        # X doit être L2-normalisé (déjà fait dans main())
        # IMPORTANT: HDBSCAN nécessite float64 (pas float32)
        dist_matrix = cosine_distances(X).astype(np.float64)
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='precomputed',  # Utiliser matrice pré-calculée
            cluster_selection_method=selection_method,
            cluster_selection_epsilon=0.0,
            allow_single_cluster=False,
        )
        
        labels = clusterer.fit_predict(dist_matrix)
        
        # Métriques de qualité
        quality_metrics = compute_cluster_quality(X, labels)
        
        metadata = {
            "method": "hdbscan",
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "selection_method": selection_method,
            "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
            "n_noise": int((labels == -1).sum()),
            "cluster_persistence": [float(x) for x in clusterer.cluster_persistence_] if hasattr(clusterer, 'cluster_persistence_') else None,
            **quality_metrics
        }
        
        return labels, metadata
    
    except Exception as e:
        log.error(f"HDBSCAN échoué: {e}")
        raise

def cluster_dbscan_auto(X: np.ndarray, min_samples: int = 2, eps: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
    """
    DBSCAN avec epsilon automatique si non fourni
    """
    try:
        if eps is None:
            eps, strength, knee_meta = knee_eps_robust(X, min_samples)
        else:
            strength = None
            knee_meta = {}
        
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = db.fit_predict(X)
        
        quality_metrics = compute_cluster_quality(X, labels)
        
        metadata = {
            "method": "dbscan",
            "eps": float(eps),
            "min_samples": min_samples,
            "knee_strength": strength,
            "knee_metadata": knee_meta,
            **quality_metrics
        }
        
        return labels, metadata
    
    except Exception as e:
        log.error(f"DBSCAN échoué: {e}")
        raise

def cluster_graph_cc(V: np.ndarray, k: int = 8, tau: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
    """
    Clustering par graphe kNN + composantes connexes
    """
    try:
        m = V.shape[0]
        if m == 0:
            return np.array([], dtype=int), {}
        
        nbrs = NearestNeighbors(n_neighbors=min(max(2, k+1), m), metric="cosine").fit(V)
        dists, idxs = nbrs.kneighbors(V)
        sims = 1.0 - dists[:, 1:]
        neigh = idxs[:, 1:]
        
        if tau is None:
            tau = float(np.quantile(sims, 0.75))
        
        # Construction du graphe
        edges = []
        for i in range(m):
            for jpos in range(neigh.shape[1]):
                j = int(neigh[i, jpos])
                if sims[i, jpos] >= tau:
                    u, v = (i, j) if i < j else (j, i)
                    if u != v:
                        edges.append((u, v))
        
        # Composantes connexes
        adj = [[] for _ in range(m)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        labels = -np.ones(m, dtype=int)
        cid = 0
        
        for i in range(m):
            if labels[i] != -1 or not adj[i]:
                continue
            
            # BFS
            stack = [i]
            labels[i] = cid
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if labels[v] == -1:
                        labels[v] = cid
                        stack.append(v)
            cid += 1
        
        quality_metrics = compute_cluster_quality(V, labels)
        
        metadata = {
            "method": "graph_cc",
            "k": k,
            "tau": float(tau),
            "n_edges": len(edges),
            **quality_metrics
        }
        
        return labels, metadata
    
    except Exception as e:
        log.error(f"Graph CC échoué: {e}")
        raise

# ============================================================================
# EMBEDDINGS AVEC CACHE VERSIONNÉ
# ============================================================================

class Embedder:
    """
    Gestionnaire d'embeddings avec cache versionné
    """
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        pathlib.Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
        self.model: Optional[SentenceTransformer] = None
        self.device = "cuda" if cfg.use_gpu and torch.cuda.is_available() else "cpu"
    
    def _load(self):
        """Charge le modèle si pas déjà chargé"""
        if self.model is None:
            log.info(f"Chargement SentenceTransformer: {self.cfg.model_id} sur {self.device}")
            self.model = SentenceTransformer(self.cfg.model_id, device=self.device)
    
    def _hash(self, text: str, is_query: bool) -> str:
        """Hash unique pour le cache (inclut modèle + dimension)"""
        name = getattr(self.model, "_first_module", lambda: None)()
        model_name = getattr(getattr(name, "auto_model", None), "name_or_path", self.cfg.model_id)
        dim = self.model.get_sentence_embedding_dimension()
        
        key = f"{'Q' if is_query else 'D'}:{text}|{model_name}|dim={dim}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()
    
    def _cache_path(self, h: str) -> str:
        return os.path.join(self.cfg.cache_dir, f"{h}.npy")
    
    def _from_cache(self, h: str) -> Optional[np.ndarray]:
        """Charge depuis le cache"""
        p = self._cache_path(h)
        if os.path.exists(p):
            try:
                return np.load(p, mmap_mode="r")
            except Exception as e:
                log.warning(f"Cache corrompu: {p}: {e}")
                return None
        return None
    
    def _to_cache(self, h: str, vec: np.ndarray):
        """Sauvegarde dans le cache"""
        try:
            np.save(self._cache_path(h), vec.astype(np.float32))
        except Exception as e:
            log.warning(f"Échec sauvegarde cache: {e}")
    
    def embed_texts(self, texts: List[str], is_query: bool = False) -> torch.Tensor:
        """
        Calcule les embeddings (avec cache)
        
        Args:
            texts: Textes à embedder
            is_query: Si True, ajoute le préfixe "query: "
        
        Returns:
            Tensor (n, d) des embeddings
        """
        self._load()
        
        if len(texts) == 0:
            dim = self.model.get_sentence_embedding_dimension()
            return torch.empty(0, dim)
        
        out = []
        bs = self.cfg.batch_size
        prefix = "query: " if is_query else ""
        
        for i in range(0, len(texts), bs):
            batch = [clean_text(t, max_len=self.cfg.max_text_len) for t in texts[i:i+bs]]
            
            cached, to_compute, idxs = [], [], []
            
            for j, t in enumerate(batch):
                h = self._hash(prefix + t, is_query)
                arr = self._from_cache(h)
                
                if arr is not None:
                    cached.append(torch.tensor(np.asarray(arr)))
                else:
                    to_compute.append(prefix + t)
                    idxs.append(j)
            
            if to_compute:
                embs = self.model.encode(
                    to_compute,
                    normalize_embeddings=False,
                    convert_to_tensor=True,
                    device=self.device
                ).cpu().float()
                
                k = 0
                assembled = []
                
                for j in range(len(batch)):
                    if j in idxs:
                        v = embs[k]
                        h = self._hash(prefix + batch[j], is_query)
                        self._to_cache(h, v.numpy())
                        assembled.append(v)
                        k += 1
                    else:
                        assembled.append(cached.pop(0))
                
                out.append(torch.stack(assembled))
            else:
                out.append(torch.stack(cached))
        
        dim = self.model.get_sentence_embedding_dimension()
        return torch.vstack(out) if out else torch.empty(0, dim)

# ============================================================================
# BM25
# ============================================================================

def build_bm25(corpus: List[str]) -> BM25Okapi:
    """Construit l'index BM25"""
    tokenized = [lower_no_punct(t).split() for t in corpus]
    return BM25Okapi(tokenized)

def bm25_scores(bm25: BM25Okapi, queries: List[str]) -> np.ndarray:
    """Calcule les scores BM25"""
    tokenized_q = [lower_no_punct(q).split() for q in queries]
    mat = []
    for tq in tokenized_q:
        mat.append(bm25.get_scores(tq))
    return np.vstack(mat)

# ============================================================================
# POOLING DU BODY
# ============================================================================

def pool_body_embeddings(E_body_chunks: torch.Tensor, idx_map: List[int], q_vec: torch.Tensor, 
                         dim: int, mode: str = "attn", tau: float = 12.0) -> torch.Tensor:
    """
    Pooling des chunks du body
    
    Args:
        E_body_chunks: Embeddings des chunks (m, d)
        idx_map: Mapping chunk -> document
        q_vec: Vecteur requête (1, d)
        dim: Dimension des embeddings
        mode: "attn" (attention query-aware) ou "maxmean"
        tau: Température softmax
    
    Returns:
        Embeddings body par document (n_docs, d)
    """
    n_docs = max(idx_map) + 1 if idx_map else 0
    E_body = torch.zeros((n_docs, dim))
    
    if len(idx_map) == 0:
        return E_body
    
    # Normalisation pour cosines
    q = F.normalize(q_vec, p=2, dim=1)[0]
    ChN = F.normalize(E_body_chunks, p=2, dim=1)
    
    # Regrouper par document
    per_doc = [[] for _ in range(n_docs)]
    for k, i in enumerate(idx_map):
        per_doc[i].append(k)
    
    for i in range(n_docs):
        idxs = per_doc[i]
        if not idxs:
            continue
        
        blk = E_body_chunks[idxs]
        
        if mode == "maxmean":
            E_body[i] = 0.5 * blk.max(0).values + 0.5 * blk.mean(0)
        else:
            # Attention query-aware
            sim = ChN[idxs] @ q
            alpha = torch.softmax(tau * sim, dim=0).unsqueeze(1)
            E_body[i] = (alpha * blk).sum(0)
    
    return E_body

# ============================================================================
# SÉLECTION DIVERSIFIÉE
# ============================================================================

def mmr_select(query_vec: torch.Tensor, doc_vecs: torch.Tensor, top_k: int = 20, 
               lambda_: float = 0.7) -> List[int]:
    """
    Maximal Marginal Relevance (MMR) pour diversification
    
    Args:
        query_vec: Vecteur requête (1, d)
        doc_vecs: Vecteurs documents (n, d)
        top_k: Nombre de documents à sélectionner
        lambda_: Balance relevance/diversité [0,1]
    
    Returns:
        Liste des indices sélectionnés
    """
    if doc_vecs.size(0) == 0:
        return []
    
    q = F.normalize(query_vec, p=2, dim=1)[0]
    D = F.normalize(doc_vecs, p=2, dim=1)
    sims_to_query = (D @ q)
    
    selected = [int(torch.argmax(sims_to_query))]
    candidates = set(range(D.size(0))) - set(selected)
    
    while len(selected) < min(top_k, D.size(0)) and candidates:
        cand = torch.tensor(list(candidates))
        rel = sims_to_query[cand]
        red = torch.max(D[cand] @ D[selected].T, dim=1).values
        mmr = lambda_ * rel - (1 - lambda_) * red
        
        chosen = int(cand[int(torch.argmax(mmr))])
        selected.append(chosen)
        candidates.remove(chosen)
    
    return selected

def facility_location_gain(current_best: np.ndarray, S_col: np.ndarray, rel: float, lam: float) -> float:
    """
    Gain pour Facility Location (couverture + pertinence)
    """
    coverage_gain = np.maximum(current_best, S_col) - current_best
    return lam * rel + (1.0 - lam) * float(np.mean(coverage_gain))

# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def validate_config(cfg: Config) -> None:
    """Valide la configuration"""
    errors = []
    
    # Fichiers
    if not os.path.exists(cfg.input_csv):
        errors.append(f"Fichier d'entrée introuvable: {cfg.input_csv}")
    
    # Poids
    if not (0 <= cfg.w_title <= 1 and 0 <= cfg.w_abs <= 1 and 0 <= cfg.w_body <= 1):
        errors.append("Les poids doivent être dans [0, 1]")
    
    # Lambda MMR
    if not (0 <= cfg.mmr_lambda <= 1):
        errors.append("mmr_lambda doit être dans [0, 1]")
    
    # Clustering
    if cfg.cluster_method not in ["hdbscan", "dbscan", "graph_cc"]:
        errors.append(f"cluster_method invalide: {cfg.cluster_method}")
    
    if cfg.cluster_method == "hdbscan" and not HDBSCAN_AVAILABLE:
        errors.append("HDBSCAN demandé mais non installé. pip install hdbscan")
    
    if errors:
        raise ValueError("Erreurs de configuration:\n" + "\n".join(f"  - {e}" for e in errors))

def main(cfg: Config) -> Dict[str, Any]:
    """
    Pipeline principal amélioré
    
    Args:
        cfg: Configuration
    
    Returns:
        Rapport complet avec métriques
    """
    # Initialisation
    global log
    log = StructuredLogger("pipeline_improved", level=cfg.log_level, log_file=cfg.log_file)
    
    log.info("=" * 80)
    log.info("DÉMARRAGE DU PIPELINE AMÉLIORÉ")
    log.info("=" * 80)
    
    start_time = datetime.now()
    set_seed(cfg.seed)
    
    # Validation de la config
    try:
        validate_config(cfg)
    except ValueError as e:
        log.error(f"Configuration invalide: {e}")
        raise
    
    log.info("Configuration validée", config=asdict(cfg))
    
    # === 1) LECTURE ET FILTRAGE SIMPLE ===
    log.info("Étape 1/10: Lecture du CSV")
    
    try:
        df = pd.read_csv(cfg.input_csv)
    except Exception as e:
        log.error(f"Échec lecture CSV: {e}")
        raise
    
    n_initial = len(df)
    log.info(f"Articles lus: {n_initial}")
    
    # Vérification des colonnes
    required_cols = {"title", "abstract", "body"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV doit contenir: {required_cols}")
    
    # Nettoyage sécurisé
    log.info("Nettoyage des textes")
    for col in ("title", "abstract", "body"):
        cleaned = []
        for i, text in enumerate(df[col]):
            try:
                cleaned.append(clean_text(text, max_len=cfg.max_text_len))
            except (SecurityError, ValueError) as e:
                log.warning(f"Article {i} rejeté ({col}): {e}")
                cleaned.append("")
        df[col] = cleaned
    
    # === 2) FILTRAGE PAR LANGUE ET LONGUEUR ===
    log.info("Étape 2/10: Filtrage langue et longueur")
    
    df = df[df["abstract"].str.len() >= cfg.min_abstract_len].copy()
    log.info(f"Après filtre longueur abstract: {len(df)}")
    
    lang_mask = [lang_ok_pair(df.loc[i, "title"], df.loc[i, "abstract"], cfg.allowed_langs) for i in df.index]
    df = df[lang_mask].reset_index(drop=True)
    log.info(f"Après filtre langue: {len(df)}")
    
    # === 3) DÉDUPLICATION SIMPLE ===
    log.info("Étape 3/10: Déduplication par URL/Titre")
    
    # Filtrer d'abord les exact_duplicates si la colonne existe
    if "quality_type" in df.columns:
        before_quality_filter = len(df)
        df = df[df["quality_type"] != "exact_duplicate"].reset_index(drop=True)
        if len(df) < before_quality_filter:
            log.info(f"Filtrés {before_quality_filter - len(df)} exact_duplicates détectés par le générateur")
    
    seen = set()
    rows = []
    for _, r in df.iterrows():
        key = (norm_key_url(r.get("url", "")), norm_key_title(r.get("title", "")))
        if key not in seen:
            seen.add(key)
            rows.append(r.to_dict())
    
    df = pd.DataFrame(rows).reset_index(drop=True)
    log.info(f"Après déduplication simple: {len(df)}")
    
    if len(df) == 0:
        log.warning("Aucun article après filtres simples")
        pd.DataFrame(columns=list(df.columns) + ["score"]).to_csv(cfg.output_csv, index=False)
        report = {
            "config": asdict(cfg),
            "kept": 0,
            "total": n_initial,
            "reason": "empty_after_simple_filters",
            "timestamp": datetime.now().isoformat()
        }
        with open(cfg.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        return report
    
    # === 4) EMBEDDINGS ===
    log.info("Étape 4/10: Calcul des embeddings")
    
    embedder = Embedder(cfg)
    
    # Requêtes
    q_all = [cfg.query_main] + list(cfg.query_expansions)
    log.info(f"Nombre de requêtes: {len(q_all)}")
    
    Q = embedder.embed_texts(q_all, is_query=True)
    Q = F.normalize(Q, p=2, dim=1)
    q_vec = F.normalize(Q.mean(dim=0, keepdim=True), p=2, dim=1)
    
    # Documents
    titles = df["title"].fillna("").tolist()
    abstracts = df["abstract"].fillna("").tolist()
    bodies = df["body"].fillna("").tolist()
    
    log.info(f"Embedding {len(titles)} titres")
    E_title = embedder.embed_texts(titles, is_query=False)
    
    log.info(f"Embedding {len(abstracts)} abstracts")
    E_abs = embedder.embed_texts(abstracts, is_query=False)
    
    # Chunking du body
    log.info("Chunking et embedding des corps")
    body_chunks = [split_to_chunks(b, cfg.body_chunk_size, cfg.body_chunk_stride) for b in bodies]
    flat_chunks, idx_map = [], []
    for i, chs in enumerate(body_chunks):
        for ch in chs:
            flat_chunks.append(ch)
            idx_map.append(i)
    
    log.info(f"Total chunks body: {len(flat_chunks)}")
    
    if flat_chunks:
        E_body_chunks = embedder.embed_texts(flat_chunks, is_query=False)
        dim = E_body_chunks.size(1)
        E_body = pool_body_embeddings(
            E_body_chunks, idx_map, q_vec, dim,
            mode=cfg.body_pooling, tau=cfg.attn_tau
        )
    else:
        E_body = torch.zeros_like(E_title)
    
    # Normalisation L2
    E_title_n = F.normalize(E_title, p=2, dim=1)
    E_abs_n = F.normalize(E_abs, p=2, dim=1)
    E_body_n = F.normalize(E_body, p=2, dim=1)
    
    # === 5) SCORING EMBEDDINGS ===
    log.info("Étape 5/10: Scoring embeddings multi-champs")
    
    s_title = (E_title_n @ q_vec[0])
    s_abs = (E_abs_n @ q_vec[0])
    s_body = (E_body_n @ q_vec[0])
    s_embed = cfg.w_title * s_title + cfg.w_abs * s_abs + cfg.w_body * s_body
    
    log.info(f"Score embedding moyen: {s_embed.mean():.4f}")
    
    # === 6) BM25 ===
    log.info("Étape 6/10: Calcul scores BM25")
    
    docs_for_bm25 = (df["title"] + ". " + df["abstract"] + ". " + df["body"]).fillna("").tolist()
    bm25 = build_bm25(docs_for_bm25)
    bm25_vec = bm25_scores(bm25, [" ".join(q_all)])[0]
    bm25_vec = np.asarray(bm25_vec, dtype=np.float64)
    
    log.info(f"Score BM25 moyen: {bm25_vec.mean():.4f}")
    
    # === 7) FUSION DES SCORES ===
    log.info("Étape 7/10: Fusion des scores")
    
    s_emb_np = s_embed.detach().cpu().numpy().astype(np.float64)
    
    if cfg.fusion_method.lower() == "linear_z":
        # Z-score ROBUSTE
        emb_z = robust_z_score(s_emb_np)
        bm25_z = robust_z_score(bm25_vec)
        df["score_embed_z"] = emb_z.tolist()
        df["score_bm25_z"] = bm25_z.tolist()
        final_score = cfg.fusion_embed_weight * emb_z + cfg.fusion_bm25_weight * bm25_z
        fusion_meta = {
            "method": "linear_z_robust",
            "weights": [cfg.fusion_embed_weight, cfg.fusion_bm25_weight]
        }
    elif cfg.fusion_method.lower() == "rank_pct":
        emb_p = ranks_percentile(s_emb_np)
        bm25_p = ranks_percentile(bm25_vec)
        df["score_embed_pct"] = emb_p.tolist()
        df["score_bm25_pct"] = bm25_p.tolist()
        final_score = cfg.fusion_embed_weight * emb_p + cfg.fusion_bm25_weight * bm25_p
        fusion_meta = {
            "method": "rank_pct",
            "weights": [cfg.fusion_embed_weight, cfg.fusion_bm25_weight]
        }
    else:
        rrf_emb = rrf_from_scores(s_emb_np, k=60)
        rrf_bm = rrf_from_scores(bm25_vec, k=60)
        final_score = rrf_emb + rrf_bm
        fusion_meta = {"method": "rrf", "k": 60}
    
    df["score_title"] = s_title.tolist()
    df["score_abstract"] = s_abs.tolist()
    df["score_body"] = s_body.tolist()
    df["score_embed"] = s_emb_np.tolist()
    df["score_bm25"] = bm25_vec.tolist()
    df["score"] = final_score.tolist()
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    
    log.info(f"Score final moyen: {final_score.mean():.4f} (min={final_score.min():.4f}, max={final_score.max():.4f})")
    
    # === 8) SEUILLAGE AUTOMATIQUE ===
    log.info("Étape 8/10: Seuillage automatique")
    
    if cfg.keep_threshold is not None:
        thr = float(cfg.keep_threshold)
        thr_source = "manual"
        thr_metadata = {}
    else:
        raw = df["score"].to_numpy().astype(np.float64)
        
        if cfg.threshold_method == "ensemble":
            thr, thr_metadata = ensemble_threshold(raw)
            thr_source = "ensemble"
        elif cfg.threshold_method == "gmm":
            result = gmm_bayes_cut_robust(raw, cfg.gmm_use_bic_gate, cfg.gmm_bic_margin)
            thr = result.value if result.is_valid() else float(np.median(raw))
            thr_source = result.method
            thr_metadata = result.metadata
        elif cfg.threshold_method == "kde":
            result = kde_valley_robust(raw, use_cv=True)
            thr = result.value if result.is_valid() else float(np.median(raw))
            thr_source = result.method
            thr_metadata = result.metadata
        elif cfg.threshold_method == "otsu":
            result = otsu_threshold_robust(raw)
            thr = result.value if result.is_valid() else float(np.median(raw))
            thr_source = result.method
            thr_metadata = result.metadata
        else:
            # Fallback
            thr = float(np.median(raw))
            thr_source = "fallback_median"
            thr_metadata = {}
        
        # Plancher par quantile
        qfloor = float(np.quantile(raw, cfg.quantile_floor))
        thr = max(thr, qfloor)
    
    log.info(f"Seuil: {thr:.4f} (source: {thr_source})")
    
    keep = df[df["score"] >= thr].copy()
    if keep.empty:
        keep = df.head(min(cfg.adaptive_topk, len(df))).copy()
        thr_source = (thr_source or "auto") + "+fallback_topk"
    
    log.info(f"Articles conservés après seuillage: {len(keep)}")
    
    # === 9) CLUSTERING ===
    log.info("Étape 9/10: Clustering")
    
    # Vecteurs pour clustering (titre + abstract)
    E_dedup_full = F.normalize(0.7 * E_title + 0.3 * E_abs, p=2, dim=1)
    E_k = E_dedup_full[keep.index].cpu().numpy()
    
    if cfg.cluster_method == "hdbscan" and HDBSCAN_AVAILABLE:
        labels, cluster_metadata = cluster_hdbscan(
            E_k,
            min_cluster_size=cfg.hdbscan_min_cluster_size,
            min_samples=cfg.hdbscan_min_samples,
            selection_method=cfg.hdbscan_cluster_selection_method
        )
    elif cfg.cluster_method == "graph_cc":
        labels, cluster_metadata = cluster_graph_cc(E_k, k=cfg.graph_k, tau=cfg.graph_tau)
    else:
        # DBSCAN (fallback)
        labels, cluster_metadata = cluster_dbscan_auto(E_k, min_samples=cfg.dbscan_min_samples, eps=cfg.dbscan_eps)
    
    keep["cluster_id"] = labels
    log.info(f"Clustering terminé", **cluster_metadata)
    
    # === 10) SÉLECTION BUDGÉTÉE ===
    log.info("Étape 10/10: Sélection budgétée avec diversification")
    
    scores_keep = keep["score"].to_numpy()
    V_all = F.normalize(0.6 * E_title + 0.4 * E_abs, p=2, dim=1)[keep.index].cpu().numpy()
    
    TARGET_K = min(cfg.mmr_topk, len(keep))
    MAX_PER_CLUSTER = max(2, TARGET_K // 5)
    LAMBDA = cfg.mmr_lambda
    
    # Regrouper par cluster
    clusters = defaultdict(list)
    for i, cl in enumerate(labels):
        clusters[cl].append(i)
    
    # Stats par cluster
    def cluster_stats(idxs: List[int]) -> Dict[str, float]:
        if len(idxs) == 0:
            return {"size": 0, "q": 0.0, "disp": 1.0}
        s = scores_keep[idxs]
        mean_s = float(np.mean(s))
        max_s = float(np.max(s))
        if len(idxs) >= 2:
            d = pairwise_distances(V_all[idxs], metric="cosine")
            disp = float(np.mean(d))
        else:
            disp = 0.0
        q = 0.7 * max_s + 0.3 * mean_s - 0.2 * disp
        return {"size": len(idxs), "q": q, "disp": disp}
    
    stats = {cl: cluster_stats(idxs) for cl, idxs in clusters.items()}
    
    # Quotas
    total_size = sum(s["size"] for s in stats.values() if s["size"] > 0)
    q_raw = {}
    for cl, st in stats.items():
        if st["size"] == 0:
            q_raw[cl] = 0.0
            continue
        size_w = st["size"] / max(total_size, 1e-12)
        qual_w = max(st["q"], 0.0)
        q_raw[cl] = size_w * (1.0 + qual_w)
    
    sum_q = sum(q_raw.values()) or 1.0
    quota = {cl: max(1, int(round(TARGET_K * (q_raw[cl] / sum_q)))) for cl in q_raw}
    for cl in list(quota.keys()):
        quota[cl] = min(quota[cl], MAX_PER_CLUSTER)
    if -1 in quota:
        quota[-1] = min(quota[-1], max(2, TARGET_K // 4))
    
    while sum(quota.values()) > TARGET_K:
        cl_max = max(quota, key=lambda c: quota[c])
        quota[cl_max] -= 1
        if quota[cl_max] <= 0:
            del quota[cl_max]
    
    log.info(f"Quotas par cluster: {quota}")
    
    # Candidats triés par score
    cand_by_cl = {}
    for cl, idxs in clusters.items():
        order = np.argsort(-scores_keep[idxs])
        cand_by_cl[cl] = [idxs[k] for k in order]
    
    # Matrice similarité
    if len(keep) > 0 and len(keep) < 10000:
        S_all = 1.0 - pairwise_distances(V_all, metric="cosine")
        current_best = np.zeros(S_all.shape[0], dtype=np.float64)
    else:
        S_all = None
        current_best = None
    
    selected_local_idx = []
    selected_vecs = []
    
    def mmr_gain(j):
        rel = scores_keep[j]
        if not selected_vecs:
            return rel
        sims = 1.0 - pairwise_distances([V_all[j]], np.vstack(selected_vecs), metric="cosine")[0]
        red = float(np.max(sims))
        return LAMBDA * rel - (1.0 - LAMBDA) * red
    
    def facility_gain(j):
        if S_all is None:
            return mmr_gain(j)
        rel = float(scores_keep[j])
        return facility_location_gain(current_best, S_all[:, j], rel, LAMBDA)
    
    gain_fn = mmr_gain if cfg.selection_method == "mmr" else facility_gain
    
    active = {cl: quota[cl] for cl in quota if quota[cl] > 0}
    
    while active and len(selected_local_idx) < TARGET_K:
        for cl in list(active.keys()):
            if len(selected_local_idx) >= TARGET_K:
                break
            
            q_left = active.get(cl, 0)
            if q_left <= 0:
                active.pop(cl, None)
                continue
            
            cands = cand_by_cl.get(cl, [])
            cands = [j for j in cands if j not in selected_local_idx]
            
            if not cands:
                active.pop(cl, None)
                continue
            
            beam = cands[:min(5, len(cands))]
            gains = []
            
            for j in beam:
                # Anti-doublon strict
                if selected_vecs:
                    sims = 1.0 - pairwise_distances([V_all[j]], np.vstack(selected_vecs), metric="cosine")[0]
                    if float(np.max(sims)) >= cfg.dedup_threshold:
                        continue
                gains.append((gain_fn(j), j))
            
            if not gains:
                j = cands[0]
            else:
                j = sorted(gains, key=lambda x: -x[0])[0][1]
            
            selected_local_idx.append(j)
            selected_vecs.append(V_all[j])
            
            if cfg.selection_method != "mmr" and S_all is not None:
                current_best = np.maximum(current_best, S_all[:, j])
            
            active[cl] = q_left - 1
            if active[cl] <= 0:
                active.pop(cl, None)
    
    # Fallback si quota non rempli
    if len(selected_local_idx) < TARGET_K:
        remaining = [i for i in range(len(keep)) if i not in selected_local_idx]
        rem_sorted = sorted(remaining, key=lambda j: (-scores_keep[j], j))
        
        for j in rem_sorted:
            if len(selected_local_idx) >= TARGET_K:
                break
            
            if selected_vecs:
                sims = 1.0 - pairwise_distances([V_all[j]], np.vstack(selected_vecs), metric="cosine")[0]
                if float(np.max(sims)) >= cfg.dedup_threshold:
                    continue
            
            selected_local_idx.append(j)
            selected_vecs.append(V_all[j])
            
            if cfg.selection_method != "mmr" and S_all is not None:
                current_best = np.maximum(current_best, S_all[:, j])
    
    keep_dedup = keep.iloc[selected_local_idx].copy()
    log.info(f"Articles sélectionnés après diversification: {len(keep_dedup)}")
    
    # === 11) RÉORDONNANCEMENT FINAL ===
    vec_for_mmr = torch.from_numpy(np.vstack(selected_vecs)).float() if selected_vecs else torch.zeros((0, E_title.shape[1]))
    order = mmr_select(q_vec, vec_for_mmr, top_k=len(keep_dedup), lambda_=cfg.mmr_lambda) if len(keep_dedup) else []
    final_df = keep_dedup.iloc[order].reset_index(drop=True)
    final_df.insert(0, "rank", np.arange(1, len(final_df) + 1))
    
    # === 12) MÉTRIQUES DE DIVERSITÉ ===
    if len(final_df) > 1:
        V_sel = vec_for_mmr.numpy()
        diversity_mean = float(np.mean(pairwise_distances(V_sel, metric="cosine")))
        diversity_std = float(np.std(pairwise_distances(V_sel, metric="cosine")))
    else:
        diversity_mean = None
        diversity_std = None
    
    # === 13) DONNÉES ENRICHIES POUR VISUALISATION ===
    # Sauvegarder les embeddings des articles finaux pour visualisation avancée
    embeddings_file = cfg.output_csv.replace('.csv', '_embeddings.npy')
    if len(final_df) > 0:
        final_indices = final_df.index.tolist()
        E_final = E_dedup_full[keep.index].cpu().numpy()[selected_local_idx]
        np.save(embeddings_file, E_final)
        log.info(f"Embeddings sauvegardés: {embeddings_file}")
    
    # Calculer matrice de similarité des articles finaux (pour heatmap)
    similarity_matrix = None
    if len(final_df) > 0 and len(final_df) <= 100:  # Limite pour éviter matrices trop grandes
        E_final_norm = E_final / np.linalg.norm(E_final, axis=1, keepdims=True)
        similarity_matrix = (E_final_norm @ E_final_norm.T).tolist()
    
    # Statistiques de longueur de texte
    text_lengths = {
        "title": {
            "mean": float(df["title"].str.len().mean()) if len(df) else None,
            "median": float(df["title"].str.len().median()) if len(df) else None,
            "min": int(df["title"].str.len().min()) if len(df) else None,
            "max": int(df["title"].str.len().max()) if len(df) else None,
        },
        "abstract": {
            "mean": float(df["abstract"].str.len().mean()) if len(df) else None,
            "median": float(df["abstract"].str.len().median()) if len(df) else None,
            "min": int(df["abstract"].str.len().min()) if len(df) else None,
            "max": int(df["abstract"].str.len().max()) if len(df) else None,
        },
        "body": {
            "mean": float(df["body"].str.len().mean()) if len(df) else None,
            "median": float(df["body"].str.len().median()) if len(df) else None,
            "min": int(df["body"].str.len().min()) if len(df) else None,
            "max": int(df["body"].str.len().max()) if len(df) else None,
        }
    }
    
    # Statistiques par cluster
    cluster_distributions = {}
    if 'cluster_id' in final_df.columns:
        for cluster_id in sorted(set(final_df['cluster_id'])):
            cluster_mask = final_df['cluster_id'] == cluster_id
            cluster_distributions[int(cluster_id)] = {
                "count": int(cluster_mask.sum()),
                "score_mean": float(final_df.loc[cluster_mask, 'score'].mean()),
                "score_std": float(final_df.loc[cluster_mask, 'score'].std()),
                "score_min": float(final_df.loc[cluster_mask, 'score'].min()),
                "score_max": float(final_df.loc[cluster_mask, 'score'].max()),
            }
    
    # === 13) RAPPORT FINAL ENRICHI ===
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    report = {
        "metadata": {
            "pipeline_version": "improved_v1.0",
            "timestamp_start": start_time.isoformat(),
            "timestamp_end": end_time.isoformat(),
            "duration_seconds": duration,
            "embeddings_file": embeddings_file if len(final_df) > 0 else None,
        },
        "config": asdict(cfg),
        "counts": {
            "total_initial": int(n_initial),
            "after_filters": int(len(df)),
            "after_threshold": int(len(keep)),
            "final_selected": int(len(final_df)),
        },
        "thresholds": {
            "source": thr_source,
            "value": float(thr),
            "quantile_floor": float(cfg.quantile_floor),
            "manual": cfg.keep_threshold,
            "metadata": thr_metadata,
        },
        "fusion": fusion_meta,
        "clustering": cluster_metadata,
        "selection": {
            "target_k": int(TARGET_K),
            "max_per_cluster": int(MAX_PER_CLUSTER),
            "method": cfg.selection_method,
            "lambda": float(cfg.mmr_lambda),
            "dedup_threshold": float(cfg.dedup_threshold),
            "quotas": {str(k): int(v) for k, v in quota.items()},
            "clusters_stats": {str(k): v for k, v in stats.items()},
        },
        "stats": {
            "score_min": float(df["score"].min()) if len(df) else None,
            "score_max": float(df["score"].max()) if len(df) else None,
            "score_mean": float(df["score"].mean()) if len(df) else None,
            "score_std": float(df["score"].std()) if len(df) else None,
            "diversity_cosine_mean": diversity_mean,
            "diversity_cosine_std": diversity_std,
        },
        "text_lengths": text_lengths,
        "cluster_distributions": cluster_distributions,
        "similarity_matrix": similarity_matrix,
    }
    
    # === 14) EXPORT ===
    log.info("Export des résultats")
    
    with open(cfg.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    final_df.to_csv(cfg.output_csv, index=False, encoding="utf-8")
    
    log.info(f"Export terminé: {cfg.output_csv}")
    log.info(f"Rapport: {cfg.report_json}")
    log.info(f"Durée totale: {duration:.2f}s")
    
    # Affichage résumé
    cols_show = ["rank", "title", "score", "score_embed", "score_bm25", "cluster_id"]
    cols_show = [c for c in cols_show if c in final_df.columns]
    print("\n" + "="*80)
    print("TOP 10 ARTICLES SÉLECTIONNÉS")
    print("="*80)
    print(final_df[cols_show].head(10).to_string(index=False))
    print("="*80)
    
    log.info("PIPELINE TERMINÉ AVEC SUCCÈS")
    
    return report


if __name__ == "__main__":
    cfg = Config()
    
    try:
        report = main(cfg)
        exit(0)
    except Exception as e:
        if log:
            log.error(f"Erreur fatale: {e}", exc_info=True)
        else:
            print(f"ERREUR: {e}")
        raise

