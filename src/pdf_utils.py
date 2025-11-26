"""
Module d'extraction de texte depuis des PDFs pour articles scientifiques.
Utilise PyMuPDF (fitz) pour une extraction intelligente avec détection de colonnes.
"""

import fitz  # PyMuPDF
import re
import os
from typing import Tuple, Optional


def extract_text_from_pdf(pdf_path: str) -> Tuple[Optional[str], str, str]:
    """
    Extrait le texte complet d'un PDF avec détection intelligente de colonnes.
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        
    Returns:
        Tuple (texte_extrait, statut, méthode)
        - texte_extrait: str ou None si échec
        - statut: "SUCCESS", "FAILED", "NOT_ATTEMPTED"
        - méthode: "pymupdf" ou "error"
    
    Exemples:
        >>> text, status, method = extract_text_from_pdf("article.pdf")
        >>> if status == "SUCCESS":
        ...     print(f"Extrait {len(text)} caractères avec {method}")
    """
    if not os.path.exists(pdf_path):
        return None, "FAILED", "error"
    
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        
        for page_num, page in enumerate(doc):
            # Extraction avec tri spatial (top-to-bottom, left-to-right)
            blocks = page.get_text("blocks", sort=True)
            
            # Séparer les colonnes si nécessaire
            columns = _detect_and_separate_columns(blocks, page.rect.width)
            
            # Extraire le texte colonne par colonne
            for column_blocks in columns:
                for block in column_blocks:
                    # block[4] contient le texte du bloc
                    if len(block) > 4 and block[4].strip():
                        full_text.append(block[4].strip())
        
        doc.close()
        
        # Joindre tout le texte avec double saut de ligne entre blocs
        text = "\n\n".join(full_text)
        
        # Post-traitement
        text = _clean_scientific_text(text)
        
        # Vérifier la qualité de l'extraction
        if _is_extraction_valid(text):
            return text, "SUCCESS", "pymupdf"
        else:
            return None, "FAILED", "pymupdf"
            
    except Exception as e:
        print(f"Erreur lors de l'extraction PDF {pdf_path}: {e}")
        return None, "FAILED", "error"


def _detect_and_separate_columns(blocks, page_width: float) -> list:
    """
    Détecte si la page a plusieurs colonnes et les sépare.
    
    Args:
        blocks: Liste des blocs de texte extraits
        page_width: Largeur de la page
        
    Returns:
        Liste de listes de blocs, une par colonne
    """
    if not blocks:
        return [[]]
    
    # Extraire les positions X des blocs
    x_positions = [block[0] for block in blocks if len(block) > 4]
    
    if not x_positions:
        return [[]]
    
    # Heuristique simple : si des blocs commencent à ~50% de la largeur,
    # on a probablement deux colonnes
    mid_page = page_width / 2
    left_blocks = [b for b in blocks if len(b) > 4 and b[0] < mid_page - 50]
    right_blocks = [b for b in blocks if len(b) > 4 and b[0] >= mid_page - 50]
    
    # Si on a des blocs dans les deux zones, c'est du multi-colonnes
    if left_blocks and right_blocks and len(right_blocks) > 3:
        # Trier chaque colonne verticalement
        left_blocks.sort(key=lambda b: b[1])  # Tri par position Y
        right_blocks.sort(key=lambda b: b[1])
        return [left_blocks, right_blocks]
    else:
        # Une seule colonne
        return [blocks]


def _clean_scientific_text(text: str) -> str:
    """
    Nettoie le texte extrait des artefacts courants dans les PDFs scientifiques.
    
    Args:
        text: Texte brut extrait
        
    Returns:
        Texte nettoyé
    """
    # Enlever les tirets de césure en fin de ligne
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Normaliser les espaces multiples
    text = re.sub(r' +', ' ', text)
    
    # Normaliser les sauts de ligne multiples (max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Enlever les espaces en début/fin de lignes
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


def _is_extraction_valid(text: str) -> bool:
    """
    Vérifie si l'extraction semble valide (heuristiques de qualité).
    
    Args:
        text: Texte extrait
        
    Returns:
        True si l'extraction semble bonne, False sinon
    """
    if not text or len(text) < 100:
        return False
    
    # Compter les caractères alphabétiques vs non-alphabétiques
    alpha_count = sum(c.isalpha() for c in text)
    total_count = len(text)
    
    if total_count == 0:
        return False
    
    alpha_ratio = alpha_count / total_count
    
    # Un article scientifique devrait avoir au moins 50% de caractères alphabétiques
    if alpha_ratio < 0.5:
        return False
    
    # Vérifier qu'il y a des mots (pas juste des symboles)
    words = text.split()
    if len(words) < 50:  # Au moins 50 mots pour un article
        return False
    
    return True


def get_pdf_metadata(pdf_path: str) -> dict:
    """
    Extrait les métadonnées d'un PDF.
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        
    Returns:
        Dictionnaire avec les métadonnées (titre, auteur, pages, etc.)
    """
    try:
        doc = fitz.open(pdf_path)
        metadata = {
            "pages": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "keywords": doc.metadata.get("keywords", ""),
        }
        doc.close()
        return metadata
    except Exception as e:
        return {"error": str(e)}
