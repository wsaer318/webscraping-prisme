"""
UI Utilities - Helpers pour l'interface Streamlit
Centralise le chargement du design system CSS
"""
import streamlit as st
from pathlib import Path

def load_premium_css():
    """
    Charge le design system premium depuis static/styles/premium.css
    À appeler au début de chaque page Streamlit
    """
    css_file = Path(__file__).parent.parent / "static" / "styles" / "premium.css"
    
    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Fichier CSS premium non trouvé: {css_file}")
