import streamlit as st
from src.database import init_db

# Page Configuration
st.set_page_config(
    page_title="PRISMA Review Manager",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Database
if "db_initialized" not in st.session_state:
    init_db()
    st.session_state["db_initialized"] = True

# Sidebar Navigation
st.sidebar.title("PRISMA Workflow")

st.title("PRISMA Systematic Review Manager")

st.markdown("""
### Bienvenue dans votre gestionnaire de revue systématique.

Cette application vous guide à travers les étapes de la méthodologie PRISMA :

1.  **Recherche (Identification)** : Collectez des articles depuis Google Scholar.
2.  **Screening** : Filtrez les articles sur base du titre et du résumé.
3.  **Éligibilité** : Validez les articles sur base du texte complet.
4.  **Dashboard** : Visualisez le diagramme de flux PRISMA en temps réel.

**Utilisez le menu à gauche pour naviguer entre les étapes.**
""")

# Quick Stats (Placeholder)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Articles Identifiés", "0")
col2.metric("À Screener", "0")
col3.metric("À Valider", "0")
col4.metric("Inclus", "0")
