import streamlit as st
from src.database import init_db

# Page Configuration
st.set_page_config(
    page_title="PRISMA Review Manager",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === DESIGN SYSTEM PREMIUM ===
from src.ui_utils import load_premium_css
load_premium_css()

# Initialize Database
if "db_initialized" not in st.session_state:
    init_db()
    st.session_state["db_initialized"] = True

from src.database import get_db, Article, SearchSession
from src.analytics import get_global_stats

# ==================== HEADER ====================
st.title("🏠 Dashboard PRISMA")
st.caption("Vue d'ensemble de votre revue systématique")

st.markdown("<hr>", unsafe_allow_html=True)

# ==================== STATISTIQUES GLOBALES ====================
db = next(get_db())
stats = get_global_stats(db)

# Dernière session
latest_session = db.query(SearchSession).order_by(SearchSession.id.desc()).first()

# Métriques principales
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "📚 Sessions", 
        db.query(SearchSession).count(),
        help="Nombre total de recherches effectuées"
    )

with col2:
    st.metric(
        "📥 Articles identifiés", 
        stats['identified'],
        help="Articles collectés depuis les bases de données"
    )

with col3:
    st.metric(
        "✅ Screenés", 
        stats['screened_in'],
        delta=f"{stats['screened_in']}/{stats['identified']}" if stats['identified'] > 0 else "0/0"
    )

with col4:
    st.metric(
        "🎯 Inclus", 
        stats['included'],
        delta=f"{stats['inclusion_rate']:.1f}%" if stats['identified'] > 0 else "0%",
        delta_color="normal"
    )

with col5:
    progress = 0
    if stats['identified'] > 0:
        progress = ((stats['screened_in'] + stats['screened_out'] + stats['excluded_eligibility']) / stats['identified']) * 100
    st.metric(
        "📊 Progression", 
        f"{progress:.0f}%",
        help="Pourcentage d'articles traités"
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ==================== DERNIÈRE SESSION ====================
if latest_session:
    st.subheader("🔬 Dernière Recherche")
    
    col_info, col_stats = st.columns([2, 1])
    
    with col_info:
        st.markdown(f"**Requête :** {latest_session.query}")
        st.caption(f"📅 Créée le {latest_session.created_at.strftime('%d/%m/%Y à %H:%M')}")
        
        # Bouton pour aller à cette session
        if st.button("📂 Ouvrir cette session dans Screening"):
            st.session_state.active_session_id = latest_session.id
            st.session_state.active_session_query = latest_session.query
            st.switch_page("pages/2_Screening.py")
    
    with col_stats:
        articles_session = db.query(Article).filter(Article.search_session_id == latest_session.id).count()
        st.metric("Articles", articles_session)
        st.metric("Résultats", latest_session.num_results)
else:
    st.info("👋 Aucune recherche effectuée. Commencez par l'onglet **Recherche** !")

st.markdown("<hr>", unsafe_allow_html=True)

# ==================== ACTIONS RAPIDES ====================
st.subheader("⚡ Actions Rapides")

col_a1, col_a2, col_a3 = st.columns(3)

with col_a1:
    if st.button("🔍 Nouvelle Recherche", use_container_width=True):
        st.switch_page("pages/1_Recherche.py")

with col_a2:
    if st.button("📋 Screening", use_container_width=True):
        st.switch_page("pages/2_Screening.py")

with col_a3:
    if st.button("📊 Voir Analyse PRISMA", use_container_width=True):
        st.switch_page("pages/4_Analyse.py")

st.markdown("<hr>", unsafe_allow_html=True)

# ==================== GUIDE RAPIDE ====================
with st.expander("📖 Guide Rapide PRISMA"):
    st.markdown("""
    ### Workflow en 4 Étapes
    
    1. **🔍 Recherche** : Collectez des articles depuis arXiv, PubMed, Crossref
    2. **📋 Screening** : Filtrez les articles (titre, abstract, texte complet)
    3. **📝 Éligibilité** : Validation finale sur texte complet
    4. **📊 Analyse** : Diagramme PRISMA et exports
    
    **Astuce :** Utilisez le filtre sémantique dans Screening pour gagner du temps !
    """)

db.close()
