import streamlit as st
from src.database import get_db, Article, SearchSession, delete_article_with_cleanup, delete_session_with_cleanup, init_db
import pandas as pd
from datetime import datetime
import os
import json

st.set_page_config(page_title="Base de données", layout="wide")

st.title("Base de données des Articles")

# Initialize database
init_db()
db = next(get_db())

# Créer des onglets
tab1, tab2 = st.tabs(["Sessions de Recherche", "Vue Globale"])

# ===================== ONGLET 1: SESSIONS =====================
with tab1:
    st.header("Gestion par Sessions de Recherche")
    
    # Récupérer toutes les sessions actives
    sessions = db.query(SearchSession).filter(SearchSession.status == 'ACTIVE').order_by(SearchSession.created_at.desc()).all()
    
    if not sessions:
        st.info("Aucune session de recherche active. Lancez une recherche pour créer une session.")
    else:
        for session in sessions:
            with st.expander(f"[#{session.id}] {session.query} - {session.num_results} articles ({session.created_at.strftime('%Y-%m-%d %H:%M')})", expanded=True):
                col_info, col_actions = st.columns([3, 1])
                
                with col_info:
                    st.markdown(f"""
                    **Requête** : `{session.query}`  
                    **Résultats** : {session.num_results} articles trouvés  
                    **PDFs téléchargés** : {session.successful_downloads}/{session.num_results}  
                    **Date** : {session.created_at.strftime('%Y-%m-%d à %H:%M')}
                    """)
                
                with col_actions:
                    if st.button("Supprimer session", key=f"del_session_{session.id}", type="secondary"):
                        if st.session_state.get(f"confirm_del_session_{session.id}"):
                            delete_session_with_cleanup(session.id, db)
                            st.success("Session supprimée!")
                            st.rerun()
                        else:
                            st.session_state[f"confirm_del_session_{session.id}"] = True
                            st.warning("Cliquez à nouveau pour confirmer")
                
                # Afficher les articles de cette session
                articles = db.query(Article).filter(Article.search_session_id == session.id).all()
                
                if articles:
                    df = pd.DataFrame([
                        {
                            "ID": a.id,
                            "Titre": a.title[:60] + "..." if len(a.title) > 60 else a.title,
                            "Année": a.year,
                            "Source": a.source,
                            "Statut": a.status,
                            "Extraction": a.text_extraction_status
                        }
                        for a in articles
                    ])
                    
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("Aucun article dans cette session")

# ===================== ONGLET 2: VUE GLOBALE =====================
with tab2:
    st.header("Vue Globale de Tous les Articles")
    
    # === STATISTIQUES ===
    total = db.query(Article).count()
    identified = db.query(Article).filter(Article.status == "IDENTIFIED").count()
    screened_in = db.query(Article).filter(Article.status == "SCREENED_IN").count()
    eligible = db.query(Article).filter(Article.status == "ELIGIBLE").count()
    included = db.query(Article).filter(Article.status == "INCLUDED").count()
    text_extracted = db.query(Article).filter(Article.text_extraction_status == "SUCCESS").count()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total", total)
    col2.metric("Identifiés", identified)
    col3.metric("Éligibles", eligible)
    col4.metric("Inclus", included)
    col5.metric("Texte extrait", text_extracted)
    
    st.divider()
    
    # === FILTRES ===
    st.subheader("Filtres")
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        status_filter = st.selectbox("Statut PRISMA", [
            "Tous",
            "IDENTIFIED",
            "SCREENED_IN",
            "EXCLUDED_SCREENING",
            "ELIGIBLE",
            "EXCLUDED_ELIGIBILITY",
            "INCLUDED"
        ])
    
    with col_filter2:
        extraction_filter = st.selectbox("Extraction", [
            "Tous",
            "SUCCESS",
            "FAILED",
            "NOT_ATTEMPTED"
        ])
    
    with col_filter3:
        search_query = st.text_input("Rechercher (titre/auteurs)")
    
    # Construire la requête
    query = db.query(Article)
    
    if status_filter != "Tous":
        query = query.filter(Article.status == status_filter)
    
    if extraction_filter != "Tous":
        query = query.filter(Article.text_extraction_status == extraction_filter)
    
    if search_query:
        query = query.filter(
            (Article.title.contains(search_query)) | (Article.authors.contains(search_query))
        )
    
    articles = query.all()
    
    st.markdown(f"**{len(articles)} article(s) trouvé(s)**")
    
    # === TABLEAU GLOBAL ===
    if articles:
        df_all = pd.DataFrame([
            {
                "ID": a.id,
                "Titre": a.title,
                "Auteurs": a.authors,
                "Année": a.year,
                "Source": a.source,
                "Abstract": (a.abstract[:100] + "...") if a.abstract else "N/A",
                "DOI": a.doi if a.doi else "N/A",
                "Statut": a.status,
                "Extraction": a.text_extraction_status,
                "Texte (aperçu)": (a.full_text[:100] + "...") if a.full_text else "N/A",
                "Session": a.search_session_id if a.search_session_id else "N/A",
                "Date": a.created_at
            }
            for a in articles
        ])
        
        st.dataframe(df_all, use_container_width=True, hide_index=True)
        
        # === EXPORT ===
        st.divider()
        st.subheader("Export")
        
        # Préparer les données complètes pour l'export
        df_export = pd.DataFrame([
            {
                "ID": a.id,
                "Titre": a.title,
                "Auteurs": a.authors,
                "Année": a.year,
                "Source": a.source,
                "Abstract": a.abstract,
                "DOI": a.doi,
                "Lien": a.link,
                "Statut": a.status,
                "Extraction": a.text_extraction_status,
                "Texte Complet": a.full_text,
                "Session": a.search_session_id,
                "Date": a.created_at
            }
            for a in articles
        ])
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger CSV (Complet)",
                data=csv,
                file_name=f"prisma_export_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col_export2:
            # Export JSON avec le texte complet
            articles_json = []
            for a in articles:
                articles_json.append({
                    "id": a.id,
                    "title": a.title,
                    "authors": a.authors,
                    "year": a.year,
                    "source": a.source,
                    "link": a.link,
                    "abstract": a.abstract,
                    "full_text": a.full_text,
                    "status": a.status,
                    "text_extraction_status": a.text_extraction_status,
                    "session_id": a.search_session_id
                })
            
            json_str = json.dumps(articles_json, indent=2, ensure_ascii=False)
            st.download_button(
                label="Télécharger JSON (avec texte)",
                data=json_str,
                file_name=f"prisma_full_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    else:
        st.info("Aucun article trouvé")
    
    # === ZONE DE DANGER ===
    st.divider()
    st.subheader("Zone de Danger")
    
    with st.expander("Actions destructrices"):
        st.warning("Attention : Ces actions sont irréversibles.")
        
        if st.button("Vider TOUTE la base de données", type="primary"):
            if st.session_state.get("confirm_wipe_db"):
                # Supprimer tous les PDFs
                all_articles = db.query(Article).all()
                for art in all_articles:
                    if art.pdf_path and os.path.exists(art.pdf_path):
                        os.remove(art.pdf_path)
                
                db.query(Article).delete()
                db.query(SearchSession).delete()
                db.commit()
                st.success("Base de données entièrement vidée!")
                st.session_state["confirm_wipe_db"] = False
                st.rerun()
            else:
                st.session_state["confirm_wipe_db"] = True
                st.warning("Cliquez à nouveau pour confirmer")

db.close()
