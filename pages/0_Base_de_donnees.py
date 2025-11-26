import streamlit as st
from src.database import get_db, Article, init_db
import pandas as pd
from datetime import datetime
import os

st.set_page_config(page_title="Base de données", layout="wide")

st.title("Base de données des Articles")
st.markdown("Visualisez, recherchez et gérez tous les articles de votre revue PRISMA.")

# Initialize database
init_db()
db = next(get_db())

# === STATISTIQUES GLOBALES ===
st.subheader("Statistiques")

# Compter par statut
total = db.query(Article).count()
identified = db.query(Article).filter(Article.status == "IDENTIFIED").count()
screened_in = db.query(Article).filter(Article.status == "SCREENED_IN").count()
excluded_screening = db.query(Article).filter(Article.status == "EXCLUDED_SCREENING").count()
eligible = db.query(Article).filter(Article.status == "ELIGIBLE").count()
excluded_eligibility = db.query(Article).filter(Article.status == "EXCLUDED_ELIGIBILITY").count()
included = db.query(Article).filter(Article.status == "INCLUDED").count()

# Extraction status
text_extracted = db.query(Article).filter(Article.text_extraction_status == "SUCCESS").count()
extraction_failed = db.query(Article).filter(Article.text_extraction_status == "FAILED").count()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total", total)
    st.metric("Texte extrait", text_extracted)
    
with col2:
    st.metric("Identifiés", identified)
    st.metric("Screenés", screened_in)

with col3:
    st.metric("Éligibles", eligible)
    st.metric("Inclus", included)
    
with col4:
    st.metric("Exclus (Screen)", excluded_screening)
    st.metric("Exclus (Élig.)", excluded_eligibility)
    
with col5:
    st.metric("Extraction échouée", extraction_failed)

st.divider()

# === FILTRES ===
st.subheader("Filtres et Recherche")

col_filter1, col_filter2, col_filter3 = st.columns(3)

with col_filter1:
    status_filter = st.selectbox(
        "Statut PRISMA",
        ["Tous", "IDENTIFIED", "SCREENED_IN", "EXCLUDED_SCREENING", "ELIGIBLE", "EXCLUDED_ELIGIBILITY", "INCLUDED"]
    )

with col_filter2:
    extraction_filter = st.selectbox(
        "Statut d'extraction",
        ["Tous", "SUCCESS", "FAILED", "NOT_ATTEMPTED"]
    )
    
with col_filter3:
    source_filter = st.selectbox(
        "Source",
        ["Toutes", "Google Scholar"]
    )

search_query = st.text_input("Rechercher dans titre ou auteurs", "")

st.divider()

# === REQUÊTE FILTRÉE ===
query = db.query(Article)

if status_filter != "Tous":
    query = query.filter(Article.status == status_filter)

if extraction_filter != "Tous":
    query = query.filter(Article.text_extraction_status == extraction_filter)
    
if source_filter != "Toutes":
    query = query.filter(Article.source == source_filter)

if search_query:
    query = query.filter(
        (Article.title.contains(search_query)) | 
        (Article.authors.contains(search_query))
    )

articles = query.order_by(Article.created_at.desc()).all()

# === LISTE DES ARTICLES ===
st.subheader(f"Articles ({len(articles)} résultats)")

if len(articles) == 0:
    st.info("Aucun article trouvé avec ces critères.")
else:
    # Pagination
    articles_per_page = 10
    total_pages = (len(articles) - 1) // articles_per_page + 1
    
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    
    start_idx = (page - 1) * articles_per_page
    end_idx = min(start_idx + articles_per_page, len(articles))
    
    for article in articles[start_idx:end_idx]:
        with st.expander(f"**{article.title}** ({article.status})"):
            col_info, col_actions = st.columns([3, 1])
            
            with col_info:
                st.markdown(f"**ID:** {article.id}")
                st.markdown(f"**Auteurs:** {article.authors}")
                st.markdown(f"**Source:** {article.source}")
                st.markdown(f"**Lien:** [{article.link[:50]}...]({article.link})" if article.link else "Pas de lien")
                st.markdown(f"**Statut PRISMA:** `{article.status}`")
                
                if article.abstract:
                    st.markdown("**Abstract:**")
                    st.info(article.abstract[:300] + "..." if len(article.abstract) > 300 else article.abstract)
                
                # Informations d'extraction
                st.markdown(f"**Extraction de texte:** `{article.text_extraction_status}` "
                          f"({article.extraction_method if article.extraction_method else 'N/A'})")
                
                if article.full_text:
                    word_count = len(article.full_text.split())
                    st.markdown(f"**Texte complet:** {word_count} mots, {len(article.full_text)} caractères")
                    
                    if st.checkbox(f"Afficher le texte complet (Article {article.id})", key=f"show_text_{article.id}"):
                        st.text_area(
                            "Texte extrait",
                            article.full_text[:2000] + "\n\n... (truncated)" if len(article.full_text) > 2000 else article.full_text,
                            height=300,
                            key=f"text_{article.id}"
                        )
                
                if article.pdf_path and os.path.exists(article.pdf_path):
                    st.markdown(f"**PDF:** `{os.path.basename(article.pdf_path)}`")
                    
                    # Bouton de téléchargement
                    with open(article.pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="Télécharger le PDF",
                            data=pdf_file,
                            file_name=os.path.basename(article.pdf_path),
                            mime="application/pdf",
                            key=f"download_{article.id}"
                        )
                
                st.markdown(f"**Créé le:** {article.created_at.strftime('%Y-%m-%d %H:%M') if article.created_at else 'N/A'}")
                
            with col_actions:
                st.markdown("**Actions**")
                
                # Bouton éditer (modal simple)
                if st.button("Éditer", key=f"edit_{article.id}", use_container_width=True):
                    st.session_state[f"editing_{article.id}"] = True
                
                # Bouton supprimer
                if st.button("Supprimer", key=f"delete_{article.id}", use_container_width=True):
                    if st.session_state.get(f"confirm_delete_{article.id}"):
                        db.delete(article)
                        db.commit()
                        st.success("Article supprimé !")
                        st.rerun()
                    else:
                        st.session_state[f"confirm_delete_{article.id}"] = True
                        st.warning("Cliquez à nouveau pour confirmer")
            
            # Modal d'édition
            if st.session_state.get(f"editing_{article.id}"):
                st.markdown("---")
                st.markdown("### Édition")
                
                with st.form(key=f"edit_form_{article.id}"):
                    new_title = st.text_input("Titre", value=article.title)
                    new_authors = st.text_input("Auteurs", value=article.authors or "")
                    new_abstract = st.text_area("Abstract", value=article.abstract or "", height=150)
                    new_status = st.selectbox(
                        "Statut",
                        ["IDENTIFIED", "SCREENED_IN", "EXCLUDED_SCREENING", "ELIGIBLE", "EXCLUDED_ELIGIBILITY", "INCLUDED"],
                        index=["IDENTIFIED", "SCREENED_IN", "EXCLUDED_SCREENING", "ELIGIBLE", "EXCLUDED_ELIGIBILITY", "INCLUDED"].index(article.status)
                    )
                    new_notes = st.text_area("Notes", value=article.notes or "", height=100)
                    
                    col_save, col_cancel = st.columns(2)
                    
                    with col_save:
                        if st.form_submit_button("Sauvegarder", use_container_width=True):
                            article.title = new_title
                            article.authors = new_authors
                            article.abstract = new_abstract
                            article.status = new_status
                            article.notes = new_notes
                            db.commit()
                            st.session_state[f"editing_{article.id}"] = False
                            st.success("Article mis à jour !")
                            st.rerun()
                    
                    with col_cancel:
                        if st.form_submit_button("Annuler", use_container_width=True):
                            st.session_state[f"editing_{article.id}"] = False
                            st.rerun()
    
    st.caption(f"Page {page} sur {total_pages}")

st.divider()

# === EXPORT ===
st.subheader("Export des données")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    if st.button("Exporter en CSV (métadonnées)", use_container_width=True):
        # Export simple sans texte complet
        df = pd.DataFrame([
            {
                "id": a.id,
                "title": a.title,
                "authors": a.authors,
                "source": a.source,
                "link": a.link,
                "status": a.status,
                "extraction_status": a.text_extraction_status,
                "created_at": a.created_at
            }
            for a in query.all()
        ])
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="💾 Télécharger CSV",
            data=csv,
            file_name=f"prisma_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col_exp2:
    if st.button("Exporter en JSON (complet)", use_container_width=True):
        # Export complet avec texte
        import json
        
        data = [
            {
                "id": a.id,
                "title": a.title,
                "authors": a.authors,
                "source": a.source,
                "link": a.link,
                "abstract": a.abstract,
                "full_text": a.full_text,
                "status": a.status,
                "extraction_status": a.text_extraction_status,
                "extraction_method": a.extraction_method,
                "pdf_path": a.pdf_path,
                "created_at": a.created_at.isoformat() if a.created_at else None
            }
            for a in query.all()
        ]
        
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        st.download_button(
            label="Télécharger JSON",
            data=json_str,
            file_name=f"prisma_export_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# === VUE TABLEAU ===
st.subheader("Vue d'ensemble (Tableau)")
df_all = pd.DataFrame([
    {
        "ID": a.id,
        "Titre": a.title,
        "Auteurs": a.authors,
        "Année": a.year,
        "Source": a.source,
        "Statut": a.status,
        "Extraction": a.text_extraction_status,
        "Date": a.created_at
    }
    for a in query.all()
])

if not df_all.empty:
    st.dataframe(df_all, use_container_width=True, hide_index=True)
else:
    st.info("Aucune donnée à afficher.")

st.divider()

# === ZONE DE DANGER ===
st.subheader("Zone de Danger")
with st.expander("Actions destructrices", expanded=False):
    st.warning("Attention : Ces actions sont irréversibles.")
    
    if st.button("Vider TOUTE la base de données", type="primary", use_container_width=True):
        if st.session_state.get("confirm_wipe_db"):
            try:
                # Supprimer tous les articles
                db.query(Article).delete()
                db.commit()
                st.success("Base de données entièrement vidée !")
                st.session_state["confirm_wipe_db"] = False
                st.rerun()
            except Exception as e:
                st.error(f"Erreur lors du vidage de la base : {e}")
        else:
            st.session_state["confirm_wipe_db"] = True
            st.warning("Êtes-vous sûr ? Cliquez à nouveau pour confirmer la suppression de TOUS les articles.")

db.close()
