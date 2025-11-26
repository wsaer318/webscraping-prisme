import streamlit as st
import pandas as pd
import os
from src.database import get_db, Article, SearchSession, ArticleHistory

st.set_page_config(page_title="Screening", layout="wide")

def save_decision(article_id, status, reason=None):
    db = next(get_db())
    article = db.query(Article).filter(Article.id == article_id).first()
    if article:
        # Enregistrer l'historique
        history = ArticleHistory(
            article_id=article.id,
            previous_status=article.status,
            new_status=status,
            action="SCREENING_DECISION",
            reason=reason if reason else "Inclusion Screening",
            user="User" # Pourrait être dynamique plus tard
        )
        db.add(history)
        
        # Mettre à jour l'article
        article.status = status
        if reason:
            article.exclusion_reason = reason
        db.commit()
    db.close()
    st.rerun()

# CSS pour améliorer l'affichage du texte
st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTextArea textarea {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Screening Unifié")
st.markdown("Triez les articles sur la base du **Titre**, de l'**Abstract** et du **Texte Complet**.")

db = next(get_db())
# Récupérer les articles à screener (IDENTIFIED)
articles = db.query(Article).filter(Article.status == "IDENTIFIED").all()

if not articles:
    st.info("Aucun article à screener ! Allez dans l'onglet 'Recherche' pour en ajouter.")
    db.close()
else:
    # --- LOGIQUE DE TRI & SUGGESTIONS (LECTURE BDD) ---
    
    # Récupérer la requête de la session (Restauré)
    session_query = "machine learning" # Fallback
    if articles and articles[0].search_session_id:
        session = db.query(SearchSession).filter(SearchSession.id == articles[0].search_session_id).first()
        if session:
            session_query = session.query.split('[')[0].strip()

    # Vérifier si des scores sont disponibles
    scores_available = any(a.relevance_score is not None for a in articles)
    
    # Si des scores sont manquants, on peut proposer de relancer (cas de vieux articles)
    missing_scores_count = sum(1 for a in articles if a.relevance_score is None)
    
    if missing_scores_count > 0:
        st.sidebar.warning(f"⚠️ {missing_scores_count} articles en cours d'analyse ou sans score.")
        if st.sidebar.button("Forcer l'analyse IA (Arrière-plan)"):
            import threading
            from src.advanced_sorting import AdvancedRanker
            def run_force_analysis(ids, q):
                ranker = AdvancedRanker()
                ranker.process_batch_and_update_db(ids, q)
            
            ids_missing = [a.id for a in articles if a.relevance_score is None]
            thread = threading.Thread(target=run_force_analysis, args=(ids_missing, session_query))
            thread.start()
            st.toast("Analyse forcée lancée en arrière-plan !")
            st.rerun()
    
    # Récupérer les scores pour le tri
    # On utilise une valeur par défaut (-1) pour les articles sans score pour qu'ils soient à la fin (ou au début ?)
    # Disons au début pour qu'on les voit arriver.
    article_scores = {a.id: (a.relevance_score if a.relevance_score is not None else -1.0) for a in articles}
    
    # Trier par score
    articles.sort(key=lambda a: article_scores.get(a.id, -1.0), reverse=True)
    
    # --- SCREENING ASSISTÉ (SUGGESTIONS) ---
    st.sidebar.header("Screening Assisté (IA)")
    
    # Calcul dynamique du seuil si on a des scores
    valid_scores = [s for s in article_scores.values() if s >= 0]
    
    if valid_scores:
        # Suggestion de seuil simple (Médiane des scores positifs)
        import numpy as np
        suggested_threshold = float(np.median(valid_scores)) if valid_scores else 0.5
        
        # Initialiser le seuil utilisateur s'il n'existe pas encore
        if "user_threshold" not in st.session_state:
            st.session_state.user_threshold = suggested_threshold
            
        # Slider connecté au session_state
        threshold = st.sidebar.slider(
            "Seuil de suggestion (0-1)", 
            0.0, 1.0, 
            st.session_state.user_threshold, # Valeur par défaut (initiale)
            0.05, 
            key="user_threshold", # Persistance automatique
            help="Score au-dessus duquel l'IA suggère d'INCLURE."
        )
        
        st.sidebar.metric("Articles > Seuil", sum(1 for s in valid_scores if s >= threshold))
    else:
        st.sidebar.info("Attente des résultats de l'IA...")
        threshold = 0.5

    # Layout 2 colonnes : Liste (1/3) | Détails (2/3)
    col_list, col_details = st.columns([1, 2])
    
    with col_list:
        st.subheader(f"File d'attente ({len(articles)})")
        st.caption(f"Trié par pertinence pour : '{session_query}'")
        
        # Liste interactive avec score
        options = [a.id for a in articles]
        
        def format_article_label(art_id):
            art = next((a for a in articles if a.id == art_id), None)
            if not art: return ""
            
            score_display = "⏳" # Loader par défaut
            if art.relevance_score is not None:
                score_display = f"[{art.relevance_score:.2f}]"
            
            return f"{score_display} {art.title[:50]}..."
            
        selected_id = st.radio(
            "Sélectionner un article :",
            options=options,
            format_func=format_article_label,
            index=0,
            key="article_selector"
        )
        
        # Trouver l'article sélectionné
        current_article = next((a for a in articles if a.id == selected_id), None)

    with col_details:
        if current_article:
            # En-tête Article
            st.info(f"**{current_article.title}**")
            st.caption(f"{current_article.authors} | {current_article.year} | {current_article.source}")
            if current_article.doi:
                st.caption(f"DOI: {current_article.doi}")
            
            # Onglets de visualisation
            tab_abstract, tab_fulltext, tab_file = st.tabs(["Résumé", "Texte Complet", "Fichier"])
            
            with tab_abstract:
                if current_article.abstract:
                    st.markdown(f"### Abstract\n\n{current_article.abstract}")
                else:
                    st.warning("Pas d'abstract disponible.")
            
            with tab_fulltext:
                if current_article.full_text:
                    st.markdown(f"### Texte Complet ({len(current_article.full_text)} caractères)")
                    st.text_area("Contenu extrait", current_article.full_text, height=600)
                else:
                    st.warning("Texte complet non extrait.")
                    if current_article.pdf_path:
                        st.info("PDF disponible mais texte non extrait. Voir onglet 'Fichier'.")
                    else:
                        st.error("Pas de PDF téléchargé pour cet article.")
            
            with tab_file:
                st.markdown("### Accès au document")
                if current_article.link:
                    st.markdown(f"🔗 [Lien original]({current_article.link})")
                
                if current_article.pdf_path and os.path.exists(current_article.pdf_path):
                    st.success(f"PDF local disponible : `{current_article.pdf_path}`")
                    # On pourrait ajouter un bouton de téléchargement ici
                    with open(current_article.pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="Ouvrir/Télécharger le PDF",
                            data=pdf_file,
                            file_name=os.path.basename(current_article.pdf_path),
                            mime="application/pdf"
                        )
                else:
                    st.warning("Pas de fichier PDF local.")

            # Barre d'actions (Fixe en bas de la colonne ou après les onglets)
            st.divider()
            
            # --- SUGGESTION IA ---
            if current_article.relevance_score is not None:
                score = current_article.relevance_score
                if score >= threshold:
                    suggestion = "INCLURE"
                    color = "green"
                    reason_suggestion = f"Score élevé ({score:.2f} >= {threshold})"
                else:
                    suggestion = "EXCLURE"
                    color = "red"
                    reason_suggestion = f"Score faible ({score:.2f} < {threshold})"
                
                st.markdown(f"**Suggestion IA** : :{color}[{suggestion}] ({reason_suggestion})")
            else:
                st.info("⏳ Analyse IA en cours... Vous pouvez screener manuellement en attendant.")
            
            st.subheader("Décision")
            
            col_act1, col_act2, col_act3 = st.columns(3)
            
            with col_act1:
                if st.button("INCLURE", use_container_width=True, type="primary"):
                    save_decision(current_article.id, "SCREENED_IN")
            
            with col_act2:
                if st.button("INCERTAIN", use_container_width=True):
                    # On passe juste au suivant sans changer le statut (ou statut spécifique)
                    # Pour l'instant, on ne fait rien, l'utilisateur change manuellement
                    st.toast("Article marqué comme incertain (non modifié)")
            
            with col_act3:
                reason = st.selectbox("Raison du rejet", 
                                     ["Hors sujet", "Mauvaise population", "Mauvaise intervention", 
                                      "Mauvais type d'étude", "Langue incorrecte", "Pas de données", "Autre"],
                                     key=f"reason_{current_article.id}")
                
                if st.button("EXCLURE", use_container_width=True, type="secondary"):
                    save_decision(current_article.id, "EXCLUDED_SCREENING", reason)

    db.close()
