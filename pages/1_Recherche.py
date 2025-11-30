import streamlit as st
from src.collection.google_scholar import GoogleScholarScraper
from src.collection.arxiv_api import ArxivScraper
from src.collection.pubmed_api import PubMedScraper
from src.collection.crossref_api import CrossrefScraper
from src.database import get_db, Article, SearchSession, init_db
from src.advanced_sorting import AdvancedRanker
import threading
import pandas as pd
import re

st.set_page_config(page_title="Recherche", layout="wide")

# === DESIGN SYSTEM PREMIUM ===
from src.ui_utils import load_premium_css
load_premium_css()

st.title("🔍 Recherche d'Articles")

init_db()
db = next(get_db())

# Onglets pour séparer les deux modes
tab1, tab2 = st.tabs(["Mode Automatique", "Mode Avancé"])

# ==================== ONGLET 1: MODE AUTOMATIQUE ====================
with tab1:
    st.info("Mode recommandé : Recherche sur arXiv avec enrichissement automatique des métadonnées")
    
    # Input HORS du formulaire pour détecter les changements en temps réel
    default_query = st.session_state.get('active_session_query', "machine learning")
    query = st.text_input("Mots-clés de recherche", default_query, key="auto_query_input")
    
    # Compteur automatique quand la requête change
    if 'last_counted_query' not in st.session_state:
        st.session_state.last_counted_query = ""
    
    if query and len(query) >= 3 and query != st.session_state.last_counted_query:
        st.session_state.last_counted_query = query
        with st.spinner("Comptage..."):
            arxiv_counter = ArxivScraper()
            st.session_state.arxiv_result_count = arxiv_counter.get_result_count(query)
    
    # Affichage du compteur
    if 'arxiv_result_count' in st.session_state and st.session_state.arxiv_result_count > 0:
        st.caption(f"📊 ~{st.session_state.arxiv_result_count:,} articles disponibles sur arXiv")
    
    # MODE DE RECHERCHE - HORS DU FORMULAIRE pour changement en temps réel
    search_mode = st.radio(
        "Mode de recherche",
        ["Nombre fixe", "MAX (Tous les articles disponibles)"],
        help="Mode MAX : Récupère TOUS les articles disponibles (peut prendre du temps)"
    )
    
    # MÉTHODE DE SAISIE en mode Nombre fixe - HORS DU FORMULAIRE
    if search_mode == "Nombre fixe":
        input_method = st.radio(
            "Méthode de saisie",
            ["Slider", "Saisie manuelle"],
            horizontal=True,
            key="input_method_choice"
        )
    else:
        input_method = None
    
    with st.form("auto_search"):
        col1, col2 = st.columns(2)
        with col1:
            if search_mode == "Nombre fixe":
                if input_method == "Slider":
                    num_results = st.slider("Nombre de résultats arXiv", 5, 10000000, 100)
                else:
                    num_results = st.number_input(
                        "Nombre de résultats arXiv",
                        min_value=5,
                        max_value=10000000,
                        value=100,
                        step=10,
                        help="Tapez directement le nombre souhaité"
                    )
            else:
                # Mode MAX
                enable_cap = st.checkbox("Plafond de sécurité", value=False, 
                                        key="enable_cap_v2",  # Clé unique pour forcer refresh
                                        help="Limite le nombre max d'articles pour éviter les recherches trop longues")
                if enable_cap:
                    cap_value = st.number_input(
                        "Plafond maximum",
                        min_value=100,
                        max_value=10000000,
                        value=10000,
                        step=1000,
                        help="Nombre maximum d'articles à télécharger"
                    )
                    num_results = cap_value
                    st.caption(f"⚠️ Max {cap_value:,} articles")
                else:
                    num_results = 999999999  # Valeur "illimitée"
        
        with col2:
            enrich = st.checkbox("Enrichir avec PubMed/Crossref (DOIs)", value=True, 
                                help="Cherche les mêmes articles sur PubMed/Crossref pour obtenir les DOIs manquants")
        
        
        submitted = st.form_submit_button("Lancer la recherche automatique")
    
    if submitted:
        if not query:
            st.error("Veuillez entrer une requête de recherche.")
        else:
            progress = st.progress(0)
            status = st.empty()
            
            # Créer la session AVANT le téléchargement
            session = SearchSession(
                query=f"{query} [arXiv]",
                num_results=0,  # Sera mis à jour après
                successful_downloads=0,
                status='ACTIVE'
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            
            # Phase 1: arXiv (principal)
            status.text("Phase 1/3 : Recherche arXiv avec téléchargement PDF...")
            progress.progress(0.1)
            
            arxiv = ArxivScraper()
            arxiv_results, total_available = arxiv.search(query, max_results=num_results, session_id=session.id)
            
            # Afficher le total disponible si connu
            # Afficher le total disponible si connu
            if total_available is not None:
                st.info(f"📊 {total_available:,} articles disponibles sur arXiv pour cette requête")
            else:
                st.info("📊 Nombre total d'articles inconnu (API n'a pas retourné le total)")
            
            progress.progress(0.4)
            
            # Mettre à jour les stats de la session
            session.num_results = len(arxiv_results)
            session.successful_downloads = sum(1 for r in arxiv_results if r.get('extraction_status') == 'SUCCESS')
            db.commit()
            
            # Debug session ID
            # print(f"DEBUG: Session créée ID={session.id} pour {len(arxiv_results)} résultats")
            
            # PERSISTANCE SESSION
            st.session_state.active_session_id = session.id
            st.session_state.active_session_query = query
            st.toast(f"Session active : {query}")

            # Sauvegarder arXiv
            status.text("Phase 2/3 : Sauvegarde des articles arXiv...")
            for result in arxiv_results:
                # Déduplication par DOI ou titre
                existing_article = None
                
                if result.get('doi'):
                    existing_article = db.query(Article).filter(Article.doi == result['doi']).first()
                
                if not existing_article:
                    normalized_title = re.sub(r'[^\w\s]', '', result['title'].lower())
                    all_articles = db.query(Article).all()
                    for art in all_articles:
                        art_normalized = re.sub(r'[^\w\s]', '', art.title.lower())
                        if art_normalized == normalized_title:
                            existing_article = art
                            break
                
                if existing_article:
                    # Si l'article existe, on le lie à la NOUVELLE session pour qu'il apparaisse
                    existing_article.search_session_id = session.id
                    # On met à jour le statut si nécessaire (optionnel, on garde l'historique)
                    # existing_article.status = "IDENTIFIED" 
                else:
                    article = Article(
                        title=result['title'],
                        authors=result.get('authors'),
                        year=result.get('year'),
                        source=result.get('source'),
                        link=result.get('link'),
                        doi=result.get('doi'),
                        abstract=result.get('abstract'),
                        pdf_path=result.get('pdf_path'),
                        full_text=result.get('full_text'),
                        text_extraction_status=result.get('extraction_status', 'NOT_ATTEMPTED'),
                        extraction_method=result.get('extraction_method'),
                        status="IDENTIFIED",
                        search_session_id=session.id
                    )
                    db.add(article)
            
            db.commit()
            progress.progress(0.7)
            
            # Phase 3: Enrichissement (optionnel)
            enriched_count = 0
            if enrich:
                status.text("Phase 3/3 : Enrichissement PubMed/Crossref (DOIs)...")
                
                # Chercher seulement les articles arXiv sans DOI
                arxiv_no_doi = [r for r in arxiv_results if not r.get('doi')]
                
                if arxiv_no_doi:
                    st.caption(f"Recherche de DOIs pour {len(arxiv_no_doi)} articles...")
                    
                    # Crossref (meilleur pour les DOIs)
                    try:
                        crossref = CrossrefScraper()
                        crossref_results = crossref.search(query, max_results=20)
                        
                        # Matcher par titre
                        for arxiv_art in arxiv_no_doi:
                            arxiv_title_norm = re.sub(r'[^\w\s]', '', arxiv_art['title'].lower())
                            
                            for cross_art in crossref_results:
                                cross_title_norm = re.sub(r'[^\w\s]', '', cross_art['title'].lower())
                                
                                if arxiv_title_norm == cross_title_norm and cross_art.get('doi'):
                                    # Mettre à jour l'article dans la BDD
                                    db_article = db.query(Article).filter(
                                        Article.title == arxiv_art['title']
                                    ).first()
                                    
                                    if db_article and not db_article.doi:
                                        db_article.doi = cross_art['doi']
                                        enriched_count += 1
                                        break
                        
                        db.commit()
                    except Exception as e:
                        st.warning(f"Enrichissement Crossref échoué: {e}")
            
            progress.progress(1.0)
            status.text("Recherche terminée!")
            
            # Résumé
            st.success(f"**{len(arxiv_results)} articles arXiv** trouvés")
            
            # Statistiques détaillées
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("PDFs téléchargés", 
                        sum(1 for r in arxiv_results if r.get('pdf_path')))
            col_b.metric("Texte extrait", 
                        sum(1 for r in arxiv_results if r.get('extraction_status') == 'SUCCESS'))
            col_c.metric("DOIs enrichis", enriched_count)
            
            # Nombre total en BDD
            total_in_db = db.query(Article).count()
            col_d.metric("📚 Total en BDD", total_in_db, 
                        delta=f"+{len(arxiv_results)}" if len(arxiv_results) > 0 else "0")
            
            st.info("Consultez l'onglet 'Base de données' pour voir les articles.")
            
            # --- Lancement Analyse IA en arrière-plan ---
            st.toast("Lancement de l'analyse IA en arrière-plan...")
            
            def run_background_analysis(article_ids, query):
                ranker = AdvancedRanker()
                ranker.process_batch_and_update_db(article_ids, query)
            
            # Récupérer les IDs des nouveaux articles
            new_articles = db.query(Article).filter(Article.search_session_id == session.id).all()
            new_ids = [a.id for a in new_articles]
            
            if new_ids:
                # Lancer IA
                thread = threading.Thread(target=run_background_analysis, args=(new_ids, query))
                thread.start()
                
                
                # Lancer récupération PDFs manquants
                from src.pdf_retriever import auto_retrieve_missing_pdfs
                pdf_thread = threading.Thread(target=auto_retrieve_missing_pdfs, args=(session.id, 50))
                pdf_thread.start()
                
                # Lancer enrichissement citations (automatique)
                from src.enrichment import enrich_session_articles
                citation_thread = threading.Thread(target=enrich_session_articles, args=(session.id,))
                citation_thread.start()
                
                st.toast("Analyse IA + Récupération PDFs + Enrichissement citations lancés en arrière-plan !")


# ==================== ONGLET 2: MODE AVANCÉ ====================
with tab2:
    st.warning("Mode avancé : Sélection manuelle des sources (PubMed/Crossref ne fournissent pas de texte complet)")
    
    with st.form("advanced_search"):
        query_adv = st.text_input("Mots-clés de recherche", "machine learning", key="adv_query")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sources = st.multiselect(
                "Sources à interroger",
                ["Google Scholar", "arXiv", "PubMed", "Crossref"],
                default=["arXiv"],
                help="arXiv: texte complet | PubMed/Crossref: métadonnées seulement"
            )
        
        with col2:
            num_results_adv = st.slider("Nombre de résultats par source", 5, 10000000, 100, key="adv_num")
        
        submitted_adv = st.form_submit_button("Lancer la recherche avancée")
    
    if submitted_adv:
        if not query_adv or not sources:
            st.error("Veuillez entrer une requête et sélectionner au moins une source.")
        else:
            all_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            scrapers = {
                "Google Scholar": GoogleScholarScraper(),
                "arXiv": ArxivScraper(),
                "PubMed": PubMedScraper(),
                "Crossref": CrossrefScraper()
            }
            
            for idx, source_name in enumerate(sources):
                status_text.text(f"Interrogation de {source_name}...")
                progress_bar.progress((idx) / len(sources))
                
                try:
                    scraper = scrapers[source_name]
                    
                    if source_name == "Google Scholar":
                        results = scraper.search(query_adv, num_results=num_results_adv, db=db, review_only=True)
                    else:
                        results = scraper.search(query_adv, max_results=num_results_adv)
                        
                        session = SearchSession(
                            query=f"{query_adv} [{source_name}]",
                            num_results=len(results),
                            successful_downloads=0,
                            status='ACTIVE'
                        )
                        db.add(session)
                        db.commit()
                        db.refresh(session)
                        
                        for result in results:
                            is_duplicate = False
                            
                            if result.get('doi'):
                                exists = db.query(Article).filter(Article.doi == result['doi']).first()
                                if exists:
                                    is_duplicate = True
                            
                            if not is_duplicate:
                                normalized_title = re.sub(r'[^\w\s]', '', result['title'].lower())
                                all_articles = db.query(Article).all()
                                for art in all_articles:
                                    art_normalized = re.sub(r'[^\w\s]', '', art.title.lower())
                                    if art_normalized == normalized_title:
                                        is_duplicate = True
                                        break
                            
                            if not is_duplicate:
                                article = Article(
                                    title=result['title'],
                                    authors=result.get('authors'),
                                    year=result.get('year'),
                                    source=result.get('source'),
                                    link=result.get('link'),
                                    doi=result.get('doi'),
                                    abstract=result.get('abstract'),
                                    pdf_path=result.get('pdf_path'),
                                    full_text=result.get('full_text'),
                                    text_extraction_status=result.get('extraction_status', 'NOT_ATTEMPTED'),
                                    extraction_method=result.get('extraction_method'),
                                    status="IDENTIFIED",
                                    search_session_id=session.id
                                )
                                db.add(article)
                        
                        db.commit()
                    
                    all_results.append({
                        'source': source_name,
                        'count': len(results)
                    })
                    
                except Exception as e:
                    st.warning(f"Erreur avec {source_name}: {e}")
                    all_results.append({'source': source_name, 'count': 0})
            
            progress_bar.progress(1.0)
            status_text.text("Recherche terminée!")
            
            df_summary = pd.DataFrame(all_results)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
            
            total = sum([r['count'] for r in all_results])
            total = sum([r['count'] for r in all_results])
            st.metric("Total d'articles trouvés", total)

            # --- Lancement Analyse IA en arrière-plan ---
            if total > 0:
                st.toast("Lancement de l'analyse IA en arrière-plan...")
                
                def run_background_analysis(session_ids, query):
                    ranker = AdvancedRanker()
                    db_thread = next(get_db())
                    all_new_ids = []
                    for sess_id in session_ids:
                        arts = db_thread.query(Article).filter(Article.search_session_id == sess_id).all()
                        all_new_ids.extend([a.id for a in arts])
                    db_thread.close()
                    
                    if all_new_ids:
                        ranker.process_batch_and_update_db(all_new_ids, query)
                
                # Récupérer les IDs des sessions créées
                # Note: On doit récupérer les IDs des sessions créées dans la boucle
                # Simplification: On relance une requête pour trouver les sessions actives récentes ou on stocke les IDs
                # Ici on va faire simple : on lance l'analyse sur tous les articles IDENTIFIED sans score
                
                def run_global_analysis_on_missing():
                    ranker = AdvancedRanker()
                    db_thread = next(get_db())
                    # Articles sans score
                    articles_to_process = db_thread.query(Article).filter(
                        Article.status == "IDENTIFIED",
                        Article.relevance_score == None
                    ).all()
                    ids = [a.id for a in articles_to_process]
                    db_thread.close()
                    
                    if ids:
                        ranker.process_batch_and_update_db(ids, query_adv)

                thread = threading.Thread(target=run_global_analysis_on_missing)
                thread.start()
                
                # Lancer récupération PDFs manquants (pour toutes les sessions)
                from src.pdf_retriever import auto_retrieve_missing_pdfs
                pdf_thread = threading.Thread(target=auto_retrieve_missing_pdfs, args=(None, 100))
                pdf_thread.start()
                
                st.toast("✓ Analyse IA + Récupération PDFs démarrées en arrière-plan !")

db.close()
