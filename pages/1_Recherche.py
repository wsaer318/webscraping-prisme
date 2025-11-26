import streamlit as st
from src.collection.google_scholar import GoogleScholarScraper
from src.database import get_db, Article

st.set_page_config(page_title="Identification")

st.title("Identification des Articles")
st.markdown("Recherchez des articles via Google Scholar et ajoutez-les à votre base de données.")
st.info("Note : La recherche est configurée pour ne récupérer que les **articles de revue** (Review Articles).")

# Search Form
with st.form("search_form"):
    query = st.text_input("Mots-clés de recherche", "proximal policy optimization ppo")
    num_results = st.slider("Nombre de résultats à récupérer", 5, 50, 10)
    submitted = st.form_submit_button("Lancer la recherche")

if submitted:
    scraper = GoogleScholarScraper()
    st.info(f"Recherche en cours pour : '{query}' (Filtre: Revues uniquement, {num_results} résultats)...")
    st.caption(f"Debug Query Sent: {query}") # Debug info for user
    
    # Progress bar
    progress_bar = st.progress(0)
    
    # Database session
    db = next(get_db())
    
    try:
        # Recherche avec filtre "Review articles" activé par défaut (review_only=True)
        results = scraper.search(query, num_results=num_results, db=db, review_only=True)
        progress_bar.progress(100)
        
        if results:
            st.success(f"{len(results)} articles trouvés et traités !")
            
            # Affichage compact sous forme de tableau
            import pandas as pd
            df_results = pd.DataFrame([
                {
                    "Titre": r['title'],
                    "Auteurs": r['authors'],
                    "Source": r['source'],
                    "Lien": r['link']
                }
                for r in results
            ])
            st.dataframe(df_results, use_container_width=True, hide_index=True)
            
        else:
            st.warning("Aucun résultat trouvé ou erreur lors du scraping.")
            
    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")
    finally:
        db.close()

# Show current database stats
db = next(get_db())
count = db.query(Article).filter(Article.status == "IDENTIFIED").count()

st.divider()
col1, col2 = st.columns([3, 1])
with col1:
    st.metric("Total Articles Identifiés (En attente de Screening)", count)
with col2:
    if st.button("Effacer l'historique"):
        try:
            # Delete only IDENTIFIED articles (keep those already screened)
            db.query(Article).filter(Article.status == "IDENTIFIED").delete()
            db.commit()
            st.rerun()
        except Exception as e:
            st.error(f"Erreur: {e}")
db.close()
