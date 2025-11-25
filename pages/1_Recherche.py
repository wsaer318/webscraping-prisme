import streamlit as st
from src.collection.google_scholar import GoogleScholarScraper
from src.database import get_db, Article

st.set_page_config(page_title="Identification", page_icon="🔍")

st.title("🔍 Identification des Articles")
st.markdown("Recherchez des articles via Google Scholar et ajoutez-les à votre base de données.")

# Search Form
with st.form("search_form"):
    query = st.text_input("Mots-clés de recherche", "proximal policy optimization ppo")
    num_results = st.slider("Nombre de résultats à récupérer", 5, 50, 10)
    submitted = st.form_submit_button("Lancer la recherche")

if submitted:
    scraper = GoogleScholarScraper()
    st.info(f"Recherche en cours pour : '{query}' ({num_results} résultats)...")
    
    # Progress bar
    progress_bar = st.progress(0)
    
    # Database session
    db = next(get_db())
    
    try:
        results = scraper.search(query, num_results=num_results, db=db)
        progress_bar.progress(100)
        
        if results:
            st.success(f"{len(results)} articles trouvés et traités !")
            
            # Display results
            for i, article in enumerate(results):
                with st.expander(f"{i+1}. {article['title']}"):
                    st.write(f"**Auteurs:** {article['authors']}")
                    st.write(f"**Lien:** {article['link']}")
                    st.write(f"**Abstract:** {article['abstract'][:200]}..." if article['abstract'] else "Pas de résumé")
        else:
            st.warning("Aucun résultat trouvé ou erreur lors du scraping.")
            
    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")
    finally:
        db.close()

# Show current database stats
db = next(get_db())
count = db.query(Article).filter(Article.status == "IDENTIFIED").count()
st.metric("Total Articles Identifiés (En attente de Screening)", count)
db.close()
