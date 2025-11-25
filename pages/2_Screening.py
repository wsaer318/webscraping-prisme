import streamlit as st
from src.database import get_db, Article

st.set_page_config(page_title="Screening", page_icon="📝")

st.title("📝 Screening (Titre & Abstract)")
st.markdown("Filtrez les articles identifiés. Décidez de les inclure ou de les exclure sur base du titre et du résumé.")

db = next(get_db())

# Fetch articles to screen
articles = db.query(Article).filter(Article.status == "IDENTIFIED").all()

if not articles:
    st.info("Aucun article à screener pour le moment. Allez dans l'onglet 'Recherche' pour en ajouter.")
else:
    st.write(f"**Reste à screener : {len(articles)} articles**")
    
    # Process one article at a time (First one)
    article = articles[0]
    
    st.divider()
    st.subheader(article.title)
    st.write(f"**Auteurs:** {article.authors}")
    st.write(f"**Source:** {article.source}")
    st.info(f"**Abstract:**\n\n{article.abstract}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Inclure (Passer à l'éligibilité)", use_container_width=True):
            article.status = "SCREENED_IN"
            db.commit()
            st.rerun()
            
    with col2:
        if st.button("❌ Exclure", use_container_width=True):
            article.status = "EXCLUDED_SCREENING"
            article.exclusion_reason = "Screening Titre/Abstract"
            db.commit()
            st.rerun()

db.close()
