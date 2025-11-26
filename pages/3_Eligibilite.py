import streamlit as st
from src.database import get_db, Article

st.set_page_config(page_title="Eligibilité")

st.title("Éligibilité (Texte Complet)")
st.markdown("Validez les articles retenus après screening. Lisez le texte complet et décidez de l'inclusion finale.")

db = next(get_db())

# Fetch articles eligible for full-text review
articles = db.query(Article).filter(Article.status == "SCREENED_IN").all()

if not articles:
    st.info("Aucun article en attente d'éligibilité. Complétez l'étape de Screening d'abord.")
else:
    st.write(f"**Reste à valider : {len(articles)} articles**")
    
    # Process one article at a time
    article = articles[0]
    
    st.divider()
    st.subheader(article.title)
    st.write(f"**Auteurs:** {article.authors}")
    
    if article.link:
        st.markdown(f"🔗 **[Lire l'article complet]({article.link})**")
    else:
        st.warning("Pas de lien disponible.")
        
    st.info(f"**Rappel Abstract:**\n\n{article.abstract}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Inclure dans l'étude", use_container_width=True):
            article.status = "INCLUDED"
            db.commit()
            st.success("Article inclus !")
            st.rerun()
            
    with col2:
        reason = st.selectbox("Raison de l'exclusion", [
            "Pas de texte complet disponible",
            "Mauvaise population",
            "Mauvaise intervention",
            "Mauvais design d'étude",
            "Doublon non détecté",
            "Autre"
        ])
        if st.button("Exclure", use_container_width=True):
            article.status = "EXCLUDED_ELIGIBILITY"
            article.exclusion_reason = reason
            db.commit()
            st.error("Article exclu.")
            st.rerun()

db.close()
